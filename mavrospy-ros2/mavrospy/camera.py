#!/usr/bin/env python3
"""
Drone camera + AprilTag detection node.

MQTT topics used by this script (drone1):
  Publishes:
    indago/drone1/camera/status    camera ready / heartbeat / offline (retained, with LWT)
    indago/drone1/landing/status   AprilTag detected / lost
    indago/drone1/error            camera / detector errors
  Subscribes:
    indago/command/drone1/land     raises landing-status publish rate
    indago/command/drone1/arm      resets landing mode
    indago/command/drone1/disarm   resets landing mode
"""

import json
import sys
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from picamera2 import Picamera2

try:
    from pupil_apriltags import Detector
except ImportError:
    Detector = None

# -------------------------
# Camera settings
# -------------------------
WIDTH = 1280
HEIGHT = 720
FPS = 30

# -------------------------
# AprilTag settings
# -------------------------
TAG_FAMILY = "tag36h11"
QUAD_DECIMATE = 2.0        # 2.0 = much faster on a Pi; use 1.0 for longer range
TAG_LOST_TIMEOUT = 1.0     # seconds without a tag before "tag_lost" is sent

# Optional pose estimation (requires calibrating the camera).
# Set CAMERA_PARAMS = (fx, fy, cx, cy) and TAG_SIZE_M = printed tag edge in metres.
CAMERA_PARAMS = None
TAG_SIZE_M = 0.16

# -------------------------
# MQTT settings
# -------------------------
MQTT_BROKER = "10.0.0.174"
MQTT_PORT = 1883
DEVICE = "drone1"

TOPIC_CAMERA_STATUS = f"indago/{DEVICE}/camera/status"
TOPIC_LANDING_STATUS = f"indago/{DEVICE}/landing/status"
TOPIC_ERROR = f"indago/{DEVICE}/error"
TOPIC_CMD_LAND = f"indago/command/{DEVICE}/land"
TOPIC_CMD_ARM = f"indago/command/{DEVICE}/arm"
TOPIC_CMD_DISARM = f"indago/command/{DEVICE}/disarm"

HEARTBEAT_INTERVAL = 5.0        # seconds
PUBLISH_INTERVAL_IDLE = 0.5     # landing/status rate normally (2 Hz)
PUBLISH_INTERVAL_LANDING = 0.1  # landing/status rate after a land command (10 Hz)

landing_mode = False


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def build_payload(**fields):
    payload = {"device": DEVICE, "component": "camera", "timestamp": now_iso()}
    payload.update(fields)
    return json.dumps(payload)


# -------------------------
# MQTT
# -------------------------
def on_connect(client, userdata, flags, reason_code, properties=None):
    # Works for both paho v1 (rc int) and v2 (ReasonCode)
    print(f"MQTT connected ({reason_code})")
    client.subscribe([(TOPIC_CMD_LAND, 1), (TOPIC_CMD_ARM, 1), (TOPIC_CMD_DISARM, 1)])


def on_message(client, userdata, msg):
    global landing_mode
    if msg.topic == TOPIC_CMD_LAND:
        landing_mode = True
        print("Land command received: landing mode ON")
    elif msg.topic in (TOPIC_CMD_ARM, TOPIC_CMD_DISARM):
        landing_mode = False
        print("Arm/disarm received: landing mode OFF")


def make_client():
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    except AttributeError:
        client = mqtt.Client()

    # If the Pi dies or loses power, the broker publishes this for us.
    client.will_set(
        TOPIC_CAMERA_STATUS,
        build_payload(status="offline", message="Camera node disconnected unexpectedly"),
        qos=1,
        retain=True,
    )
    client.on_connect = on_connect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=10)

    client.connect(MQTT_BROKER, MQTT_PORT, keepalive=30)
    client.loop_start()  # network loop in background thread
    return client


def publish(client, topic, payload, qos=0, retain=False):
    if client is None:
        return
    try:
        client.publish(topic, payload, qos=qos, retain=retain)
    except Exception as error:
        print(f"MQTT publish error: {error}")


def report_error(client, message):
    print(f"ERROR: {message}")
    publish(client, TOPIC_ERROR, build_payload(status="error", message=message), qos=1)
    publish(client, TOPIC_CAMERA_STATUS,
            build_payload(status="error", message=message), qos=1, retain=True)


# -------------------------
# Main
# -------------------------
def main():
    if Detector is None:
        sys.exit("pupil-apriltags is not installed. Run: pip install pupil-apriltags")

    client = None
    try:
        client = make_client()
    except Exception as error:
        print(f"MQTT unavailable, continuing without it: {error}")

    camera = Picamera2()
    frame_us = int(1_000_000 / FPS)

    # YUV420: the top HEIGHT rows are the Y (grayscale) plane, so no color
    # conversion is needed.
    config = camera.create_video_configuration(
        main={"size": (WIDTH, HEIGHT), "format": "YUV420"},
        controls={"FrameDurationLimits": (frame_us, frame_us)},
    )
    camera.configure(config)

    detector = Detector(
        families=TAG_FAMILY,
        nthreads=3,
        quad_decimate=QUAD_DECIMATE,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )
    use_pose = CAMERA_PARAMS is not None

    try:
        print("Starting drone camera...")
        camera.start()
        time.sleep(2)

        yuv = camera.capture_array()
        if yuv is None:
            raise RuntimeError("Camera started but no frame was received.")

        print(f"Camera started: {WIDTH}x{HEIGHT} @ {FPS} FPS")
        publish(client, TOPIC_CAMERA_STATUS,
                build_payload(status="ready", message="Drone camera successfully started.",
                              width=WIDTH, height=HEIGHT, fps=FPS),
                qos=1, retain=True)

        last_heartbeat = 0.0
        last_landing_publish = 0.0
        last_seen = 0.0
        tag_visible = False
        frames = 0
        fps_timer = time.monotonic()
        measured_fps = 0.0

        while True:
            yuv = camera.capture_array()
            gray = yuv[:HEIGHT, :WIDTH]
            now = time.monotonic()

            # Detection (optionally with pose)
            if use_pose:
                detections = detector.detect(
                    gray, estimate_tag_pose=True,
                    camera_params=CAMERA_PARAMS, tag_size=TAG_SIZE_M)
            else:
                detections = detector.detect(gray)

            # Pick the most confident tag
            best = max(detections, key=lambda d: d.decision_margin) if detections else None

            interval = PUBLISH_INTERVAL_LANDING if landing_mode else PUBLISH_INTERVAL_IDLE

            if best is not None:
                last_seen = now
                cx, cy = float(best.center[0]), float(best.center[1])

                # Publish immediately on first sighting, then at the throttled rate
                if not tag_visible or (now - last_landing_publish) >= interval:
                    fields = {
                        "status": "tag_detected",
                        "tag_id": int(best.tag_id),
                        "center": [cx, cy],
                        # -1..1, where 0 means centered in the frame (useful for alignment)
                        "offset_x": (cx - WIDTH / 2) / (WIDTH / 2),
                        "offset_y": (cy - HEIGHT / 2) / (HEIGHT / 2),
                        "margin": float(best.decision_margin),
                        "landing_mode": landing_mode,
                    }
                    if use_pose and best.pose_t is not None:
                        fields["pose_t_m"] = [float(v) for v in best.pose_t.flatten()]
                    publish(client, TOPIC_LANDING_STATUS, build_payload(**fields))
                    last_landing_publish = now
                    print(f"Tag {best.tag_id} at ({cx:.0f}, {cy:.0f})")
                tag_visible = True

            elif tag_visible and (now - last_seen) > TAG_LOST_TIMEOUT:
                tag_visible = False
                publish(client, TOPIC_LANDING_STATUS,
                        build_payload(status="tag_lost", landing_mode=landing_mode),
                        qos=1)
                print("Tag lost")

            # FPS measurement
            frames += 1
            if now - fps_timer >= 2.0:
                measured_fps = frames / (now - fps_timer)
                frames = 0
                fps_timer = now

            # Heartbeat so the server knows the camera is alive
            if now - last_heartbeat >= HEARTBEAT_INTERVAL:
                publish(client, TOPIC_CAMERA_STATUS,
                        build_payload(status="ok", message="heartbeat",
                                      detect_fps=round(measured_fps, 1),
                                      tag_visible=tag_visible,
                                      landing_mode=landing_mode),
                        retain=True)
                last_heartbeat = now

    except KeyboardInterrupt:
        print("\nStopping camera...")

    except Exception as error:
        report_error(client, f"Camera error: {error}")

    finally:
        try:
            camera.stop()
        except Exception:
            pass

        if client is not None:
            publish(client, TOPIC_CAMERA_STATUS,
                    build_payload(status="offline", message="Camera stopped."),
                    qos=1, retain=True)
            time.sleep(0.3)  # let the final message flush
            client.loop_stop()
            client.disconnect()

        print("Camera stopped.")


if __name__ == "__main__":
    main()