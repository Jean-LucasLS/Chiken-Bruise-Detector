"""
Frame Capture & Bruise Annotation Pipeline
==========================================
Reads a video file, samples frames at a configurable time interval,
runs bruise detection on each sampled frame, and writes the annotated
images to an output directory.

Usage:
    python frame_capture.py

    Press 'q' during playback to stop early.
"""

import logging
import os

import cv2

from bruiser_detection import detect_bruises

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
VIDEO_PATH:       str   = "videos/Frango1.mp4"
OUTPUT_DIR:       str   = "frames"
CAPTURE_INTERVAL: float = 1.0  # Seconds between captured frames

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def process_video(video_path: str, output_dir: str, capture_interval: float) -> None:
    """
    Sample frames from a video, annotate bruise regions, and save to disk.

    Args:
        video_path:       Path to the source video file.
        output_dir:       Directory where annotated frames are saved.
        capture_interval: Time in seconds between sampled frames.
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video: %s", video_path)
        return

    fps        = cap.get(cv2.CAP_PROP_FPS)
    frame_step = max(1, int(fps * capture_interval))
    wait_ms    = max(1, int(1000 / fps))
    frame_idx  = 0

    logger.info(
        "Processing '%s' | %.2f fps | sampling every %d frames",
        video_path, fps, frame_step,
    )

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                annotated   = detect_bruises(frame)
                timestamp   = int(frame_idx / fps)
                output_path = os.path.join(output_dir, f"frame_{timestamp:04d}.jpg")
                cv2.imwrite(output_path, annotated)
                logger.info("Saved  %s", output_path)

            cv2.imshow("Bruise Detection", frame)
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                logger.info("Playback interrupted by user.")
                break

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    process_video(VIDEO_PATH, OUTPUT_DIR, CAPTURE_INTERVAL)
