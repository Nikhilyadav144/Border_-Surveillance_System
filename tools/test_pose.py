import cv2
import sys
from pathlib import Path

# Allow importing from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from core.pose_tracker import FullBodyPoseTracker


VIDEO_PATH = PROJECT_ROOT / "data" / "input" / "demo_border.mp4"


def main():

    if not VIDEO_PATH.exists():
        print(f"ERROR: Video not found:")
        print(VIDEO_PATH)
        return

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("ERROR: Could not open video.")
        return

    pose_tracker = FullBodyPoseTracker(
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    print("CAM-01 full-body pose tracking started.")
    print("Press Q to quit.")

    while True:

        success, frame = cap.read()

        if not success:
            print("Video ended.")
            break

        annotated, landmarks = pose_tracker.process(frame)

        # Number of detected body landmarks
        landmark_count = 0

        if landmarks:
            landmark_count = len(landmarks.landmark)

        # Display information
        cv2.putText(
            annotated,
            "CAM-01 | FULL BODY TRACKING",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated,
            f"Pose landmarks: {landmark_count}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.imshow("IBVAP - CAM-01", annotated)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    pose_tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()