import cv2
import mediapipe as mp


class FullBodyPoseTracker:
    """
    Full-body pose estimation using MediaPipe Pose.

    Input:
        OpenCV BGR frame

    Output:
        Frame with body skeleton drawn on it
        Pose landmark data
    """

    def __init__(
        self,
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_draw = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame):
        """
        Process one OpenCV frame.

        Returns:
            annotated_frame, landmarks
        """

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.pose.process(rgb)

        annotated = frame.copy()

        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                annotated,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=2,
                    circle_radius=3,
                ),
                self.mp_draw.DrawingSpec(
                    color=(255, 0, 0),
                    thickness=2,
                    circle_radius=2,
                ),
            )

        return annotated, results.pose_landmarks

    def get_landmark_data(self, landmarks):
        """
        Convert MediaPipe landmarks into simple Python data.
        """

        if landmarks is None:
            return []

        data = []

        for index, landmark in enumerate(landmarks.landmark):
            data.append(
                {
                    "id": index,
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility,
                }
            )

        return data

    def close(self):
        self.pose.close()