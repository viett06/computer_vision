import cv2

from app.ml_service.vision_core import (
    VehicleDetector,
    DistanceEstimator,
    SafetyEvaluator,
    DrawDetections,
)


class HighwaySafetyService:

    def __init__(self):

        self.detector = VehicleDetector()

        self.distance_estimator = (
            DistanceEstimator()
        )

        self.safety_evaluator = (
            SafetyEvaluator()
        )

        self.draw_detections = (
            DrawDetections()
        )

    def process_frame(self, image):

        detections = self.detector.detect(
            image
        )

        detections = self.distance_estimator.estimate(
            detections
        )

        detections = self.safety_evaluator.evaluate(
            detections
        )

        image = self.draw_detections.draw_detections(
            image,
            detections
        )

        return image, detections

    def handle_video(self, video_path: str):

        cap = cv2.VideoCapture(video_path)

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            image, detections = (
                self.process_frame(frame)
            )

            cv2.imshow(
                "Highway Safety",
                image
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()

        cv2.destroyAllWindows()