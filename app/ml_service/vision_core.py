import cv2
from ultralytics import YOLO


class VehicleDetector:

    def __init__(
        self,
        model_path="yolov8n.pt",
        confidence_threshold=0.5
    ):
        print(
            f"[Init] Khởi tạo VehicleDetector "
            f"với model: {model_path}"
        )

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def detect(self, image):

        print(
            "\n[Trace: 1. VehicleDetector] "
            "Bắt đầu nhận diện xe..."
        )

        results = self.model(
            image,
            classes=[2, 3, 5, 7],
            verbose=False
        )

        detections = []

        for i, box in enumerate(results[0].boxes):

            confidence = float(box.conf[0])

            if confidence < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            bbox = [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ]

            class_id = int(box.cls[0])

            detections.append({
                "bbox": bbox,
                "confidence": confidence,
                "class_id": class_id
            })

            print(
                f"  -> [Xe {i + 1}] "
                f"Bounding Box: {bbox}"
            )

        print(
            f"[Trace: 1. VehicleDetector] "
            f"Hoàn thành. Tổng số xe: {len(detections)}"
        )

        return detections


class DistanceEstimator:

    def __init__(
        self,
        focal_length=800,
        real_car_width_m=1.8
    ):
        
        print(
            f"[Init] DistanceEstimator "
            f"(focal_length={focal_length}, "
            f"car_width={real_car_width_m}m)"
        )

        self.focal_length = focal_length
        self.real_car_width = real_car_width_m

    def calculate_distance(self, pixels_width):

        if pixels_width <= 0:
            return 999.0

        return (
            self.real_car_width * self.focal_length
        ) / pixels_width

    def estimate(self, detections):

        print(
            f"\n[Trace: 2. DistanceEstimator] "
            f"Bắt đầu đo khoảng cách cho "
            f"{len(detections)} đối tượng..."
        )

        for i, detection in enumerate(detections):

            x1, y1, x2, y2 = detection["bbox"]

            pixels_width = x2 - x1

            distance = self.calculate_distance(
                pixels_width
            )

            detection["pixels_width"] = pixels_width
            detection["distance_m"] = round(
                distance,
                2
            )

            print(
                f"  -> [Xe {i + 1}] "
                f"Pixel Width: {pixels_width}px | "
                f"Khoảng cách: "
                f"{detection['distance_m']}m"
            )

        return detections


class SafetyEvaluator:

    def __init__(self, safe_distance_m=55.0):

        print(
            f"[Init] SafetyEvaluator "
            f"(safe_distance={safe_distance_m}m)"
        )

        self.safe_distance_m = safe_distance_m

    def evaluate(self, detections):

        print(
            f"\n[Trace: 3. SafetyEvaluator] "
            f"Đánh giá an toàn "
            f"(Ngưỡng: {self.safe_distance_m}m)..."
        )

        for i, detection in enumerate(detections):

            distance = detection["distance_m"]

            is_dangerous = (
                distance < self.safe_distance_m
            )

            detection["is_dangerous"] = is_dangerous

            status = (
                "NGUY HIỂM"
                if is_dangerous
                else "AN TOÀN"
            )

            print(
                f"  -> [Xe {i + 1}] "
                f"Trạng thái: {status}"
            )

        return detections


class DrawDetections:

    def draw_detections(self, image, detections):

        for detection in detections:

            x1, y1, x2, y2 = detection["bbox"]

            distance = detection["distance_m"]
            confidence = detection["confidence"]
            is_dangerous = detection["is_dangerous"]

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            status = (
                "DANGER"
                if is_dangerous
                else "SAFE"
            )

            label = (
                f"{status} | "
                f"{distance:.1f}m | "
                f"conf={confidence:.2f}"
            )

            cv2.putText(
                image,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return image