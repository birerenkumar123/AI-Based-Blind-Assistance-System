"""
AI-Based Blind Assistance System (educational prototype)
- PyTorch + TorchVision Faster R-CNN pretrained on COCO
- Webcam or still-image inference
- Image-relative location narration (left/center/right)
- Optional offline text-to-speech

Safety: detections are not guaranteed, distance is not measured, and this is
not a navigation or collision-avoidance device.
"""
import argparse
import threading
import time
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.transforms.functional import to_tensor

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


class Speaker:
    """Non-blocking speech queue with a small repeat-suppression window."""
    def __init__(self, enabled=True, cooldown=3.0):
        self.enabled = enabled and pyttsx3 is not None
        self.cooldown = cooldown
        self.last_text = ""
        self.last_time = 0.0
        self.queue = []
        self.lock = threading.Lock()
        self.thread = None
        if self.enabled:
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()

    def say(self, text):
        if not self.enabled:
            return
        now = time.time()
        if text == self.last_text and now - self.last_time < self.cooldown:
            return
        self.last_text, self.last_time = text, now
        with self.lock:
            self.queue.append(text)

    def _worker(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            while True:
                text = None
                with self.lock:
                    if self.queue:
                        text = self.queue.pop(0)
                        self.queue.clear()  # keep narration current
                if text:
                    engine.say(text)
                    engine.runAndWait()
                else:
                    time.sleep(0.08)
        except Exception as exc:
            print(f"[Audio] Text-to-speech unavailable: {exc}")


class BlindAssistanceDetector:
    def __init__(self, threshold=0.65, device=None):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.categories = self.weights.meta["categories"]
        self.model = fasterrcnn_resnet50_fpn(weights=self.weights)
        self.model.to(self.device).eval()
        self.threshold = threshold
        print(f"[Model] Device: {self.device}")
        print("[Model] COCO-pretrained Faster R-CNN loaded.")

    @torch.inference_mode()
    def predict(self, bgr_frame, threshold=None):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        tensor = to_tensor(image).to(self.device)  # float tensor in [0, 1]
        output = self.model([tensor])[0]
        detections = []
        for box, label, score in zip(
            output["boxes"].cpu(),
            output["labels"].cpu(),
            output["scores"].cpu(),
        ):
            score = float(score)
            if score < (self.threshold if threshold is None else threshold):
                continue
            x1, y1, x2, y2 = [int(v) for v in box.tolist()]
            label_id = int(label)
            name = self.categories[label_id] if label_id < len(self.categories) else "object"
            center_x = (x1 + x2) / 2.0
            width = bgr_frame.shape[1]
            if center_x < width / 3:
                position = "left"
            elif center_x > 2 * width / 3:
                position = "right"
            else:
                position = "center"
            detections.append({
                "box": (x1, y1, x2, y2),
                "label": name,
                "score": score,
                "position": position,
            })
        return detections

    @staticmethod
    def describe(detections, max_objects=3):
        if not detections:
            return "No familiar objects detected."
        # Prioritize high confidence, concise descriptions
        detections = sorted(detections, key=lambda d: d["score"], reverse=True)
        parts = []
        for d in detections[:max_objects]:
            parts.append(f"{d['label']} on your {d['position']}")
        return "Detected " + ", ".join(parts) + "."

    @staticmethod
    def draw(frame, detections):
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            text = f"{d['label']} {d['score']:.2f} ({d['position']})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 220, 30), 2)
            cv2.putText(
                frame, text, (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 220, 30), 2,
                cv2.LINE_AA,
            )
        return frame


def run_image(detector, image_path, output_path, speaker):
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    start = time.perf_counter()
    detections = detector.predict(frame)
    elapsed = time.perf_counter() - start
    print(detector.describe(detections))
    print(f"Detections: {len(detections)} | Inference: {elapsed:.3f}s")
    speaker.say(detector.describe(detections))
    annotated = detector.draw(frame.copy(), detections)
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        print(f"Saved annotated image: {output_path}")
    cv2.imshow("Blind Assistance - Image", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_webcam(detector, camera_index, speaker, speak_every=3.0):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {camera_index}. "
            "Try --camera 1 or check camera permissions."
        )
    last_speech = 0.0
    last_description = ""
    print("[Camera] Press q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[Camera] Frame read failed.")
                break
            start = time.perf_counter()
            detections = detector.predict(frame)
            elapsed = time.perf_counter() - start
            description = detector.describe(detections)
            now = time.time()
            if description != last_description or now - last_speech >= speak_every:
                speaker.say(description)
                last_description = description
                last_speech = now
            annotated = detector.draw(frame.copy(), detections)
            cv2.putText(
                annotated, f"Inference: {elapsed:.2f}s | q = quit",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2,
                cv2.LINE_AA,
            )
            cv2.imshow("AI Blind Assistance Prototype", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="PyTorch blind-assistance prototype")
    parser.add_argument("--image", help="Path to an image; omit to use webcam")
    parser.add_argument("--output", default="outputs/annotated.jpg", help="Annotated image output path")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.65, help="Detection confidence threshold")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Force inference device")
    parser.add_argument("--no-speech", action="store_true", help="Disable text-to-speech")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    detector = BlindAssistanceDetector(args.threshold, args.device)
    speaker = Speaker(enabled=not args.no_speech)
    if not speaker.enabled and not args.no_speech:
        print("[Audio] pyttsx3 unavailable; continuing without speech.")
    if args.image:
        run_image(detector, args.image, args.output, speaker)
    else:
        run_webcam(detector, args.camera, speaker)


if __name__ == "__main__":
    main()
