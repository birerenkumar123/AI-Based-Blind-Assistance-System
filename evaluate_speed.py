"""Basic latency benchmark on a sample image (not an accuracy evaluation)."""
import argparse
import time
import cv2
from app import BlindAssistanceDetector

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to a representative image")
parser.add_argument("--runs", type=int, default=10)
parser.add_argument("--threshold", type=float, default=0.65)
args = parser.parse_args()

frame = cv2.imread(args.image)
if frame is None:
    raise FileNotFoundError(args.image)

detector = BlindAssistanceDetector(threshold=args.threshold)
# Warm-up includes model setup already; one prediction warms backend kernels.
detector.predict(frame)
times = []
for _ in range(args.runs):
    start = time.perf_counter()
    detections = detector.predict(frame)
    if detector.device.type == "cuda":
        import torch
        torch.cuda.synchronize()
    times.append(time.perf_counter() - start)

times.sort()
mean_ms = sum(times) / len(times) * 1000
p95_ms = times[min(len(times) - 1, int(0.95 * len(times)))] * 1000
print(f"Runs: {len(times)}")
print(f"Mean latency: {mean_ms:.1f} ms")
print(f"Approx. throughput: {1000 / mean_ms:.2f} frames/sec")
print(f"P95 latency: {p95_ms:.1f} ms")
print(f"Last-run detections: {len(detections)}")
print("Note: latency is not an accuracy metric. Test on your target device.")
