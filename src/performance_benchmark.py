import cv2
import time
import statistics
import torch
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

CAMERA_ID = 0

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CAMERA_FPS = 30

MODEL_PATH = "yolo11n.pt"

CONFIDENCE_THRESHOLD = 0.5

WARMUP_FRAMES = 30
TEST_FRAMES = 300


# ============================================================
# Print System Information
# ============================================================

print("=" * 60)
print("YOLO REAL-TIME PERFORMANCE BENCHMARK")
print("=" * 60)

print()

print("PyTorch version:")
print(torch.__version__)

print()

print("CUDA available:")
print(torch.cuda.is_available())

if torch.cuda.is_available():

    print("GPU:")
    print(torch.cuda.get_device_name(0))

    print("CUDA version:")
    print(torch.version.cuda)

print()


# ============================================================
# Load Model
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded.")

print()


# ============================================================
# Open Camera
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_ID,
    cv2.CAP_V4L2
)

cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc(*"MJPG")
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    CAMERA_FPS
)

cap.set(
    cv2.CAP_PROP_BUFFERSIZE,
    1
)


if not cap.isOpened():

    print("ERROR: Could not open camera.")

    exit()


print("Camera opened.")

print(
    f"Resolution: "
    f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}"
    f"x"
    f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
)

print(
    f"Requested FPS: "
    f"{cap.get(cv2.CAP_PROP_FPS):.1f}"
)

print()


# ============================================================
# Warmup
# ============================================================

print(
    f"Warming up for {WARMUP_FRAMES} frames..."
)

for _ in range(WARMUP_FRAMES):

    ret, frame = cap.read()

    if not ret:

        print("Camera read failed during warmup.")

        cap.release()

        exit()


    model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )


print("Warmup complete.")

print()


# ============================================================
# Benchmark Variables
# ============================================================

read_times = []
inference_times = []
total_times = []

detection_counts = []


# ============================================================
# Benchmark
# ============================================================

print(
    f"Running benchmark for "
    f"{TEST_FRAMES} frames..."
)

print()


benchmark_start = time.perf_counter()


for frame_number in range(TEST_FRAMES):

    # --------------------------------------------------------
    # Camera read
    # --------------------------------------------------------

    read_start = time.perf_counter()

    ret, frame = cap.read()

    read_end = time.perf_counter()


    if not ret:

        print(
            f"Camera read failed at frame "
            f"{frame_number}"
        )

        break


    read_time = (
        read_end - read_start
    ) * 1000


    # --------------------------------------------------------
    # YOLO inference
    # --------------------------------------------------------

    inference_start = time.perf_counter()


    results = model(
        frame,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )


    # Force CUDA synchronization so
    # timing includes GPU work.
    if torch.cuda.is_available():

        torch.cuda.synchronize()


    inference_end = time.perf_counter()


    inference_time = (
        inference_end - inference_start
    ) * 1000


    # --------------------------------------------------------
    # Detection count
    # --------------------------------------------------------

    detection_count = 0


    for result in results:

        if result.boxes is not None:

            detection_count += len(
                result.boxes
            )


    # --------------------------------------------------------
    # Total frame time
    # --------------------------------------------------------

    total_time = (
        read_time + inference_time
    )


    read_times.append(read_time)

    inference_times.append(
        inference_time
    )

    total_times.append(total_time)

    detection_counts.append(
        detection_count
    )


benchmark_end = time.perf_counter()


# ============================================================
# Cleanup
# ============================================================

cap.release()


# ============================================================
# Calculate Results
# ============================================================

actual_frames = len(total_times)

benchmark_duration = (
    benchmark_end - benchmark_start
)


overall_fps = (
    actual_frames /
    benchmark_duration
)


average_read = statistics.mean(
    read_times
)

average_inference = statistics.mean(
    inference_times
)

average_total = statistics.mean(
    total_times
)


min_inference = min(
    inference_times
)

max_inference = max(
    inference_times
)


pipeline_fps = (
    1000 /
    average_total
)


average_detections = statistics.mean(
    detection_counts
)


# ============================================================
# Print Results
# ============================================================

print()
print("=" * 60)
print("BENCHMARK RESULTS")
print("=" * 60)

print()

print(
    f"Frames tested:       {actual_frames}"
)

print(
    f"Benchmark duration:  "
    f"{benchmark_duration:.2f} s"
)

print()

print(
    f"Camera read:         "
    f"{average_read:.2f} ms"
)

print(
    f"YOLO inference:      "
    f"{average_inference:.2f} ms"
)

print(
    f"Total frame time:    "
    f"{average_total:.2f} ms"
)

print()

print(
    f"Pipeline FPS:        "
    f"{pipeline_fps:.2f}"
)

print(
    f"Overall FPS:         "
    f"{overall_fps:.2f}"
)

print()

print(
    f"Min inference:       "
    f"{min_inference:.2f} ms"
)

print(
    f"Max inference:       "
    f"{max_inference:.2f} ms"
)

print()

print(
    f"Average detections:  "
    f"{average_detections:.2f}"
)

print()

print("=" * 60)
print("BENCHMARK COMPLETE")
print("=" * 60)
