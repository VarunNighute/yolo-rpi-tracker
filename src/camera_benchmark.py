import cv2
import time

CAMERA_ID = 0

WIDTH = 640
HEIGHT = 480
FPS = 30

print("Opening camera...")

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    raise SystemExit(1)

print("Camera opened.")

# Camera configuration
cap.set(
    cv2.CAP_PROP_FOURCC,
    cv2.VideoWriter_fourcc("M", "J", "P", "G")
)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

# Try a small V4L2 buffer queue
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
actual_fps = cap.get(cv2.CAP_PROP_FPS)
actual_buffer = int(cap.get(cv2.CAP_PROP_BUFFERSIZE))

print()
print("Camera configuration:")
print(f"Resolution : {actual_width}x{actual_height}")
print(f"FPS        : {actual_fps:.2f}")
print(f"Buffer     : {actual_buffer}")
print()

time.sleep(1.0)

# Warm up
print("Warming up camera...")

for _ in range(30):
    if not cap.grab():
        print("ERROR: grab() failed during warm-up.")
        cap.release()
        raise SystemExit(1)

print("Starting benchmark...")
print()

frame_count = 0
start_time = time.perf_counter()

intervals = []

last_time = time.perf_counter()

while frame_count < 300:

    success = cap.grab()

    now = time.perf_counter()

    if not success:
        print("ERROR: grab() failed.")
        break

    interval_ms = (now - last_time) * 1000.0
    intervals.append(interval_ms)

    last_time = now

    frame_count += 1

end_time = time.perf_counter()

cap.release()

total_time = end_time - start_time

if frame_count > 0:

    average_fps = frame_count / total_time
    average_interval = sum(intervals[1:]) / len(intervals[1:])

    print()
    print("========== RESULTS ==========")
    print(f"Frames grabbed : {frame_count}")
    print(f"Total time     : {total_time:.2f} s")
    print(f"OpenCV FPS     : {average_fps:.2f}")
    print(f"Average interval : {average_interval:.2f} ms")
    print(f"Min interval   : {min(intervals[1:]):.2f} ms")
    print(f"Max interval   : {max(intervals[1:]):.2f} ms")
    print("=============================")
