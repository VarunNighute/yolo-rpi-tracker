import cv2
import time
import threading
import math
from collections import deque

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

STREAM_URL = "http://192.168.1.2:8080/video"

YOLO_MODEL = "yolo11n.pt"
YOLO_SIZE = 320

CONFIDENCE_THRESHOLD = 0.5

# Run YOLO once every N captured frames.
DETECTION_INTERVAL = 8

# Tracking
MAX_MATCH_DISTANCE = 120
MAX_MISSED_DETECTIONS = 5

# Prediction limits
MAX_PREDICTION_STEP = 25

# Image dimensions from the laptop camera stream.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Status output
STATUS_INTERVAL = 20

# Camera buffer
BUFFER_SIZE = 1


# ============================================================
# ASYNCHRONOUS CAMERA
# ============================================================

class LatestFrameCamera:

    def __init__(self, source):

        self.source = source

        self.cap = None

        self.lock = threading.Lock()
        self.latest_frame = None

        self.running = False
        self.thread = None

        self.frames_received = 0
        self.start_time = None

    def start(self):

        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():

            raise RuntimeError(
                f"Could not open camera stream: {self.source}"
            )

        self.cap.set(
            cv2.CAP_PROP_BUFFERSIZE,
            BUFFER_SIZE
        )

        self.running = True
        self.start_time = time.perf_counter()

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )

        self.thread.start()

        return self

    def _capture_loop(self):

        while self.running:

            ok, frame = self.cap.read()

            if not ok:

                time.sleep(0.01)

                continue

            with self.lock:

                self.latest_frame = frame

            self.frames_received += 1

    def get_latest_frame(self):

        with self.lock:

            if self.latest_frame is None:

                return None

            return self.latest_frame.copy()

    def get_fps(self):

        if self.start_time is None:

            return 0.0

        elapsed = (
            time.perf_counter()
            -
            self.start_time
        )

        if elapsed <= 0:

            return 0.0

        return (
            self.frames_received
            /
            elapsed
        )

    def stop(self):

        self.running = False

        if self.thread is not None:

            self.thread.join(
                timeout=2.0
            )

        if self.cap is not None:

            self.cap.release()


# ============================================================
# TRACKED OBJECT
# ============================================================

class TrackedObject:

    def __init__(
        self,
        object_id,
        class_id,
        class_name,
        centroid,
        bbox
    ):

        self.id = object_id

        self.class_id = class_id
        self.class_name = class_name

        # Most recent reliable YOLO position.
        self.detected_centroid = centroid

        self.detected_bbox = bbox

        # Current displayed/predicted position.
        self.centroid = centroid

        self.bbox = bbox

        # Velocity measured ONLY from consecutive YOLO
        # detections.
        self.velocity_x = 0.0
        self.velocity_y = 0.0

        # Number of YOLO cycles missed.
        self.missed_detections = 0

        self.age = 1

    def update_from_detection(
        self,
        centroid,
        bbox
    ):

        old_x, old_y = (
            self.detected_centroid
        )

        new_x, new_y = centroid

        # Calculate velocity only from reliable YOLO
        # measurements.
        raw_velocity_x = (
            new_x - old_x
        )

        raw_velocity_y = (
            new_y - old_y
        )

        # Limit velocity so one bad detection cannot create
        # enormous prediction movement.
        self.velocity_x = max(
            -MAX_PREDICTION_STEP,
            min(
                MAX_PREDICTION_STEP,
                raw_velocity_x
            )
        )

        self.velocity_y = max(
            -MAX_PREDICTION_STEP,
            min(
                MAX_PREDICTION_STEP,
                raw_velocity_y
            )
        )

        # YOLO is authoritative.
        self.detected_centroid = centroid
        self.detected_bbox = bbox

        self.centroid = centroid
        self.bbox = bbox

        self.missed_detections = 0

        self.age += 1

    def predict(self):

        """
        Conservative prediction.

        We predict from the LAST RELIABLE YOLO position.
        We do not recursively predict from the previous
        prediction. This prevents runaway drift.
        """

        dx = (
            self.velocity_x
            *
            self.missed_detections
        )

        dy = (
            self.velocity_y
            *
            self.missed_detections
        )

        # Hard-limit total prediction displacement.
        max_total_prediction = (
            MAX_PREDICTION_STEP
            *
            self.missed_detections
        )

        dx = max(
            -max_total_prediction,
            min(
                max_total_prediction,
                dx
            )
        )

        dy = max(
            -max_total_prediction,
            min(
                max_total_prediction,
                dy
            )
        )

        x, y = self.detected_centroid

        predicted_x = int(
            x + dx
        )

        predicted_y = int(
            y + dy
        )

        # Never allow centroid outside image.
        predicted_x = max(
            0,
            min(
                FRAME_WIDTH - 1,
                predicted_x
            )
        )

        predicted_y = max(
            0,
            min(
                FRAME_HEIGHT - 1,
                predicted_y
            )
        )

        self.centroid = (
            predicted_x,
            predicted_y
        )

        # Predict bounding box from the same reliable YOLO
        # bounding box.
        x1, y1, x2, y2 = (
            self.detected_bbox
        )

        predicted_x1 = int(
            x1 + dx
        )

        predicted_y1 = int(
            y1 + dy
        )

        predicted_x2 = int(
            x2 + dx
        )

        predicted_y2 = int(
            y2 + dy
        )

        # Keep bbox inside image.
        predicted_x1 = max(
            0,
            min(
                FRAME_WIDTH - 1,
                predicted_x1
            )
        )

        predicted_y1 = max(
            0,
            min(
                FRAME_HEIGHT - 1,
                predicted_y1
            )
        )

        predicted_x2 = max(
            0,
            min(
                FRAME_WIDTH - 1,
                predicted_x2
            )
        )

        predicted_y2 = max(
            0,
            min(
                FRAME_HEIGHT - 1,
                predicted_y2
            )
        )

        self.bbox = (
            predicted_x1,
            predicted_y1,
            predicted_x2,
            predicted_y2
        )


# ============================================================
# CENTROID TRACKER
# ============================================================

class CentroidTracker:

    def __init__(
        self,
        max_distance=120,
        max_missed=5
    ):

        self.max_distance = (
            max_distance
        )

        self.max_missed = (
            max_missed
        )

        self.next_id = 0

        self.objects = {}

    def _register(
        self,
        class_id,
        class_name,
        centroid,
        bbox
    ):

        object_id = self.next_id

        self.next_id += 1

        self.objects[object_id] = (
            TrackedObject(
                object_id,
                class_id,
                class_name,
                centroid,
                bbox
            )
        )

    def _deregister(
        self,
        object_id
    ):

        if object_id in self.objects:

            del self.objects[
                object_id
            ]

    @staticmethod
    def _distance(
        point_a,
        point_b
    ):

        return math.sqrt(
            (
                point_a[0]
                -
                point_b[0]
            ) ** 2
            +
            (
                point_a[1]
                -
                point_b[1]
            ) ** 2
        )

    def update(
        self,
        detections
    ):

        """
        Match new YOLO detections against the current
        predicted object positions.
        """

        # ----------------------------------------------------
        # No detections
        # ----------------------------------------------------

        if len(detections) == 0:

            for object_id in list(
                self.objects.keys()
            ):

                obj = self.objects[
                    object_id
                ]

                obj.missed_detections += 1

                if (
                    obj.missed_detections
                    >
                    self.max_missed
                ):

                    self._deregister(
                        object_id
                    )

            return self.objects

        # ----------------------------------------------------
        # No existing objects
        # ----------------------------------------------------

        if len(self.objects) == 0:

            for detection in detections:

                self._register(
                    detection["class_id"],
                    detection["class_name"],
                    detection["centroid"],
                    detection["bbox"]
                )

            return self.objects

        # ----------------------------------------------------
        # Candidate matching
        # ----------------------------------------------------

        object_ids = list(
            self.objects.keys()
        )

        candidates = []

        for object_id in object_ids:

            obj = self.objects[
                object_id
            ]

            predicted_position = (
                obj.centroid
            )

            for detection_index, detection in enumerate(
                detections
            ):

                # Never match different classes.
                if (
                    obj.class_id
                    !=
                    detection["class_id"]
                ):

                    continue

                distance = self._distance(
                    predicted_position,
                    detection["centroid"]
                )

                if (
                    distance
                    <=
                    self.max_distance
                ):

                    candidates.append(
                        (
                            distance,
                            object_id,
                            detection_index
                        )
                    )

        # Closest match first.
        candidates.sort(
            key=lambda item: item[0]
        )

        matched_objects = set()
        matched_detections = set()

        for (
            distance,
            object_id,
            detection_index
        ) in candidates:

            if (
                object_id
                in
                matched_objects
            ):

                continue

            if (
                detection_index
                in
                matched_detections
            ):

                continue

            detection = detections[
                detection_index
            ]

            self.objects[
                object_id
            ].update_from_detection(
                detection["centroid"],
                detection["bbox"]
            )

            matched_objects.add(
                object_id
            )

            matched_detections.add(
                detection_index
            )

        # ----------------------------------------------------
        # Unmatched objects
        # ----------------------------------------------------

        for object_id in object_ids:

            if (
                object_id
                not in
                matched_objects
            ):

                if (
                    object_id
                    not in
                    self.objects
                ):

                    continue

                obj = self.objects[
                    object_id
                ]

                obj.missed_detections += 1

                if (
                    obj.missed_detections
                    >
                    self.max_missed
                ):

                    self._deregister(
                        object_id
                    )

        # ----------------------------------------------------
        # New objects
        # ----------------------------------------------------

        for detection_index, detection in enumerate(
            detections
        ):

            if (
                detection_index
                not in
                matched_detections
            ):

                self._register(
                    detection["class_id"],
                    detection["class_name"],
                    detection["centroid"],
                    detection["bbox"]
                )

        return self.objects

    def predict(self):

        """
        Predict positions between YOLO detections.
        """

        # Increase the missed-detection counter once per
        # prediction cycle only when appropriate.
        #
        # The actual YOLO miss count is handled in update().
        for object_id in list(
            self.objects.keys()
        ):

            if (
                object_id
                not in
                self.objects
            ):

                continue

            self.objects[
                object_id
            ].predict()

        return self.objects


# ============================================================
# YOLO HELPERS
# ============================================================

def get_class_name(
    names,
    class_id
):

    try:

        return names[
            int(class_id)
        ]

    except Exception:

        return str(class_id)


def extract_detections(
    result,
    names
):

    detections = []

    if result.boxes is None:

        return detections

    for box in result.boxes:

        confidence = float(
            box.conf[0]
        )

        if (
            confidence
            <
            CONFIDENCE_THRESHOLD
        ):

            continue

        class_id = int(
            box.cls[0]
        )

        x1, y1, x2, y2 = (
            box.xyxy[0].tolist()
        )

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        centroid = (
            int(
                (x1 + x2)
                /
                2
            ),
            int(
                (y1 + y2)
                /
                2
            )
        )

        detections.append(
            {
                "class_id": class_id,

                "class_name": get_class_name(
                    names,
                    class_id
                ),

                "confidence": confidence,

                "centroid": centroid,

                "bbox": (
                    x1,
                    y1,
                    x2,
                    y2
                )
            }
        )

    return detections


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)

    print(
        "Raspberry Pi Real-Time Object Detection & Tracking"
    )

    print(
        "Asynchronous Capture + YOLO + Stable Tracking"
    )

    print("=" * 60)

    print()

    print(
        f"Stream:             {STREAM_URL}"
    )

    print(
        f"YOLO model:         {YOLO_MODEL}"
    )

    print(
        f"YOLO input:         {YOLO_SIZE}x{YOLO_SIZE}"
    )

    print(
        f"Detection interval: every "
        f"{DETECTION_INTERVAL} frames"
    )

    print()

    # --------------------------------------------------------
    # Load YOLO
    # --------------------------------------------------------

    print(
        "Loading YOLO model..."
    )

    model = YOLO(
        YOLO_MODEL
    )

    print(
        "YOLO model loaded successfully."
    )

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    camera = LatestFrameCamera(
        STREAM_URL
    )

    camera.start()

    print(
        "Camera capture thread started."
    )

    # --------------------------------------------------------
    # Tracker
    # --------------------------------------------------------

    tracker = CentroidTracker(
        max_distance=MAX_MATCH_DISTANCE,
        max_missed=MAX_MISSED_DETECTIONS
    )

    target_id = None

    # --------------------------------------------------------
    # Performance
    # --------------------------------------------------------

    total_frames = 0

    detection_count = 0

    detection_times = deque(
        maxlen=20
    )

    start_time = time.perf_counter()

    last_status_time = start_time

    frames_since_detection = 0

    try:

        while True:

            frame = (
                camera.get_latest_frame()
            )

            if frame is None:

                time.sleep(
                    0.01
                )

                continue

            total_frames += 1

            frames_since_detection += 1

            # ------------------------------------------------
            # Run YOLO
            # ------------------------------------------------

            if (
                frames_since_detection
                >=
                DETECTION_INTERVAL
            ):

                frames_since_detection = 0

                detection_start = (
                    time.perf_counter()
                )

                results = model.predict(
                    source=frame,
                    imgsz=YOLO_SIZE,
                    conf=CONFIDENCE_THRESHOLD,
                    device="cpu",
                    verbose=False
                )

                detection_time = (
                    time.perf_counter()
                    -
                    detection_start
                )

                detection_times.append(
                    detection_time
                )

                detection_count += 1

                detections = (
                    extract_detections(
                        results[0],
                        model.names
                    )
                )

                tracked_objects = (
                    tracker.update(
                        detections
                    )
                )

                # ------------------------------------------------
                # Target handling
                # ------------------------------------------------

                if (
                    target_id is not None
                    and
                    target_id not in tracked_objects
                ):

                    print(
                        f"[TARGET] ID "
                        f"{target_id} lost"
                    )

                    target_id = None

                if (
                    target_id is None
                    and
                    len(tracked_objects) > 0
                ):

                    largest_object = max(
                        tracked_objects.values(),
                        key=lambda obj:
                            (
                                obj.bbox[2]
                                -
                                obj.bbox[0]
                            )
                            *
                            (
                                obj.bbox[3]
                                -
                                obj.bbox[1]
                            )
                    )

                    target_id = (
                        largest_object.id
                    )

                    print(
                        f"[TARGET] Selected ID "
                        f"{target_id} "
                        f"({largest_object.class_name})"
                    )

                # ------------------------------------------------
                # Detection output
                # ------------------------------------------------

                print(
                    f"[DETECTION "
                    f"{detection_count:04d}] "
                    f"{len(detections)} objects | "
                    f"inference "
                    f"{detection_time * 1000:.0f} ms"
                )

                for detection in detections:

                    print(
                        f"    "
                        f"{detection['class_name']:<15} "
                        f"conf="
                        f"{detection['confidence']:.2f} "
                        f"center="
                        f"{detection['centroid']}"
                    )

            # ------------------------------------------------
            # Prediction between YOLO runs
            # ------------------------------------------------

            else:

                tracked_objects = (
                    tracker.predict()
                )

                if (
                    target_id is not None
                    and
                    target_id not in tracked_objects
                ):

                    print(
                        f"[TARGET] ID "
                        f"{target_id} lost"
                    )

                    target_id = None

            # ------------------------------------------------
            # Status
            # ------------------------------------------------

            now = time.perf_counter()

            if (
                now - last_status_time
                >=
                STATUS_INTERVAL
            ):

                elapsed = (
                    now
                    -
                    start_time
                )

                camera_fps = (
                    camera.get_fps()
                )

                pipeline_fps = (
                    total_frames
                    /
                    elapsed
                    if elapsed > 0
                    else 0
                )

                detection_fps = (
                    detection_count
                    /
                    elapsed
                    if elapsed > 0
                    else 0
                )

                if len(
                    detection_times
                ) > 0:

                    avg_yolo = (
                        sum(
                            detection_times
                        )
                        /
                        len(
                            detection_times
                        )
                    )

                else:

                    avg_yolo = 0

                print()

                print(
                    "-" * 60
                )

                print(
                    "STATUS"
                )

                print(
                    f"Camera capture FPS: "
                    f"{camera_fps:.2f}"
                )

                print(
                    f"Pipeline loop FPS:  "
                    f"{pipeline_fps:.2f}"
                )

                print(
                    f"Detection FPS:      "
                    f"{detection_fps:.2f}"
                )

                print(
                    f"Avg YOLO time:      "
                    f"{avg_yolo * 1000:.0f} ms"
                )

                print(
                    f"Tracked objects:    "
                    f"{len(tracked_objects)}"
                )

                if (
                    target_id is not None
                ):

                    target = (
                        tracked_objects.get(
                            target_id
                        )
                    )

                    if target is not None:

                        print(
                            f"Target:             "
                            f"ID {target.id} "
                            f"({target.class_name}) "
                            f"center="
                            f"{target.centroid} "
                            f"missed="
                            f"{target.missed_detections}"
                        )

                    else:

                        print(
                            "Target:             LOST"
                        )

                else:

                    print(
                        "Target:             NONE"
                    )

                print(
                    "-" * 60
                )

                print()

                last_status_time = now

            time.sleep(
                0.001
            )

    except KeyboardInterrupt:

        print()

        print(
            "Stopping..."
        )

    finally:

        camera.stop()

        elapsed = (
            time.perf_counter()
            -
            start_time
        )

        camera_fps = (
            camera.get_fps()
        )

        pipeline_fps = (
            total_frames
            /
            elapsed
            if elapsed > 0
            else 0
        )

        detection_fps = (
            detection_count
            /
            elapsed
            if elapsed > 0
            else 0
        )

        if len(
            detection_times
        ) > 0:

            average_yolo = (
                sum(
                    detection_times
                )
                /
                len(
                    detection_times
                )
            )

        else:

            average_yolo = 0

        print()

        print(
            "=" * 60
        )

        print(
            "FINAL PERFORMANCE"
        )

        print(
            "=" * 60
        )

        print(
            f"Frames processed:       "
            f"{total_frames}"
        )

        print(
            f"YOLO detections:         "
            f"{detection_count}"
        )

        print(
            f"Camera capture FPS:      "
            f"{camera_fps:.2f}"
        )

        print(
            f"Pipeline loop FPS:       "
            f"{pipeline_fps:.2f}"
        )

        print(
            f"Detection FPS:           "
            f"{detection_fps:.2f}"
        )

        print(
            f"Average YOLO time:       "
            f"{average_yolo * 1000:.2f} ms"
        )

        if average_yolo > 0:

            print(
                f"YOLO-only FPS:           "
                f"{1 / average_yolo:.2f}"
            )

        print(
            "=" * 60
        )


if __name__ == "__main__":

    main()
