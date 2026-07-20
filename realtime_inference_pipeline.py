"""
Real-Time Parking Occupancy Inference Pipeline
================================================
Processes a live CCTV / RTSP / webcam feed frame-by-frame, runs a trained
object-detection model (YOLOv8, YOLOv11, or RF-DETR via TorchScript export),
and reports per-slot occupancy status with temporal hysteresis to suppress
transient false positives.

Architecture Overview
---------------------
    CCTV Stream ──► Frame Sampler ──► Object Detector ──► Slot Matcher ──► Overlay / API
                     (configurable      (pluggable         (point-in-       (visual
                      sample rate)       backend)           polygon)         feedback)

Usage Examples
--------------
    # Webcam with a YOLOv8 model (display overlay window):
    python realtime_inference_pipeline.py \\
        --source 0 \\
        --slots parking_slots.json \\
        --backend ultralytics \\
        --weights runs/detect/train_exp/weights/best.pt \\
        --display

    # RTSP IP-camera feed (headless, console output only):
    python realtime_inference_pipeline.py \\
        --source rtsp://admin:pass@192.168.1.64:554/stream1 \\
        --slots parking_slots.json \\
        --backend ultralytics \\
        --weights best.pt \\
        --conf 0.30

    # Save annotated output video for review:
    python realtime_inference_pipeline.py \\
        --source parking_footage.mp4 \\
        --slots parking_slots.json \\
        --backend ultralytics \\
        --weights best.pt \\
        --save output_annotated.mp4

Slot Configuration (parking_slots.json)
---------------------------------------
    {
      "slots": [
        {
          "id": "A1",
          "polygon": [[100, 200], [300, 200], [300, 400], [100, 400]]
        },
        {
          "id": "A2",
          "polygon": [[320, 200], [520, 200], [520, 400], [320, 400]]
        }
      ]
    }

Author : Dhaval Shah
License: All rights reserved – portfolio / learning reference only.
"""

import cv2
import time
import json
import argparse
import numpy as np
from typing import List, Tuple, Dict

# ─────────────────────────────────────────────
# Geometry: point-in-polygon (ray-casting)
# ─────────────────────────────────────────────

def point_in_polygon(
    x: float, y: float,
    polygon: List[Tuple[float, float]],
) -> bool:
    """Return True if (x, y) lies inside *polygon* using the ray-casting algorithm.

    This avoids an external dependency on Shapely / scikit-geometry for a single
    geometric predicate.
    """
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    x0, y0 = polygon[0]
    for i in range(1, n + 1):
        x1, y1 = polygon[i % n]
        if (y0 > y) != (y1 > y):
            x_intersect = (x1 - x0) * (y - y0) / (y1 - y0 + 1e-9) + x0
            if x < x_intersect:
                inside = not inside
        x0, y0 = x1, y1
    return inside


# ─────────────────────────────────────────────
# Slot state tracker with temporal hysteresis
# ─────────────────────────────────────────────

class SlotState:
    """Tracks a single parking slot with temporal hysteresis.

    Hysteresis prevents flickering: a slot only transitions from *empty → occupied*
    after ``occ_thresh`` consecutive occupied frames, and vice-versa for
    ``emp_thresh``.  This is critical for real-world feeds where partial
    occlusions, pedestrians, or lighting changes can cause momentary
    mis-detections.
    """

    def __init__(
        self,
        slot_id: str,
        polygon: List[Tuple[int, int]],
        occ_thresh: int = 3,
        emp_thresh: int = 3,
    ):
        self.slot_id = slot_id
        self.polygon = polygon
        self.state: str = "empty"  # 'empty' | 'occupied'
        self._occ_streak: int = 0
        self._emp_streak: int = 0
        self._occ_thresh = occ_thresh
        self._emp_thresh = emp_thresh

    def update(self, candidate: str) -> None:
        """Update internal counters and transition state when the streak threshold is met."""
        if candidate == "occupied":
            self._occ_streak += 1
            self._emp_streak = 0
            if self.state != "occupied" and self._occ_streak >= self._occ_thresh:
                self.state = "occupied"
        else:
            self._emp_streak += 1
            self._occ_streak = 0
            if self.state != "empty" and self._emp_streak >= self._emp_thresh:
                self.state = "empty"


# ─────────────────────────────────────────────
# Pluggable detector backend
# ─────────────────────────────────────────────

class Detector:
    """Thin wrapper around different detection backends.

    Supported backends
    ------------------
    * ``ultralytics`` – YOLOv8 / YOLOv11 via the Ultralytics Python API.
    * ``torch_script`` – any model exported to TorchScript (e.g. RF-DETR).
      Requires a custom decoding step (see ``_decode_torchscript``).

    The ``predict()`` method returns a list of detections as
    ``(x1, y1, x2, y2, confidence)`` tuples in *pixel coordinates* of the
    original frame.
    """

    _SUPPORTED_BACKENDS = {"ultralytics", "torch_script"}

    def __init__(
        self,
        backend: str = "ultralytics",
        weights: str | None = None,
        imgsz: int = 640,
        conf: float = 0.25,
    ):
        if backend not in self._SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose from: {self._SUPPORTED_BACKENDS}"
            )

        self.backend = backend
        self.imgsz = imgsz
        self.conf = conf
        self.model = None

        if backend == "ultralytics":
            try:
                from ultralytics import YOLO  # noqa: WPS433
            except ImportError as exc:
                raise RuntimeError(
                    "Ultralytics is required for this backend.\n"
                    "Install with:  pip install ultralytics"
                ) from exc
            self.model = YOLO(weights or "yolov8n.pt")

        elif backend == "torch_script":
            import torch  # noqa: WPS433
            if weights is None:
                raise ValueError("--weights is required for the torch_script backend.")
            self.model = torch.jit.load(weights)
            self.model.eval()

    # ── public API ──────────────────────────────

    def predict(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """Run inference on a single BGR frame and return bounding-box detections."""
        if self.backend == "ultralytics":
            return self._predict_ultralytics(frame_bgr)
        elif self.backend == "torch_script":
            return self._predict_torchscript(frame_bgr)
        return []

    # ── private helpers ─────────────────────────

    def _predict_ultralytics(
        self, frame_bgr: np.ndarray
    ) -> List[Tuple[int, int, int, int, float]]:
        results = self.model(frame_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)
        boxes_out: list = []
        for r in results:
            if r.boxes is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                boxes_out.append((int(x1), int(y1), int(x2), int(y2), float(c)))
        return boxes_out

    def _predict_torchscript(
        self, frame_bgr: np.ndarray
    ) -> List[Tuple[int, int, int, int, float]]:
        """Preprocess → infer → decode for a TorchScript-exported model.

        NOTE: The decoding step is model-specific.  Adjust the post-processing
        below to match the output tensor format of your exported RF-DETR or
        other transformer model.
        """
        import torch  # noqa: WPS433

        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = frame_bgr.shape[:2]
        img = cv2.resize(img, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        tensor = torch.from_numpy(img).unsqueeze(0)  # 1×3×H×W

        with torch.no_grad():
            output = self.model(tensor)

        # ── TODO: decode *output* into (x1, y1, x2, y2, conf) ──
        # The exact tensor layout depends on your export.  A typical RF-DETR
        # export produces a tensor of shape (1, num_queries, 6) where each row
        # is [x_center, y_center, width, height, class_id, score].
        # Scale coordinates back to (w_orig, h_orig) before returning.
        raise NotImplementedError(
            "Decode your RF-DETR/TorchScript outputs here.  "
            "See docstring for expected format."
        )


# ─────────────────────────────────────────────
# Occupancy evaluation
# ─────────────────────────────────────────────

def evaluate_occupancy(
    slots: Dict[str, SlotState],
    detections: List[Tuple[int, int, int, int, float]],
) -> None:
    """For every slot, check whether any detection centroid falls inside its polygon.

    A more sophisticated version could use IoU-based assignment or Hungarian
    matching, but centroid-in-polygon is robust for top-down camera angles where
    bounding boxes rarely span multiple slots.
    """
    for slot in slots.values():
        is_occupied = False
        for x1, y1, x2, y2, _conf in detections:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            if point_in_polygon(cx, cy, slot.polygon):
                is_occupied = True
                break
        slot.update("occupied" if is_occupied else "empty")


# ─────────────────────────────────────────────
# Visual overlay renderer
# ─────────────────────────────────────────────

def draw_overlay(
    frame: np.ndarray,
    slots: Dict[str, SlotState],
    thickness: int = 2,
) -> np.ndarray:
    """Draw colour-coded slot polygons on a copy of *frame*.

    Green = empty, Red = occupied.  Each slot is labelled with its ID.
    """
    canvas = frame.copy()
    for slot in slots.values():
        color = (0, 200, 0) if slot.state == "empty" else (0, 0, 200)
        pts = np.array(slot.polygon, dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=thickness)

        # Draw a semi-transparent fill for better visibility
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], color=color)
        cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)

        # Label with slot ID and status
        centroid = np.mean(pts, axis=0).astype(int)
        label = f"{slot.slot_id}: {'OCC' if slot.state == 'occupied' else 'FREE'}"
        cv2.putText(
            canvas, label,
            (centroid[0] - 25, centroid[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return canvas


# ─────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time parking occupancy detection from a CCTV / RTSP stream.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  # Local webcam
  python realtime_inference_pipeline.py --source 0 --slots slots.json --display

  # RTSP feed
  python realtime_inference_pipeline.py \\
      --source rtsp://admin:pass@192.168.1.64:554/stream1 \\
      --slots slots.json --weights best.pt

  # Pre-recorded video → annotated output
  python realtime_inference_pipeline.py \\
      --source parking_lot.mp4 --slots slots.json \\
      --weights best.pt --save annotated.mp4
""",
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video source: camera index (0/1/2), path to video file, or RTSP URL.",
    )
    parser.add_argument(
        "--slots", type=str, required=True,
        help="Path to the parking-slot configuration JSON (see README for schema).",
    )
    parser.add_argument(
        "--backend", type=str, default="ultralytics",
        choices=["ultralytics", "torch_script"],
        help="Detection backend (default: ultralytics).",
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to model weights (.pt for YOLO, .ts for TorchScript).",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Model input resolution (square, default: 640).",
    )
    parser.add_argument(
        "--conf", type=float, default=0.25,
        help="Confidence threshold for the detector (default: 0.25).",
    )
    parser.add_argument(
        "--sample-ms", type=int, default=500,
        help="Process one frame every N milliseconds (default: 500).",
    )
    parser.add_argument(
        "--occ-thresh", type=int, default=3,
        help="Consecutive frames needed to confirm OCCUPIED (default: 3).",
    )
    parser.add_argument(
        "--emp-thresh", type=int, default=3,
        help="Consecutive frames needed to confirm EMPTY (default: 3).",
    )
    parser.add_argument(
        "--display", action="store_true",
        help="Show a live overlay window (for local testing / demos).",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="Path to save an annotated output video (e.g. output.mp4).",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # ── Load slot definitions ──
    with open(args.slots, "r") as fh:
        slot_config = json.load(fh)

    slots: Dict[str, SlotState] = {}
    for entry in slot_config["slots"]:
        sid = entry["id"]
        poly = [(int(x), int(y)) for x, y in entry["polygon"]]
        slots[sid] = SlotState(sid, poly, args.occ_thresh, args.emp_thresh)

    print(f"[INFO] Loaded {len(slots)} parking slot(s) from '{args.slots}'")

    # ── Open video source ──
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {args.source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w_frame = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Source opened – {w_frame}×{h_frame} @ {fps_in:.1f} FPS")

    # ── Initialise detector ──
    detector = Detector(
        backend=args.backend,
        weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
    )
    print(f"[INFO] Detector ready – backend={args.backend}, imgsz={args.imgsz}, conf={args.conf}")

    # ── Optional: video writer ──
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, 20.0, (w_frame, h_frame))
        print(f"[INFO] Recording annotated video → {args.save}")

    # ── Main loop ──
    last_sample_ts = 0.0
    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # For live streams, retry; for files, we've reached the end.
                if isinstance(source, str) and not source.isdigit():
                    print("[WARN] Frame read failed – retrying …")
                    time.sleep(0.1)
                    continue
                else:
                    break

            now = time.time()
            if (now - last_sample_ts) * 1000.0 < args.sample_ms:
                continue  # skip until the next sample window
            last_sample_ts = now
            frame_count += 1

            # Run detection
            detections = detector.predict(frame)

            # Update slot occupancy with hysteresis
            evaluate_occupancy(slots, detections)

            # Compute summary stats
            occupied = sum(1 for s in slots.values() if s.state == "occupied")
            total = len(slots)
            available = total - occupied

            # Render overlay or print status
            if args.display or writer is not None:
                overlay = draw_overlay(frame, slots)
                status_text = f"Available: {available}/{total}  |  Occupied: {occupied}/{total}"
                cv2.putText(
                    overlay, status_text, (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA,
                )
                if args.display:
                    cv2.imshow("Parking Occupancy – Press Q to quit", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                if writer is not None:
                    writer.write(overlay)
            else:
                print(
                    f"[Frame {frame_count:>5}]  "
                    f"Available: {available}/{total}  |  "
                    f"Occupied: {occupied}/{total}  |  "
                    f"Detections: {len(detections)}"
                )

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print(f"[INFO] Processed {frame_count} sampled frames. Pipeline stopped.")


if __name__ == "__main__":
    main()
