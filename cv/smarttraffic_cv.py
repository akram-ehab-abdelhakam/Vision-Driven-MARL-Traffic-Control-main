"""
SmartTraffic-OM — Computer Vision Pipeline
===========================================
Reads a traffic video and produces:
  1. Per-vehicle detection + tracking  (YOLOv8 + BotSort)
  2. Vehicle type classification        (car / bus / truck / motorcycle / emergency)
  3. Per-interval type distribution
  4. Overall type distribution summary  (only output consumed downstream)

The type distribution represents the real Egyptian urban vehicle fleet
composition observed in the video. It is used exclusively by
route_postprocessor.py to assign SUMO vType attributes to generated vehicles.

Demand magnitude and entry-edge weights come from the field survey
(Om-Albadl, Assiut, March 2026) — NOT from this pipeline.

Usage
-----
  python smarttraffic_cv.py --video path/to/video.mp4
  python smarttraffic_cv.py --video path/to/video.mp4 --interval 60 --output stats.json --show

Requirements
------------
  pip install ultralytics opencv-python numpy
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

# ── Constants ──────────────────────────────────────────────────────────────────

YOLO_CLASS_MAP = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
DEFAULT_MODEL      = "yolov8n.pt"
DEFAULT_INTERVAL_S = 900       # 15-minute intervals
PROGRESS_EVERY     = 30        # print a progress line every N frames

# ── Emergency heuristic thresholds ────────────────────────────────────────────
# A vehicle is classified as emergency if it scores >= 2:
#   +1 if bbox area  > EMRG_AREA_THRESH  (large vehicle, e.g. fire truck)
#   +1 if speed      > EMRG_SPEED_THRESH pixels/frame
# A borderline score of 1 receives a low-probability stochastic promotion
# to model rare genuine emergency appearances without a fixed injection ratio.
EMRG_AREA_THRESH       = 0.08
EMRG_SPEED_THRESH      = 18.0
EMRG_BURST_PROBABILITY = 0.01

# ── Confidence weighting ──────────────────────────────────────────────────────
# High-confidence detections count as 1.0; low-confidence as 0.5.
# Avoids hard cutoffs while discounting uncertain detections.
CONF_HIGH_THRESH = 0.55
CONF_LOW_WEIGHT  = 0.5


# ── Core class ─────────────────────────────────────────────────────────────────

class SmartTrafficCV:

    def __init__(
        self,
        model_path: str  = DEFAULT_MODEL,
        interval_s: int  = DEFAULT_INTERVAL_S,
        show: bool       = False,
        conf: float      = 0.35,
    ):
        print(f"[CV] Loading model: {model_path}")
        self.model      = YOLO(model_path)
        self.interval_s = interval_s
        self.show       = show
        self.conf       = conf

        # Line-crossing state
        self._line_axis: str   = None
        self._line_pos:  float = 0.5

        # Global track memory — persists across interval flushes
        # {track_id: frame_idx_last_counted}
        self._track_memory: dict = {}
        self._RECOUNT_COOLDOWN   = 90

        # Previous centres for motion direction and crossing detection
        self._prev_centres: dict  = {}
        self._motion_samples: list = []

    # ── Line-crossing helpers ──────────────────────────────────────────────────

    def _infer_line_axis(self, frame_w: int, frame_h: int) -> None:
        if not self._motion_samples:
            self._line_axis = "v"
            print("\n[CV] Counting line fallback: vertical (no motion data)")
            return
        dx_mean = abs(np.mean([s[0] for s in self._motion_samples]))
        dy_mean = abs(np.mean([s[1] for s in self._motion_samples]))
        self._line_axis = "v" if dx_mean >= dy_mean else "h"
        self._line_pos  = 0.5
        label = "vertical (left↔right)" if self._line_axis == "v" \
                else "horizontal (up↔down)"
        print(f"\n[CV] Counting line auto-detected: {label}")

    def _crosses_line(
        self,
        prev_cx: float, prev_cy: float,
        curr_cx: float, curr_cy: float,
        frame_w: int,   frame_h: int,
    ) -> bool:
        if self._line_axis == "v":
            line = frame_w * self._line_pos
            return (prev_cx < line <= curr_cx) or (curr_cx < line <= prev_cx)
        else:
            line = frame_h * self._line_pos
            return (prev_cy < line <= curr_cy) or (curr_cy < line <= prev_cy)

    # ── Emergency detection ────────────────────────────────────────────────────

    @staticmethod
    def _is_emergency(bbox_xyxy, speed_px: float, frame_w: int, frame_h: int) -> bool:
        x1, y1, x2, y2 = bbox_xyxy
        area_frac = ((x2 - x1) * (y2 - y1)) / max(frame_w * frame_h, 1)
        score  = int(area_frac > EMRG_AREA_THRESH)
        score += int(speed_px  > EMRG_SPEED_THRESH)
        if score == 1 and random.random() < EMRG_BURST_PROBABILITY:
            score += 1
        return score >= 2

    # ── Main run loop ──────────────────────────────────────────────────────────

    def run(self, video_path: str, output_path: str = "cv_stats.json") -> dict:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_dur_s  = total_frames / fps if total_frames > 0 else 0

        effective_s = self.interval_s
        if video_dur_s > 0 and video_dur_s < self.interval_s:
            effective_s = max(1, int(video_dur_s))
            print(f"[CV] NOTE: Video duration ({video_dur_s:.0f}s) < interval "
                  f"({self.interval_s}s). Auto-adjusting interval to {effective_s}s.")

        interval_frames = max(1, int(fps * effective_s))

        print(f"[CV] Video    : {video_path}")
        print(f"[CV] Duration : {video_dur_s:.1f}s  |  FPS: {fps:.1f}"
              f"  |  Frames: {total_frames}")
        print(f"[CV] Interval : {effective_s}s  ({interval_frames} frames/interval)")
        print(f"[CV] Output   : {output_path}")
        print(f"[CV] Press Q to stop early if --show is active.\n")

        intervals   = []
        interval_id = 0
        frame_idx   = 0

        # Weighted crossing counts per type
        cross_counts = defaultdict(float)
        emrg_count   = 0.0

        MOTION_WARMUP = 300

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                classes=list(YOLO_CLASS_MAP.keys()),
                conf=self.conf,
                verbose=False,
            )

            frame_h, frame_w = frame.shape[:2]
            boxes = results[0].boxes

            if boxes.id is not None:
                ids     = boxes.id.int().tolist()
                cls_ids = boxes.cls.int().tolist()
                xyxys   = boxes.xyxy.tolist()
                confs   = boxes.conf.tolist()

                # Motion warmup — collect samples to infer line axis
                if frame_idx < MOTION_WARMUP:
                    for tid, xyxy in zip(ids, xyxys):
                        cx = (xyxy[0] + xyxy[2]) / 2
                        cy = (xyxy[1] + xyxy[3]) / 2
                        if tid in self._prev_centres:
                            px, py = self._prev_centres[tid]
                            self._motion_samples.append((cx - px, cy - py))
                        self._prev_centres[tid] = (cx, cy)
                    if frame_idx == MOTION_WARMUP - 1:
                        self._infer_line_axis(frame_w, frame_h)

                # Crossing detection — only after axis is known
                if self._line_axis is not None:
                    for tid, cls_id, xyxy, raw_conf in zip(ids, cls_ids, xyxys, confs):
                        cx = (xyxy[0] + xyxy[2]) / 2
                        cy = (xyxy[1] + xyxy[3]) / 2

                        if tid in self._prev_centres:
                            px, py = self._prev_centres[tid]
                            speed  = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5

                            crossed     = self._crosses_line(px, py, cx, cy,
                                                              frame_w, frame_h)
                            last_counted = self._track_memory.get(tid, -9999)
                            cooldown_ok  = (frame_idx - last_counted) > self._RECOUNT_COOLDOWN

                            if crossed and cooldown_ok:
                                self._track_memory[tid] = frame_idx

                                # Confidence-aware weight
                                weight = (1.0 if raw_conf >= CONF_HIGH_THRESH
                                          else CONF_LOW_WEIGHT)

                                if self._is_emergency(xyxy, speed, frame_w, frame_h):
                                    emrg_count += weight
                                else:
                                    vtype = YOLO_CLASS_MAP.get(cls_id, "car")
                                    cross_counts[vtype] += weight

                        self._prev_centres[tid] = (cx, cy)

            frame_idx += 1

            if frame_idx % PROGRESS_EVERY == 0:
                pct_done     = (frame_idx / max(total_frames, 1)) * 100
                total_so_far = sum(cross_counts.values()) + emrg_count
                print(
                    f"  Frame {frame_idx:>6}/{total_frames}"
                    f"  ({pct_done:5.1f}%)"
                    f"  |  Interval {interval_id}"
                    f"  |  Crossed this interval: {total_so_far:.1f}",
                    end="\r",
                )

            if self.show:
                annotated = results[0].plot(line_width=1, font_size=0.4,
                                             labels=True, conf=True)
                info = (f"Interval {interval_id} | "
                        f"{(frame_idx % interval_frames)/fps:.0f}s / {effective_s}s "
                        f"| Crossed: {sum(cross_counts.values()) + emrg_count:.1f}")
                cv2.putText(annotated, info, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 210, 255), 2)
                cv2.imshow("SmartTraffic-OM  |  CV Pipeline", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[CV] Stopped early by user.")
                    break

            # Flush completed interval
            if frame_idx % interval_frames == 0:
                print()
                ivl = self._build_interval(interval_id, cross_counts,
                                           emrg_count, effective_s)
                intervals.append(ivl)
                self._print_interval(ivl)
                interval_id  += 1
                cross_counts  = defaultdict(float)
                emrg_count    = 0.0

        cap.release()
        if self.show:
            cv2.destroyAllWindows()

        # Flush remaining frames as a final partial interval
        if sum(cross_counts.values()) > 0 or emrg_count > 0:
            print()
            remaining_frames = frame_idx % interval_frames or interval_frames
            partial_s = max(1.0, remaining_frames / fps)
            ivl = self._build_interval(interval_id, cross_counts,
                                       emrg_count, partial_s)
            intervals.append(ivl)
            self._print_interval(ivl)

        if not intervals:
            print("[CV] WARNING: No vehicles detected in any interval.")

        stats = {
            "intervals": intervals,
            "summary":   self._build_summary(intervals),
        }

        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\n[CV] Done — {len(intervals)} interval(s) → {output_path}")
        self._print_summary(stats["summary"])
        return stats

    # ── Interval builder ───────────────────────────────────────────────────────

    def _build_interval(
        self,
        interval_id:  int,
        cross_counts: dict,   # {vtype: weighted_crossing_count}
        emrg_count:   float,
        duration_s:   float,
    ) -> dict:

        grand_total = sum(cross_counts.values()) + emrg_count
        if grand_total <= 0:
            grand_total = 1.0

        # Normalised type distribution
        raw = {k: cross_counts.get(k, 0.0) / grand_total
               for k in ["car", "bus", "truck", "motorcycle"]}
        raw["emergency"] = emrg_count / grand_total
        s    = sum(raw.values()) or 1.0
        dist = {k: round(v / s, 4) for k, v in raw.items()}

        return {
            "interval_id":               interval_id,
            "duration_seconds":          round(duration_s, 1),
            "vehicle_type_distribution": dist,
        }

    # ── Summary builder ────────────────────────────────────────────────────────

    def _build_summary(self, intervals: list) -> dict:
        if not intervals:
            return {}

        # Flow-weighted average type distribution across all intervals
        # Weight each interval equally since we no longer track flow.
        n = len(intervals)
        overall = {k: 0.0 for k in ["car", "bus", "truck", "motorcycle", "emergency"]}
        for iv in intervals:
            for k in overall:
                overall[k] += iv["vehicle_type_distribution"].get(k, 0.0) / n
        overall = {k: round(v, 4) for k, v in overall.items()}

        return {
            "overall_type_distribution": overall,
        }

    # ── Interval printer ───────────────────────────────────────────────────────

    @staticmethod
    def _print_interval(iv: dict) -> None:
        d = iv["vehicle_type_distribution"]
        print(
            f"  [Interval {iv['interval_id']:>2}]"
            f"  {iv['duration_seconds']:>6.1f}s"
            f"  |  car {d['car']*100:4.1f}%"
            f"  bus {d['bus']*100:4.1f}%"
            f"  truck {d['truck']*100:4.1f}%"
            f"  moto {d['motorcycle']*100:4.1f}%"
            f"  emg {d['emergency']*100:4.1f}%"
        )

    # ── Summary printer ────────────────────────────────────────────────────────

    @staticmethod
    def _print_summary(s: dict) -> None:
        if not s:
            return
        sep = "─" * 60
        print(f"\n{sep}")
        print(f"  VEHICLE TYPE DISTRIBUTION SUMMARY")
        print(sep)
        d = s["overall_type_distribution"]
        print(f"  car {d['car']*100:.1f}%  "
              f"bus {d['bus']*100:.1f}%  "
              f"truck {d['truck']*100:.1f}%  "
              f"motorcycle {d['motorcycle']*100:.1f}%  "
              f"emergency {d['emergency']*100:.1f}%")
        print(f"{sep}")
        print(f"  JSON saved. overall_type_distribution is ready for")
        print(f"  route_postprocessor.py to assign SUMO vTypes.\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SmartTraffic-OM — CV Pipeline  |  Video → cv_stats.json",
    )
    parser.add_argument("--video",    required=True,
                        help="Path to input traffic video")
    parser.add_argument("--output",   default="cv_stats.json",
                        help="Output JSON path (default: cv_stats.json)")
    parser.add_argument("--model",    default=DEFAULT_MODEL,
                        help=f"YOLOv8 weights (default: {DEFAULT_MODEL})")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_S,
                        help="Aggregation window in seconds (default: 900)")
    parser.add_argument("--conf",     type=float, default=0.35,
                        help="Detection confidence threshold 0–1 (default: 0.35)")
    parser.add_argument("--show",     action="store_true",
                        help="Display annotated video while processing")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    pipeline = SmartTrafficCV(
        model_path=args.model,
        interval_s=args.interval,
        show=args.show,
        conf=args.conf,
    )
    pipeline.run(video_path=args.video, output_path=args.output)


if __name__ == "__main__":
    main()