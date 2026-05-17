from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2


@dataclass
class TrackState:
    trail: deque[tuple[int, int]] = field(default_factory=lambda: deque(maxlen=30))
    previous_distance: float | None = None
    previous_stable_side: int | None = None
    counted: bool = False
    last_frame_seen: int = -1


def load_line_definition(
    line_config: Path | None,
    line_points: list[int] | None,
    default_dead_zone: float,
) -> tuple[tuple[int, int], tuple[int, int], float]:
    if line_config is not None:
        payload = json.loads(line_config.read_text(encoding="utf-8"))
        line = payload.get("line")
        if not isinstance(line, list) or len(line) != 2:
            raise ValueError("line_config JSON must contain 'line': [[x1, y1], [x2, y2]]")
        dead_zone = float(payload.get("dead_zone", default_dead_zone))
        return (tuple(line[0]), tuple(line[1]), dead_zone)

    if line_points is None or len(line_points) != 4:
        raise ValueError("Please provide either --line-config or four integers via --line")
    return ((line_points[0], line_points[1]), (line_points[2], line_points[3]), default_dead_zone)


def bbox_center(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int]:
    return (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0)))


def bbox_count_point(x1: float, y1: float, x2: float, y2: float, mode: str) -> tuple[int, int]:
    if mode == "center":
        return bbox_center(x1, y1, x2, y2)
    if mode == "bottom_center":
        return (int(round((x1 + x2) / 2.0)), int(round(y2)))
    if mode == "bottom_mid_80":
        return (int(round((x1 + x2) / 2.0)), int(round(y1 * 0.2 + y2 * 0.8)))
    raise ValueError(f"Unsupported count point mode: {mode}")


def signed_distance_to_line(
    point: tuple[int, int],
    line_start: tuple[int, int],
    line_end: tuple[int, int],
) -> float:
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    numerator = (x2 - x1) * (y0 - y1) - (y2 - y1) * (x0 - x1)
    denominator = math.hypot(x2 - x1, y2 - y1)
    if denominator == 0:
        raise ValueError("Virtual line endpoints must be different")
    return numerator / denominator


def check_line_crossing(previous_distance: float | None, current_distance: float, dead_zone: float) -> bool:
    if previous_distance is None:
        return False
    if abs(previous_distance) <= dead_zone or abs(current_distance) <= dead_zone:
        return False
    return previous_distance * current_distance < 0


def stable_side(distance: float, dead_zone: float) -> int:
    if abs(distance) <= dead_zone:
        return 0
    return 1 if distance > 0 else -1


def color_for_track(track_id: int) -> tuple[int, int, int]:
    palette = (
        (56, 56, 255),
        (151, 157, 255),
        (31, 112, 255),
        (29, 178, 255),
        (49, 210, 207),
        (10, 249, 72),
        (23, 204, 146),
        (134, 219, 61),
        (52, 147, 26),
        (187, 212, 0),
        (168, 153, 44),
        (255, 194, 0),
        (147, 69, 52),
        (255, 115, 100),
        (236, 24, 0),
        (255, 56, 132),
        (133, 0, 82),
        (255, 56, 203),
        (200, 149, 255),
        (199, 55, 255),
    )
    return palette[track_id % len(palette)]


def draw_virtual_line(
    frame,
    line_start: tuple[int, int],
    line_end: tuple[int, int],
    total_count: int,
    forward_count: int,
    backward_count: int,
) -> None:
    cv2.line(frame, line_start, line_end, (0, 255, 255), thickness=3)
    cv2.putText(
        frame,
        f"Total crossings: {total_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"A->B: {forward_count}  B->A: {backward_count}",
        (20, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def iou_xyxy(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union
