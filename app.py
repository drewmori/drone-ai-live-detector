from __future__ import annotations

import json
import shutil
import sys
import time
from ctypes import windll
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import mss
import numpy as np
import pygetwindow as gw
from ultralytics import YOLO

try:
    import win32con
    import win32gui
    import win32ui
except ImportError:
    win32con = None
    win32gui = None
    win32ui = None


PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_DIR / "config.json"
RUNS_DIR = PROJECT_DIR / "runs"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR = RUNS_DIR / RUN_ID
SNAPSHOT_DIR = RUN_DIR / "snapshots"
LABEL_CANDIDATE_DIR = RUN_DIR / "label_candidates"
DETECTION_LOG_PATH = RUN_DIR / "detections.jsonl"
PREVIEW_WINDOW = "Drone AI Live Detector"
CLASS_MENU_WINDOW = "Thermal Highlight Classes"
CONTROL_BAR_HEIGHT = 72
SHOW_THERMAL_HUMANS = False
SMALL_OBJECT_MODE = False
THERMAL_CLASS_NAMES: list[str] = []
SELECTED_THERMAL_CLASSES: set[str] = {"person"}
SELECTED_NORMAL_CLASSES: set[str] = set()
OCCLUSION_PREDICTION_ENABLED = True
OCCLUSION_PREDICTION_FRAMES = 10
OCCLUSION_MIN_MISSING_FRAMES = 2
OCCLUSION_MIN_TRACKED_FRAMES = 5
OCCLUSION_MIN_VELOCITY = 1.5
OCCLUSION_MAX_VELOCITY = 35.0
OCCLUSION_MIN_AREA = 900
OCCLUSION_TRACKS: dict[int, "ThermalTrack"] = {}
NEXT_OCCLUSION_TRACK_ID = 1
CONTROL_BUTTONS = {
    "thermal": (10, 12, 160, 58),
    "classes": (174, 12, 324, 58),
    "small": (338, 12, 500, 58),
}

cv2.setUseOptimized(True)


def prune_old_runs(keep: int = 3) -> None:
    RUNS_DIR.mkdir(exist_ok=True)
    run_dirs = sorted(
        [path for path in RUNS_DIR.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    for old_run in run_dirs[keep:]:
        shutil.rmtree(old_run, ignore_errors=True)


@dataclass
class CaptureRegion:
    left: int
    top: int
    width: int
    height: int

    def as_mss(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class Detection:
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]
    confirmed_frames: int = 1


@dataclass
class ThermalInstance:
    class_name: str
    box: tuple[int, int, int, int]
    mask: np.ndarray


@dataclass
class ThermalTrack:
    class_name: str
    box: tuple[int, int, int, int]
    mask: np.ndarray
    velocity: tuple[float, float] = (0.0, 0.0)
    missed_frames: int = 0
    tracked_frames: int = 1


class WindowCapture:
    def __init__(self, window, crop: dict[str, Any]):
        self.window = window
        self.hwnd = int(getattr(window, "_hWnd", 0) or 0)
        self.crop = crop
        self.direct_available = all((self.hwnd, win32con, win32gui, win32ui))

    def current_region(self) -> CaptureRegion:
        return region_from_window(self.window, self.crop)

    def grab(self, sct: mss.mss) -> np.ndarray:
        if self.direct_available:
            frame = self._grab_direct()
            if frame is not None:
                return frame

        raw = np.array(sct.grab(self.current_region().as_mss()))
        return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

    def _grab_direct(self) -> np.ndarray | None:
        if not win32gui.IsWindow(self.hwnd) or win32gui.IsIconic(self.hwnd):
            return None

        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        width = max(1, right - left)
        height = max(1, bottom - top)

        hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
        save_dc.SelectObject(bitmap)

        try:
            result = windll.user32.PrintWindow(self.hwnd, save_dc.GetSafeHdc(), 2)
            if result != 1:
                return None
            bmpinfo = bitmap.GetInfo()
            bmpstr = bitmap.GetBitmapBits(True)
            frame = np.frombuffer(bmpstr, dtype=np.uint8)
            frame.shape = (bmpinfo["bmHeight"], bmpinfo["bmWidth"], 4)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return apply_crop(frame, self.crop)
        finally:
            win32gui.DeleteObject(bitmap.GetHandle())
            save_dc.DeleteDC()
            mfc_dc.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hwnd_dc)


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing config file: {CONFIG_PATH}")
    with CONFIG_PATH.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def list_candidate_windows(title_hint: str = ""):
    windows = []
    hint = title_hint.casefold().strip()
    for window in gw.getAllWindows():
        title = (window.title or "").strip()
        if not title or window.width <= 80 or window.height <= 80:
            continue
        if hint and hint not in title.casefold():
            continue
        windows.append(window)
    return windows


def choose_window(title_hint: str = ""):
    windows = list_candidate_windows(title_hint)
    if not windows and title_hint:
        print(f"No windows matched hint {title_hint!r}. Showing all visible windows instead.\n")
        windows = list_candidate_windows("")

    if not windows:
        raise RuntimeError("No usable windows found. Open your AirPlay/mirror window and try again.")

    print("Pick the mirrored iPhone/AirPlay window:\n")
    for i, window in enumerate(windows, start=1):
        print(f"{i:2}. {window.title}  ({window.width}x{window.height} at {window.left},{window.top})")

    while True:
        choice = input("\nWindow number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(windows):
            return windows[int(choice) - 1]
        print("Enter one of the listed numbers.")


def region_from_window(window, crop: dict[str, Any]) -> CaptureRegion:
    left = max(0, int(window.left))
    top = max(0, int(window.top))
    width = max(1, int(window.width))
    height = max(1, int(window.height))

    if crop.get("enabled"):
        crop_left = max(0, int(crop.get("left", 0)))
        crop_top = max(0, int(crop.get("top", 0)))
        crop_right = max(0, int(crop.get("right", 0)))
        crop_bottom = max(0, int(crop.get("bottom", 0)))
        left += crop_left
        top += crop_top
        width = max(1, width - crop_left - crop_right)
        height = max(1, height - crop_top - crop_bottom)

    return CaptureRegion(left=left, top=top, width=width, height=height)


def apply_crop(frame: np.ndarray, crop: dict[str, Any]) -> np.ndarray:
    if not crop.get("enabled"):
        return frame
    height, width = frame.shape[:2]
    left = max(0, int(crop.get("left", 0)))
    top = max(0, int(crop.get("top", 0)))
    right = max(0, int(crop.get("right", 0)))
    bottom = max(0, int(crop.get("bottom", 0)))
    x2 = max(left + 1, width - right)
    y2 = max(top + 1, height - bottom)
    return frame[top:y2, left:x2]


def resolve_model(model_name: str) -> str:
    project_model = PROJECT_DIR / model_name
    if project_model.exists():
        return str(project_model)
    local_model = PROJECT_DIR / "models" / model_name
    if local_model.exists():
        return str(local_model)
    return model_name


def fit_preview(frame: np.ndarray, max_width: int = 960, max_height: int = 720) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def letterbox_preview(frame: np.ndarray, canvas_width: int = 1280, canvas_height: int = 720) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return frame

    scale = min(canvas_width / width, canvas_height / height)
    scaled_width = max(1, int(width * scale))
    scaled_height = max(1, int(height * scale))
    resized = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    x = (canvas_width - scaled_width) // 2
    y = (canvas_height - scaled_height) // 2
    canvas[y:y + scaled_height, x:x + scaled_width] = resized
    return canvas


def draw_button(canvas: np.ndarray, name: str, label: str, active: bool = False) -> None:
    x1, y1, x2, y2 = CONTROL_BUTTONS[name]
    bg = (82, 82, 82) if active else (48, 48, 48)
    border = (235, 235, 235) if active else (130, 130, 130)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 1)
    cv2.putText(canvas, label, (x1 + 12, y1 + 29), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (245, 245, 245), 1, cv2.LINE_AA)


def compose_preview(
    frame: np.ndarray,
    canvas_width: int,
    canvas_height: int,
    fps: float,
    confidence: float,
    paused: bool,
    counts: dict[str, int],
) -> np.ndarray:
    video = letterbox_preview(frame, canvas_width, canvas_height)
    canvas = np.zeros((canvas_height + CONTROL_BAR_HEIGHT, canvas_width, 3), dtype=np.uint8)
    canvas[:CONTROL_BAR_HEIGHT, :] = (16, 16, 16)
    canvas[CONTROL_BAR_HEIGHT:, :] = video

    draw_button(canvas, "thermal", "Thermal", SHOW_THERMAL_HUMANS)
    draw_button(canvas, "classes", "Classes", False)
    draw_button(canvas, "small", "Small Obj", SMALL_OBJECT_MODE)

    status = "PAUSED" if paused else "LIVE"
    view = "THERMAL" if SHOW_THERMAL_HUMANS else "NORMAL"
    object_mode = "SMALL" if SMALL_OBJECT_MODE else "STD"
    prediction = "PRED" if OCCLUSION_PREDICTION_ENABLED else "NO-PRED"
    summary = f"{status} | {view} | {object_mode} | {prediction} | FPS {fps:.1f} | conf {confidence:.2f}"
    if counts:
        summary += " | " + "  ".join(f"{name}:{count}" for name, count in sorted(counts.items()))
    cv2.putText(canvas, summary, (520, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas


def place_preview_window(region: CaptureRegion) -> None:
    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PREVIEW_WINDOW, 1280, 720 + CONTROL_BAR_HEIGHT)
    x = region.left + region.width + 24
    y = max(0, region.top)
    if x > 1200:
        x = 24
        y = 24
    cv2.moveWindow(PREVIEW_WINDOW, x, y)


def draw_hud(frame: np.ndarray, fps: float, confidence: float, paused: bool, counts: dict[str, int]) -> None:
    status = "PAUSED" if paused else "LIVE"
    view = "THERMAL HUMANS" if SHOW_THERMAL_HUMANS else "NORMAL"
    object_mode = "SMALL" if SMALL_OBJECT_MODE else "STD"
    parts = [f"{status}", view, object_mode, f"FPS {fps:.1f}", f"conf {confidence:.2f}"]
    if counts:
        parts.append("  ".join(f"{name}:{count}" for name, count in sorted(counts.items())))
    text = " | ".join(parts)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (10, 10, 10), -1)
    cv2.putText(frame, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


def save_snapshot(frame: np.ndarray, reason: str, metadata: dict[str, Any] | None = None) -> Path:
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe_reason = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in reason)[:50]
    path = SNAPSHOT_DIR / f"{stamp}_{safe_reason}.jpg"
    cv2.imwrite(str(path), frame)
    if metadata:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "run_id": RUN_ID,
            "snapshot": str(path),
            "reason": reason,
            **metadata,
        }
        with DETECTION_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    return path


def save_label_candidate(frame: np.ndarray, reason: str, detections: list[Detection]) -> Path:
    LABEL_CANDIDATE_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    safe_reason = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in reason)[:50]
    image_path = LABEL_CANDIDATE_DIR / f"{stamp}_{safe_reason}.jpg"
    json_path = LABEL_CANDIDATE_DIR / f"{stamp}_{safe_reason}.json"
    cv2.imwrite(str(image_path), frame)
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": RUN_ID,
        "image": str(image_path),
        "reason": reason,
        "detections": [
            {
                "class": detection.class_name,
                "confidence": round(detection.confidence, 3),
                "box": list(detection.box),
                "confirmed_frames": detection.confirmed_frames,
            }
            for detection in detections
        ],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return image_path


def extract_detections(results: Any, model: YOLO) -> list[Detection]:
    detections: list[Detection] = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = str(model.names.get(cls_id, cls_id))
        confidence = float(box.conf[0]) if box.conf is not None else 0.0
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        detections.append(Detection(class_name=class_name, confidence=confidence, box=(x1, y1, x2, y2)))
    return detections


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union else 0.0


def confirm_detections(
    detections: list[Detection],
    previous: list[Detection],
    required_frames: int,
    iou_threshold: float = 0.25,
) -> tuple[list[Detection], list[Detection]]:
    updated: list[Detection] = []
    confirmed: list[Detection] = []
    used_previous: set[int] = set()

    for detection in detections:
        best_index = -1
        best_score = 0.0
        for i, old in enumerate(previous):
            if i in used_previous or old.class_name != detection.class_name:
                continue
            overlap = box_iou(detection.box, old.box)
            distance = center_distance(detection.box, old.box)
            old_w, old_h = box_size(old.box)
            distance_score = max(0.0, 1.0 - distance / max(32.0, max(old_w, old_h) * 1.2))
            score = max(overlap, distance_score)
            if score > best_score:
                best_score = score
                best_index = i

        if best_index >= 0 and likely_same_object(detection.box, previous[best_index].box, iou_threshold):
            detection.confirmed_frames = previous[best_index].confirmed_frames + 1
            used_previous.add(best_index)

        updated.append(detection)
        if detection.confirmed_frames >= required_frames:
            confirmed.append(detection)

    return updated, confirmed


def box_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def box_size(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return max(1, x2 - x1), max(1, y2 - y1)


def center_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay = box_center(a)
    bx, by = box_center(b)
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def likely_same_object(a: tuple[int, int, int, int], b: tuple[int, int, int, int], iou_threshold: float = 0.08) -> bool:
    if box_iou(a, b) >= iou_threshold:
        return True

    aw, ah = box_size(a)
    bw, bh = box_size(b)
    max_reasonable_jump = max(32.0, max(aw, ah, bw, bh) * 0.85)
    return center_distance(a, b) <= max_reasonable_jump


def shift_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        mask,
        matrix,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def shifted_box(box: tuple[int, int, int, int], dx: float, dy: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width, int(x1 + dx))),
        max(0, min(height, int(y1 + dy))),
        max(0, min(width, int(x2 + dx))),
        max(0, min(height, int(y2 + dy))),
    )


def mask_area(mask: np.ndarray) -> int:
    return int(cv2.countNonZero(mask))


def velocity_magnitude(velocity: tuple[float, float]) -> float:
    return float((velocity[0] ** 2 + velocity[1] ** 2) ** 0.5)


def update_occlusion_tracks(instances: list[ThermalInstance], frame_shape: tuple[int, int, int]) -> np.ndarray:
    global NEXT_OCCLUSION_TRACK_ID
    predicted_mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if not OCCLUSION_PREDICTION_ENABLED:
        OCCLUSION_TRACKS.clear()
        return predicted_mask

    frame_height, frame_width = frame_shape[:2]
    matched_tracks: set[int] = set()

    for instance in instances:
        best_track_id = None
        best_score = 0.0
        for track_id, track in OCCLUSION_TRACKS.items():
            if track_id in matched_tracks or track.class_name != instance.class_name:
                continue
            overlap = box_iou(instance.box, track.box)
            distance = center_distance(instance.box, track.box)
            track_w, track_h = box_size(track.box)
            distance_score = max(0.0, 1.0 - distance / max(32.0, max(track_w, track_h) * 1.2))
            score = max(overlap, distance_score)
            if score > best_score:
                best_score = score
                best_track_id = track_id

        if best_track_id is None or not likely_same_object(instance.box, OCCLUSION_TRACKS[best_track_id].box):
            OCCLUSION_TRACKS[NEXT_OCCLUSION_TRACK_ID] = ThermalTrack(
                class_name=instance.class_name,
                box=instance.box,
                mask=instance.mask.copy(),
            )
            matched_tracks.add(NEXT_OCCLUSION_TRACK_ID)
            NEXT_OCCLUSION_TRACK_ID += 1
            continue

        track = OCCLUSION_TRACKS[best_track_id]
        old_cx, old_cy = box_center(track.box)
        new_cx, new_cy = box_center(instance.box)
        measured_velocity = (new_cx - old_cx, new_cy - old_cy)
        track.velocity = (
            track.velocity[0] * 0.55 + measured_velocity[0] * 0.45,
            track.velocity[1] * 0.55 + measured_velocity[1] * 0.45,
        )
        track.box = instance.box
        track.mask = instance.mask.copy()
        track.missed_frames = 0
        track.tracked_frames += 1
        matched_tracks.add(best_track_id)

    for track_id in list(OCCLUSION_TRACKS):
        if track_id in matched_tracks:
            continue

        track = OCCLUSION_TRACKS[track_id]
        if any(
            instance.class_name == track.class_name and center_distance(instance.box, track.box) <= max(96.0, max(box_size(track.box)) * 1.5)
            for instance in instances
        ):
            track.missed_frames = 0
            continue

        track.missed_frames += 1
        if track.missed_frames > OCCLUSION_PREDICTION_FRAMES:
            del OCCLUSION_TRACKS[track_id]
            continue
        if track.missed_frames < OCCLUSION_MIN_MISSING_FRAMES:
            continue

        speed = velocity_magnitude(track.velocity)
        if track.tracked_frames < OCCLUSION_MIN_TRACKED_FRAMES:
            continue
        if mask_area(track.mask) < OCCLUSION_MIN_AREA:
            continue
        if speed > OCCLUSION_MAX_VELOCITY:
            continue

        if speed < OCCLUSION_MIN_VELOCITY:
            dx = 0.0
            dy = 0.0
        else:
            dx = track.velocity[0] * track.missed_frames
            dy = track.velocity[1] * track.missed_frames
        ghost = shift_mask(track.mask, dx, dy)
        predicted_mask = cv2.max(predicted_mask, ghost)
        if speed >= OCCLUSION_MIN_VELOCITY:
            track.box = shifted_box(track.box, track.velocity[0], track.velocity[1], frame_width, frame_height)

    return predicted_mask


def handle_mouse(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
    global SHOW_THERMAL_HUMANS, SMALL_OBJECT_MODE
    if event == cv2.EVENT_LBUTTONDOWN:
        if y > CONTROL_BAR_HEIGHT:
            return
        if 165 <= x <= 335:
            show_class_menu()
            print("Thermal class menu opened.")
            return
        for name, (x1, y1, x2, y2) in CONTROL_BUTTONS.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                if name == "thermal":
                    SHOW_THERMAL_HUMANS = not SHOW_THERMAL_HUMANS
                    mode = "thermal" if SHOW_THERMAL_HUMANS else "normal"
                    print(f"View mode: {mode}")
                elif name == "classes":
                    show_class_menu()
                    print("Thermal class menu opened.")
                elif name == "small":
                    SMALL_OBJECT_MODE = not SMALL_OBJECT_MODE
                    mode = "small-object" if SMALL_OBJECT_MODE else "standard"
                    print(f"Object mode: {mode}")
                return
    if event == cv2.EVENT_RBUTTONDOWN:
        show_class_menu()
        print("Thermal class menu opened.")


def handle_class_menu_mouse(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
    global SHOW_THERMAL_HUMANS
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    selected_classes = active_selected_classes()
    row_height = 28
    header_height = 72
    col_width = 190

    if y < 38:
        if 8 <= x <= 130:
            SHOW_THERMAL_HUMANS = not SHOW_THERMAL_HUMANS
            mode = "thermal" if SHOW_THERMAL_HUMANS else "normal"
            print(f"View mode: {mode}")
            show_class_menu()
        elif 140 <= x <= 238:
            cv2.destroyWindow(CLASS_MENU_WINDOW)
        elif 248 <= x <= 346:
            selected_classes.clear()
            print(f"{active_view_name()} classes cleared.")
            show_class_menu()
        elif 356 <= x <= 454:
            if len(selected_classes) == len(THERMAL_CLASS_NAMES):
                selected_classes.clear()
                print(f"{active_view_name()} classes cleared.")
            else:
                selected_classes.clear()
                selected_classes.update(THERMAL_CLASS_NAMES)
                print(f"{active_view_name()} classes set to all.")
            show_class_menu()
        return

    if y < header_height:
        return

    col = x // col_width
    row = (y - header_height) // row_height
    rows_per_col = 20
    index = int(col * rows_per_col + row)
    if 0 <= index < len(THERMAL_CLASS_NAMES):
        name = THERMAL_CLASS_NAMES[index]
        if name in selected_classes:
            selected_classes.remove(name)
        else:
            selected_classes.add(name)
        print(f"{active_view_name()} classes: " + ", ".join(sorted(selected_classes)))
        show_class_menu()


def active_selected_classes() -> set[str]:
    return SELECTED_THERMAL_CLASSES if SHOW_THERMAL_HUMANS else SELECTED_NORMAL_CLASSES


def active_view_name() -> str:
    return "Thermal" if SHOW_THERMAL_HUMANS else "Normal"


def show_class_menu() -> None:
    if not THERMAL_CLASS_NAMES:
        return

    selected_classes = active_selected_classes()
    row_height = 28
    header_height = 72
    rows_per_col = 20
    col_width = 190
    cols = int(np.ceil(len(THERMAL_CLASS_NAMES) / rows_per_col))
    width = max(420, cols * col_width)
    height = header_height + rows_per_col * row_height
    menu = np.full((height, width, 3), 28, dtype=np.uint8)

    thermal_label = "Thermal: ON" if SHOW_THERMAL_HUMANS else "Thermal: OFF"
    all_label = "All: OFF" if len(selected_classes) == len(THERMAL_CLASS_NAMES) else "All: ON"
    buttons = [
        ((8, 8), (130, 34), thermal_label),
        ((140, 8), (238, 34), "Close"),
        ((248, 8), (346, 34), "Clear"),
        ((356, 8), (454, 34), all_label),
    ]
    for (x1, y1), (x2, y2), label in buttons:
        cv2.rectangle(menu, (x1, y1), (x2, y2), (64, 64, 64), -1)
        cv2.rectangle(menu, (x1, y1), (x2, y2), (120, 120, 120), 1)
        cv2.putText(menu, label, (x1 + 8, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (245, 245, 245), 1, cv2.LINE_AA)

    help_text = f"{active_view_name()} class picker. Click classes to show in this view."
    cv2.putText(menu, help_text, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (210, 210, 210), 1, cv2.LINE_AA)

    for i, name in enumerate(THERMAL_CLASS_NAMES):
        col = i // rows_per_col
        row = i % rows_per_col
        x = col * col_width
        y = header_height + row * row_height
        selected = name in selected_classes
        bg = (70, 70, 70) if selected else (38, 38, 38)
        fg = (255, 255, 255) if selected else (170, 170, 170)
        cv2.rectangle(menu, (x + 4, y + 2), (x + col_width - 6, y + row_height - 3), bg, -1)
        marker = "[x]" if selected else "[ ]"
        cv2.putText(menu, f"{marker} {name}", (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, fg, 1, cv2.LINE_AA)

    cv2.namedWindow(CLASS_MENU_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CLASS_MENU_WINDOW, width, height)
    cv2.setMouseCallback(CLASS_MENU_WINDOW, handle_class_menu_mouse)
    cv2.imshow(CLASS_MENU_WINDOW, menu)


def thermal_highlight_view(frame: np.ndarray, results: Any, model: YOLO) -> np.ndarray:
    highlight_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    current_instances: list[ThermalInstance] = []
    used_segmentation_mask = False

    if results.masks is not None and results.masks.data is not None:
        masks = results.masks.data.cpu().numpy()
        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0])
            class_name = str(model.names.get(cls_id, cls_id))
            if class_name not in SELECTED_THERMAL_CLASSES or i >= len(masks):
                continue

            mask = masks[i]
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            object_mask = (mask > 0.5).astype(np.uint8) * 255
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            highlight_mask = cv2.max(highlight_mask, object_mask)
            current_instances.append(ThermalInstance(class_name=class_name, box=(x1, y1, x2, y2), mask=object_mask))
            used_segmentation_mask = True

    for box in results.boxes:
        if used_segmentation_mask:
            break
        cls_id = int(box.cls[0])
        class_name = str(model.names.get(cls_id, cls_id))
        if class_name not in SELECTED_THERMAL_CLASSES:
            continue

        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        x1 = max(0, min(frame.shape[1] - 1, x1))
        x2 = max(0, min(frame.shape[1], x2))
        y1 = max(0, min(frame.shape[0] - 1, y1))
        y2 = max(0, min(frame.shape[0], y2))
        if x2 > x1 and y2 > y1:
            box_width = x2 - x1
            box_height = y2 - y1
            local_mask = np.zeros((box_height, box_width), dtype=np.uint8)

            if class_name == "person":
                cx = box_width // 2
                head_center = (cx, max(4, int(box_height * 0.14)))
                head_axes = (max(5, int(box_width * 0.18)), max(6, int(box_height * 0.12)))
                torso_center = (cx, int(box_height * 0.48))
                torso_axes = (max(8, int(box_width * 0.34)), max(10, int(box_height * 0.31)))
                leg_y1 = int(box_height * 0.56)
                leg_y2 = max(leg_y1 + 1, int(box_height * 0.98))
                leg_half = max(3, int(box_width * 0.10))
                leg_gap = max(2, int(box_width * 0.07))

                cv2.ellipse(local_mask, head_center, head_axes, 0, 0, 360, 255, -1)
                cv2.ellipse(local_mask, torso_center, torso_axes, 0, 0, 360, 255, -1)
                cv2.rectangle(local_mask, (cx - leg_gap - leg_half, leg_y1), (cx - leg_gap, leg_y2), 255, -1)
                cv2.rectangle(local_mask, (cx + leg_gap, leg_y1), (cx + leg_gap + leg_half, leg_y2), 255, -1)

                arm_y1 = int(box_height * 0.25)
                arm_y2 = int(box_height * 0.62)
                arm_half = max(3, int(box_width * 0.08))
                cv2.rectangle(local_mask, (max(0, cx - int(box_width * 0.42)), arm_y1), (max(0, cx - int(box_width * 0.29) + arm_half), arm_y2), 255, -1)
                cv2.rectangle(local_mask, (min(box_width, cx + int(box_width * 0.29) - arm_half), arm_y1), (min(box_width, cx + int(box_width * 0.42)), arm_y2), 255, -1)
            else:
                inset_x = max(1, int(box_width * 0.05))
                inset_y = max(1, int(box_height * 0.07))
                cv2.rectangle(local_mask, (inset_x, inset_y), (box_width - inset_x, box_height - inset_y), 255, -1)

            kernel = np.ones((5, 5), np.uint8)
            local_mask = cv2.morphologyEx(local_mask, cv2.MORPH_CLOSE, kernel)
            full_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = local_mask
            highlight_mask = cv2.max(highlight_mask, full_mask)
            current_instances.append(ThermalInstance(class_name=class_name, box=(x1, y1, x2, y2), mask=full_mask))

    predicted_mask = update_occlusion_tracks(current_instances, frame.shape)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dim_background = cv2.convertScaleAbs(gray, alpha=0.25, beta=8)
    thermal = cv2.cvtColor(dim_background, cv2.COLOR_GRAY2BGR)

    if np.any(predicted_mask):
        ghost_mask = cv2.GaussianBlur(predicted_mask, (19, 19), 0)
        ghost_float = (ghost_mask.astype(np.float32) / 255.0)[:, :, None] * 0.45
        ghost_fill = np.full_like(thermal, (150, 150, 150))
        thermal = (thermal.astype(np.float32) * (1.0 - ghost_float) + ghost_fill.astype(np.float32) * ghost_float)
        thermal = np.clip(thermal, 0, 255).astype(np.uint8)
        ghost_contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(thermal, ghost_contours, -1, (190, 190, 190), 1)
        cv2.putText(thermal, "PREDICTED OCCLUDED TARGET", (12, max(54, frame.shape[0] - 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (190, 190, 190), 1, cv2.LINE_AA)

    if np.any(highlight_mask):
        soft_mask = cv2.GaussianBlur(highlight_mask, (11, 11), 0)
        mask_float = (soft_mask.astype(np.float32) / 255.0)[:, :, None]

        bright_gray = cv2.convertScaleAbs(gray, alpha=0.65, beta=120)
        bright_human = cv2.cvtColor(bright_gray, cv2.COLOR_GRAY2BGR)
        glow = np.full_like(bright_human, (245, 245, 245))
        bright_human = cv2.addWeighted(bright_human, 0.55, glow, 0.45, 0)

        thermal = (thermal.astype(np.float32) * (1.0 - mask_float) + bright_human.astype(np.float32) * mask_float)
        thermal = np.clip(thermal, 0, 255).astype(np.uint8)

        contours, _ = cv2.findContours(highlight_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(thermal, contours, -1, (255, 255, 255), 2)

    return thermal


def draw_selected_normal_view(frame: np.ndarray, results: Any, model: YOLO) -> np.ndarray:
    annotated = frame.copy()
    selected = SELECTED_NORMAL_CLASSES
    if not selected:
        return annotated

    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = str(model.names.get(cls_id, cls_id))
        if class_name not in selected:
            continue

        confidence = float(box.conf[0]) if box.conf is not None else 0.0
        x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
        color = (255, 255, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {confidence:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y1 = max(0, y1 - label_height - baseline - 5)
        cv2.rectangle(annotated, (x1, label_y1), (x1 + label_width + 8, label_y1 + label_height + baseline + 6), (0, 0, 0), -1)
        cv2.putText(annotated, label, (x1 + 4, label_y1 + label_height + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    return annotated


def main() -> int:
    global SHOW_THERMAL_HUMANS, SMALL_OBJECT_MODE, THERMAL_CLASS_NAMES, SELECTED_NORMAL_CLASSES
    global OCCLUSION_PREDICTION_ENABLED, OCCLUSION_PREDICTION_FRAMES, OCCLUSION_MIN_MISSING_FRAMES, OCCLUSION_TRACKS
    global OCCLUSION_MIN_TRACKED_FRAMES, OCCLUSION_MIN_VELOCITY, OCCLUSION_MAX_VELOCITY, OCCLUSION_MIN_AREA
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    prune_old_runs(keep=3)
    config = load_config()
    confidence = float(config.get("confidence", 0.35))
    img_size = int(config.get("img_size", 640))
    SMALL_OBJECT_MODE = bool(config.get("small_object_mode", False))
    small_object_img_size = int(config.get("small_object_img_size", 960))
    small_object_confidence = float(config.get("small_object_confidence", 0.25))
    inference_every_n_frames = max(1, int(config.get("inference_every_n_frames", 1)))
    confirm_frames = max(1, int(config.get("confirm_frames", 2)))
    OCCLUSION_PREDICTION_ENABLED = bool(config.get("occlusion_prediction", True))
    OCCLUSION_PREDICTION_FRAMES = max(0, int(config.get("occlusion_prediction_frames", 10)))
    OCCLUSION_MIN_MISSING_FRAMES = max(1, int(config.get("occlusion_min_missing_frames", 2)))
    OCCLUSION_MIN_TRACKED_FRAMES = max(1, int(config.get("occlusion_min_tracked_frames", 5)))
    OCCLUSION_MIN_VELOCITY = max(0.0, float(config.get("occlusion_min_velocity", 1.5)))
    OCCLUSION_MAX_VELOCITY = max(OCCLUSION_MIN_VELOCITY, float(config.get("occlusion_max_velocity", 35)))
    OCCLUSION_MIN_AREA = max(0, int(config.get("occlusion_min_area", 900)))
    preview_width = max(320, int(config.get("preview_width", 1280)))
    preview_height = max(240, int(config.get("preview_height", 720)))
    detect_classes = set(config.get("detect_classes", []))
    snapshot_classes = set(config.get("snapshot_classes", []))
    snapshot_cooldown = float(config.get("snapshot_cooldown_seconds", 3))

    window = choose_window(config.get("window_title_hint", ""))
    capture = WindowCapture(window, config.get("crop", {}))
    region = capture.current_region()
    print(f"\nCapturing: {window.title}")
    print(f"Run: {RUN_ID}")
    print(f"Saving run data in: {RUN_DIR}")
    if capture.direct_available:
        print("Capture mode: selected-window direct capture")
    else:
        print("Capture mode: visible screen rectangle fallback")
    print(f"Region: left={region.left}, top={region.top}, width={region.width}, height={region.height}")
    print("Controls: q quit | s snapshot | p pause | m thermal | c/right-click classes | o small-object | g ghost-predict | +/- confidence")
    print("Mouse: use only the top buttons for Thermal, Classes, and Small Obj.\n")

    model = YOLO(resolve_model(str(config.get("model", "yolov8n.pt"))))
    try:
        model.fuse()
    except Exception:
        pass
    class_name_to_id = {name: idx for idx, name in model.names.items()}
    THERMAL_CLASS_NAMES = [str(model.names[idx]) for idx in sorted(model.names)]
    if not SELECTED_NORMAL_CLASSES:
        SELECTED_NORMAL_CLASSES = set(THERMAL_CLASS_NAMES)
    class_ids = None
    missing_classes = sorted(name for name in detect_classes if name not in class_name_to_id)
    if missing_classes:
        print("This model does not have these classes, so they will be skipped:")
        print("  " + ", ".join(missing_classes))
    print("Detecting all model classes so the thermal class picker can choose any available class.")

    paused = False
    last_snapshot_at = 0.0
    last_frame_time = time.perf_counter()
    fps = 0.0
    frame_index = 0
    last_results = None
    previous_detections: list[Detection] = []

    place_preview_window(region)
    cv2.setMouseCallback(PREVIEW_WINDOW, handle_mouse)

    with mss.mss() as sct:
        while True:
            now = time.perf_counter()
            frame = capture.grab(sct)
            annotated = frame.copy()
            counts: dict[str, int] = {}
            detected_snapshot_classes: set[str] = set()

            if not paused:
                active_img_size = small_object_img_size if SMALL_OBJECT_MODE else img_size
                active_confidence = small_object_confidence if SMALL_OBJECT_MODE else confidence
                ran_inference = False
                if frame_index % inference_every_n_frames == 0 or last_results is None:
                    last_results = model.predict(
                        frame,
                        conf=active_confidence,
                        imgsz=active_img_size,
                        classes=class_ids,
                        verbose=False,
                    )[0]
                    ran_inference = True
                results = last_results
                detections = extract_detections(results, model)
                if ran_inference:
                    previous_detections, confirmed_detections = confirm_detections(
                        detections,
                        previous_detections,
                        confirm_frames,
                    )
                else:
                    confirmed_detections = [
                        detection for detection in previous_detections
                        if detection.confirmed_frames >= confirm_frames
                    ]
                for detection in confirmed_detections:
                    counts[detection.class_name] = counts.get(detection.class_name, 0) + 1
                    if detection.class_name in snapshot_classes:
                        detected_snapshot_classes.add(detection.class_name)

                if SHOW_THERMAL_HUMANS:
                    annotated = thermal_highlight_view(frame, results, model)
                else:
                    annotated = draw_selected_normal_view(frame, results, model)

                if detected_snapshot_classes and now - last_snapshot_at >= snapshot_cooldown:
                    reason = "detected_" + "_".join(sorted(detected_snapshot_classes))
                    path = save_snapshot(
                        annotated,
                        reason,
                        {
                            "mode": "thermal" if SHOW_THERMAL_HUMANS else "normal",
                            "small_object_mode": SMALL_OBJECT_MODE,
                            "confirm_frames": confirm_frames,
                            "detections": [
                                {
                                    "class": detection.class_name,
                                    "confidence": round(detection.confidence, 3),
                                    "box": list(detection.box),
                                    "confirmed_frames": detection.confirmed_frames,
                                }
                                for detection in confirmed_detections
                                if detection.class_name in detected_snapshot_classes
                            ],
                        },
                    )
                    save_label_candidate(
                        frame,
                        reason,
                        [
                            detection for detection in confirmed_detections
                            if detection.class_name in detected_snapshot_classes
                        ],
                    )
                    print(f"Saved snapshot: {path}")
                    last_snapshot_at = now

            frame_delta = now - last_frame_time
            last_frame_time = now
            if frame_delta > 0:
                fps = (fps * 0.85) + ((1.0 / frame_delta) * 0.15)

            preview = compose_preview(annotated, preview_width, preview_height, fps, confidence, paused, counts)
            cv2.imshow(PREVIEW_WINDOW, preview)
            frame_index += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = save_snapshot(
                    annotated,
                    "manual",
                    {
                        "mode": "thermal" if SHOW_THERMAL_HUMANS else "normal",
                        "small_object_mode": SMALL_OBJECT_MODE,
                        "manual": True,
                    },
                )
                print(f"Saved snapshot: {path}")
            if key == ord("p"):
                paused = not paused
            if key == ord("m"):
                SHOW_THERMAL_HUMANS = not SHOW_THERMAL_HUMANS
                mode = "thermal-humans" if SHOW_THERMAL_HUMANS else "normal"
                print(f"View mode: {mode}")
            if key == ord("c"):
                show_class_menu()
                print("Thermal class menu opened.")
            if key == ord("o"):
                SMALL_OBJECT_MODE = not SMALL_OBJECT_MODE
                mode = "small-object" if SMALL_OBJECT_MODE else "standard"
                print(f"Object mode: {mode}")
            if key == ord("g"):
                OCCLUSION_PREDICTION_ENABLED = not OCCLUSION_PREDICTION_ENABLED
                OCCLUSION_TRACKS.clear()
                mode = "on" if OCCLUSION_PREDICTION_ENABLED else "off"
                print(f"Occlusion prediction: {mode}")
            if key in (ord("+"), ord("=")):
                confidence = min(0.95, confidence + 0.05)
                print(f"Confidence: {confidence:.2f}")
            if key == ord("-"):
                confidence = max(0.05, confidence - 0.05)
                print(f"Confidence: {confidence:.2f}")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)



