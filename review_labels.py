from __future__ import annotations

import json
import shutil
import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_DIR / "runs"
DATASET_DIR = PROJECT_DIR / "dataset"
WINDOW_NAME = "Review Labels"
DELETE_KEYS = {3014656}
BAR_HEIGHT = 140
MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_IMAGE_HEIGHT = 820
MIN_REVIEW_WIDTH = 980


@dataclass
class LabelBox:
    class_name: str
    box: tuple[int, int, int, int]
    confidence: float = 0.0


class ReviewState:
    def __init__(self, items: list[Path], class_names: list[str]):
        self.items = items
        self.class_names = class_names
        self.index = 0
        self.boxes: list[LabelBox] = []
        self.image: np.ndarray | None = None
        self.image_path: Path | None = None
        self.drag_start: tuple[int, int] | None = None
        self.drag_current: tuple[int, int] | None = None
        self.default_class = "person" if "person" in class_names else class_names[0]
        self.selected_box: int | None = None
        self.text_mode: str | None = None
        self.text_buffer = ""
        self.display_scale = 1.0
        self.display_x_offset = 0
        self.message = "Click a box or draw a new one, type the real label, then press Enter. Press s to skip."


def start_box_label_edit(state: ReviewState, box_index: int) -> None:
    state.selected_box = box_index
    state.text_mode = "box_label"
    state.text_buffer = ""
    current = state.boxes[box_index].class_name
    state.message = f"Rename box {box_index + 1}. Current: {current}. Type label from scratch, then Enter. Esc keeps it."


def latest_candidate_jsons() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    run_dirs = sorted([path for path in RUNS_DIR.iterdir() if path.is_dir()], key=lambda path: path.name, reverse=True)[:3]
    items: list[Path] = []
    for run_dir in run_dirs:
        candidate_dir = run_dir / "label_candidates"
        if candidate_dir.exists():
            items.extend(candidate_dir.glob("*.json"))
    return sorted(items, key=lambda path: path.stat().st_mtime)


def load_class_names() -> list[str]:
    model_path = "yolov8n-seg.pt"
    config_path = PROJECT_DIR / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        model_path = str(config.get("model", model_path))
    model = YOLO(str(PROJECT_DIR / model_path) if (PROJECT_DIR / model_path).exists() else model_path)
    class_names = [str(model.names[idx]) for idx in sorted(model.names)]
    return merge_class_names(class_names, load_existing_dataset_classes())


def load_existing_dataset_classes() -> list[str]:
    data_yaml = DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        return []

    names: list[tuple[int, str]] = []
    in_names = False
    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.strip() == "names:":
            in_names = True
            continue
        if not in_names:
            continue
        if line and not line.startswith((" ", "\t")):
            break
        if ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        try:
            class_id = int(key.strip())
        except ValueError:
            continue
        class_name = value.strip().strip("'\"")
        if class_name:
            names.append((class_id, class_name))
    return [name for _, name in sorted(names)]


def merge_class_names(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for raw_name in group:
            name = normalize_label(raw_name)
            if name and name not in seen:
                seen.add(name)
                merged.append(name)
    return merged


def normalize_label(value: str) -> str:
    return " ".join(value.strip().lower().split())


def screen_size() -> tuple[int, int]:
    try:
        user32 = ctypes.windll.user32
        user32.SetProcessDPIAware()
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1400, 900


def load_item(state: ReviewState) -> None:
    item_path = state.items[state.index]
    payload = json.loads(item_path.read_text(encoding="utf-8"))
    image_path = Path(payload["image"])
    state.image_path = image_path
    state.image = cv2.imread(str(image_path))
    state.boxes = [
        LabelBox(
            class_name=str(detection.get("class", state.default_class)),
            box=tuple(int(v) for v in detection.get("box", [0, 0, 1, 1])),
            confidence=float(detection.get("confidence", 0.0)),
        )
        for detection in payload.get("detections", [])
    ]


def box_contains(box: tuple[int, int, int, int], x: int, y: int) -> bool:
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def draw(state: ReviewState) -> np.ndarray:
    assert state.image is not None
    canvas = state.image.copy()
    for i, label in enumerate(state.boxes):
        x1, y1, x2, y2 = label.box
        selected = i == state.selected_box
        color = (0, 255, 255) if selected else (255, 255, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = f"{i + 1}: {label.class_name}"
        cv2.rectangle(canvas, (x1, max(0, y1 - 24)), (x1 + 180, y1), (0, 0, 0), -1)
        cv2.putText(canvas, text, (x1 + 4, max(14, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    if state.drag_start and state.drag_current:
        cv2.rectangle(canvas, state.drag_start, state.drag_current, (0, 255, 255), 2)

    image_h, image_w = canvas.shape[:2]
    screen_w, screen_h = screen_size()
    review_w = max(MIN_REVIEW_WIDTH, min(screen_w, 1920))
    review_image_h = max(360, min(MAX_DISPLAY_IMAGE_HEIGHT, screen_h - BAR_HEIGHT - 120))
    scale = min(MAX_DISPLAY_WIDTH / image_w, review_image_h / image_h, 1.0)
    display_w = max(1, int(round(image_w * scale)))
    display_h = max(1, int(round(image_h * scale)))
    state.display_scale = scale
    if scale != 1.0:
        canvas = cv2.resize(canvas, (display_w, display_h), interpolation=cv2.INTER_AREA)

    frame_w = max(display_w, review_w)
    state.display_x_offset = max(0, (frame_w - display_w) // 2)
    image_stage = np.zeros((display_h, frame_w, 3), dtype=np.uint8)
    image_stage[:, state.display_x_offset:state.display_x_offset + display_w] = canvas

    bar = np.zeros((BAR_HEIGHT, frame_w, 3), dtype=np.uint8)
    selected = "-" if state.selected_box is None else str(state.selected_box + 1)
    status = f"{state.index + 1}/{len(state.items)} | selected: {selected} | default new-box label: {state.default_class}"
    controls = "Enter save label | n export+next | s skip | e export | b back | Delete/x delete | l default | q quit"
    cv2.putText(bar, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.putText(bar, controls, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (205, 205, 205), 1, cv2.LINE_AA)
    cv2.putText(bar, state.message, (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 220, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, f"display: {display_w}x{display_h} from {image_w}x{image_h} | aspect locked", (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
    if state.text_mode:
        prompt = "New label" if state.text_mode == "box_label" else "Default label"
        if state.text_mode == "box_label" and state.selected_box is not None:
            prompt = f"Box {state.selected_box + 1} label"
        cv2.rectangle(bar, (8, 114), (min(frame_w - 8, 760), 138), (40, 40, 40), -1)
        cv2.putText(bar, f"{prompt}: {state.text_buffer}_", (14, 133), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, image_stage])


def on_mouse(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
    state: ReviewState = userdata
    display_y = y - BAR_HEIGHT
    if state.image is None or display_y < 0:
        return
    scale = state.display_scale or 1.0
    image_h, image_w = state.image.shape[:2]
    display_w = max(1, int(round(image_w * scale)))
    display_x = x - state.display_x_offset
    if display_x < 0 or display_x >= display_w:
        return
    image_x = min(image_w - 1, max(0, int(round(display_x / scale))))
    image_y = min(image_h - 1, max(0, int(round(display_y / scale))))

    if event == cv2.EVENT_LBUTTONDOWN:
        for i, label in enumerate(state.boxes):
            if box_contains(label.box, image_x, image_y):
                start_box_label_edit(state, i)
                return
        state.selected_box = None
        state.drag_start = (image_x, image_y)
        state.drag_current = (image_x, image_y)
    elif event == cv2.EVENT_MOUSEMOVE and state.drag_start:
        state.drag_current = (image_x, image_y)
    elif event == cv2.EVENT_LBUTTONUP and state.drag_start:
        x1, y1 = state.drag_start
        x2, y2 = image_x, image_y
        x1, x2 = sorted((max(0, x1), max(0, x2)))
        y1, y2 = sorted((max(0, y1), max(0, y2)))
        if x2 - x1 > 8 and y2 - y1 > 8:
            state.boxes.append(LabelBox(class_name=state.default_class, box=(x1, y1, x2, y2)))
            start_box_label_edit(state, len(state.boxes) - 1)
        state.drag_start = None
        state.drag_current = None


def yolo_line(label: LabelBox, width: int, height: int, class_to_id: dict[str, int]) -> str:
    x1, y1, x2, y2 = label.box
    cx = ((x1 + x2) / 2) / width
    cy = ((y1 + y2) / 2) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_to_id[label.class_name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def export_current(state: ReviewState) -> None:
    assert state.image is not None and state.image_path is not None
    image_dir = DATASET_DIR / "images" / "train"
    label_dir = DATASET_DIR / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    output_image = image_dir / state.image_path.name
    shutil.copy2(state.image_path, output_image)

    used_classes = merge_class_names(state.class_names, [box.class_name for box in state.boxes])
    state.class_names = used_classes
    class_to_id = {name: i for i, name in enumerate(used_classes)}
    height, width = state.image.shape[:2]
    lines = [yolo_line(box, width, height, class_to_id) for box in state.boxes if box.class_name in class_to_id]
    (label_dir / f"{state.image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    data_yaml = DATASET_DIR / "data.yaml"
    names = "\n".join(f"  {i}: {name}" for name, i in class_to_id.items())
    data_yaml.write_text(f"path: {DATASET_DIR.as_posix()}\ntrain: images/train\nval: images/train\nnames:\n{names}\n", encoding="utf-8")
    state.message = f"Exported labels for {state.image_path.name}"
    print(state.message)


def delete_clicked_box(state: ReviewState) -> None:
    if state.selected_box is None or not (0 <= state.selected_box < len(state.boxes)):
        state.message = "Select a box first, then press d."
        return
    removed = state.boxes.pop(state.selected_box)
    state.selected_box = None
    state.text_mode = None
    state.text_buffer = ""
    state.message = f"Deleted {removed.class_name}"
    print(state.message)


def move_to_next(state: ReviewState, exported: bool = False) -> None:
    if state.index >= len(state.items) - 1:
        state.message = "Already on the last snapshot."
        return
    state.index += 1
    load_item(state)
    if not exported:
        state.message = "Skipped this snapshot. Nothing was exported or added to training data."


def apply_text_entry(state: ReviewState) -> None:
    value = normalize_label(state.text_buffer)
    if state.text_mode == "box_label" and state.selected_box is not None and value:
        if value not in state.class_names:
            state.class_names.append(value)
        state.boxes[state.selected_box].class_name = value
        state.message = f"Box {state.selected_box + 1} label set to {value}."
    elif state.text_mode == "default_label" and value:
        if value not in state.class_names:
            state.class_names.append(value)
        state.default_class = value
        state.message = f"Default new-box label set to {value}."
    else:
        state.message = "No label typed, so the old label was kept."
    state.text_mode = None
    state.text_buffer = ""


def handle_text_key(state: ReviewState, key: int) -> bool:
    if not state.text_mode:
        return False
    if key in DELETE_KEYS or key == ord("x"):
        delete_clicked_box(state)
        return True
    if key in (13, 10):
        apply_text_entry(state)
        return True
    if key == 27:
        state.text_mode = None
        state.text_buffer = ""
        state.message = "Label edit cancelled."
        return True
    if key in (8, 127):
        state.text_buffer = state.text_buffer[:-1]
        return True
    if 32 <= key <= 126:
        state.text_buffer += chr(key)
        return True
    return True


def main() -> int:
    items = latest_candidate_jsons()
    if not items:
        print("No label candidates found yet. Run the detector until it saves detections first.")
        return 0

    state = ReviewState(items, load_class_names())
    load_item(state)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, state)

    while True:
        cv2.imshow(WINDOW_NAME, draw(state))
        key = cv2.waitKeyEx(15)
        if key < 0:
            continue
        if key in DELETE_KEYS or key == ord("x"):
            delete_clicked_box(state)
            continue
        if handle_text_key(state, key):
            continue
        if key == ord("q"):
            break
        if key == ord("n"):
            export_current(state)
            move_to_next(state, exported=True)
        if key == ord("s"):
            move_to_next(state)
        if key == ord("b"):
            state.index = max(0, state.index - 1)
            load_item(state)
        if key == ord("e"):
            export_current(state)
        if key == ord("d"):
            delete_clicked_box(state)
        if key == ord("l"):
            state.text_mode = "default_label"
            state.text_buffer = state.default_class
            state.message = "Type default label for new boxes, then Enter. Esc cancels."

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
