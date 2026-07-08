from __future__ import annotations

import json
from pathlib import Path

from ultralytics import YOLO


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_YAML = PROJECT_DIR / "dataset" / "data.yaml"
CONFIG_PATH = PROJECT_DIR / "config.json"


def update_config_model(best_model: Path) -> None:
    if not CONFIG_PATH.exists():
        return
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    config["model"] = best_model.relative_to(PROJECT_DIR).as_posix()
    CONFIG_PATH.write_text(json.dumps(config, indent=2), encoding="utf-8")


def main() -> int:
    if not DATASET_YAML.exists():
        print("No dataset/data.yaml found. Run review_labels.py and export labels first.")
        return 1

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(DATASET_YAML),
        epochs=40,
        imgsz=640,
        project=str(PROJECT_DIR / "training_runs"),
        name="drone_custom",
        exist_ok=True,
    )
    print("Training complete.")
    best_model = PROJECT_DIR / "training_runs" / "drone_custom" / "weights" / "best.pt"
    if best_model.exists():
        update_config_model(best_model)
        print(f"Updated config.json to use: {best_model}")
    else:
        print("Training finished, but best.pt was not found at the expected path.")
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
