# DJI Mirrored Live AI Detector

This watches a mirrored iPhone/DJI Fly window on Windows, runs YOLO object detection, and shows a separate annotated live view. Then the detections can be analyzed and edited to teach the model and train its detection sills.

## Setup

1. Open the project in PyCharm.
2. Use the project `.venv` or create one.
3. Install dependencies:

```powershell
pip install -r requirements.txt
```

The first run downloads `yolov8n.pt` automatically unless you put a model file in `models/` and update `config.json`.

## Flying Setup

1. Connect iPhone to the DJI controller like normal.
2. Open DJI Fly and get the live feed working.
3. Mirror the iPhone screen wirelessly to Windows using an AirPlay receiver app.
4. Make sure the mirrored screen is visible as a normal Windows window.
5. Run:

```powershell
python app.py
```

Pick the AirPlay/mirror window from the list. The app opens an annotated window.

## Controls

- `q`: quit
- `s`: save a snapshot manually
- `p`: pause/resume detection
- `m` or the top `Thermal` button: toggle thermal/highlight mode
- `c`, right-click, or the top `Classes` button: open the class menu for the current view
- `o`: toggle small-object mode for high-altitude drone views
- `g`: toggle thermal occlusion prediction/ghost tracking
- `+` / `=`: raise confidence threshold
- `-`: lower confidence threshold
- The class menu remembers separate selections for normal and thermal views.
- In the class menu, click classes to toggle them. Use `All` to select/deselect everything for the current view.
- Use the top buttons to toggle thermal, close the menu, clear selections, or select all.
- Regular clicks on the camera/video area do not change modes.
- If the `Classes` button misses because the window is scaled, press `c` or right-click anywhere in the detector window.

Thermal mode includes optional occlusion prediction. If a selected object disappears briefly behind another object, the app draws a dim predicted ghost silhouette based on its recent motion. It waits for the object to be missing for a short delay to reduce false ghosts. This is an estimate, not a real detection.
Ghost prediction is intentionally conservative: it only appears for objects that were tracked for multiple frames and are large enough. If the object was still, the ghost stays in place; if it was moving, the ghost continues along a similar velocity until the object resurfaces or the prediction timeout ends.

Automatic snapshots are saved in `runs/<run_id>/snapshots/` when configured target classes are detected.
Snapshots are only saved after the same class/object is confirmed across the configured number of frames.
The app keeps only the latest 3 run folders.

## Reports

After a flight, generate an HTML report from the latest 3 run folders:

```powershell
python generate_report.py
```

Reports are saved in `reports/`.
Only the latest 3 reports are kept.

## Label Review And Training

The detector saves clean label candidates in each run when confirmed detections happen:

```text
runs/<run_id>/label_candidates/
```

Review and correct labels:

```powershell
python review_labels.py
```

Controls:

- Click a detection box to rename it, for example `toothbrush` -> `scissors`
- Type the new label directly in the review window and press Enter
- Drag empty space to add a missed object box
- Click a box, then `d`: delete selected box
- `l`: change the default label used for new boxes
- `e`: export the current image labels
- `n`: export current image and go next
- `b`: go back
- `q`: quit

Reviewed labels are exported to `dataset/` in YOLO box format.

Train a custom model after you have enough reviewed labels:

```powershell
python train_custom.py
```

This creates training output in `training_runs/`. Use the printed `best.pt` path in `config.json` when you want the detector to use your custom model.
When training finishes successfully, `train_custom.py` automatically updates `config.json` to use the new `training_runs/drone_custom/weights/best.pt` model.

## Restore Point

A restore checkpoint was created before the labeling workflow was added:

```text
backups/before_labeling_20260707_205213/
```

To scrap the labeling workflow, restore `app.py`, `config.json`, `generate_report.py`, `README.md`, and `requirements.txt` from that folder.

## Small Object Mode

Press `o` during detection to use small-object mode. It uses a larger inference size and lower confidence threshold from `config.json`, which can help with tiny drone-view targets like people and vehicles.

## Notes

This is a visual assistant only. Keep flying from DJI Fly and maintain safe visual line of sight. Wireless mirroring can lag, so do not use the annotated laptop view as your only flight view.
