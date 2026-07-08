from __future__ import annotations

import json
import shutil
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
RUNS_DIR = PROJECT_DIR / "runs"
REPORT_DIR = PROJECT_DIR / "reports"
KEEP_RUNS = 3
KEEP_REPORTS = 3


def latest_runs(keep: int = KEEP_RUNS) -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    runs = sorted(
        [path for path in RUNS_DIR.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    return runs[:keep]


def prune_old_runs(keep: int = KEEP_RUNS) -> None:
    if not RUNS_DIR.exists():
        return
    runs = sorted(
        [path for path in RUNS_DIR.iterdir() if path.is_dir()],
        key=lambda path: path.name,
        reverse=True,
    )
    for old_run in runs[keep:]:
        shutil.rmtree(old_run, ignore_errors=True)


def prune_old_reports(keep: int = KEEP_REPORTS) -> None:
    if not REPORT_DIR.exists():
        return
    reports = sorted(REPORT_DIR.glob("flight_report_*.html"), key=lambda path: path.name, reverse=True)
    for old_report in reports[keep:]:
        old_report.unlink(missing_ok=True)


def read_detection_logs(run_dirs: list[Path]) -> list[dict]:
    records: list[dict] = []
    for run_dir in run_dirs:
        log_path = run_dir / "detections.jsonl"
        if not log_path.exists():
            continue
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record.setdefault("run_id", run_dir.name)
                records.append(record)
    return records


def snapshot_files(run_dirs: list[Path]) -> list[Path]:
    snapshots: list[Path] = []
    for run_dir in run_dirs:
        snapshot_dir = run_dir / "snapshots"
        if snapshot_dir.exists():
            snapshots.extend(snapshot_dir.glob("*.jpg"))
    return sorted(snapshots, key=lambda path: path.stat().st_mtime)


def classes_for_record(record: dict) -> list[str]:
    classes = []
    for detection in record.get("detections", []):
        class_name = detection.get("class")
        if class_name:
            classes.append(str(class_name))
    if classes:
        return classes

    reason = str(record.get("reason", ""))
    if reason.startswith("detected_"):
        return [part for part in reason.removeprefix("detected_").split("_") if part]
    return []


def relative_to_report(path: Path) -> str:
    return path.relative_to(REPORT_DIR).as_posix() if path.is_relative_to(REPORT_DIR) else path.as_posix()


def build_report() -> Path:
    REPORT_DIR.mkdir(exist_ok=True)
    prune_old_runs()
    run_dirs = latest_runs()
    records = read_detection_logs(run_dirs)
    snapshots = snapshot_files(run_dirs)
    records_by_snapshot = {
        Path(record.get("snapshot", "")).resolve(): record
        for record in records
        if record.get("snapshot")
    }

    all_classes = []
    for record in records:
        all_classes.extend(classes_for_record(record))
    class_counts = Counter(all_classes)

    created_at = datetime.now()
    report_path = REPORT_DIR / f"flight_report_{created_at.strftime('%Y%m%d_%H%M%S')}.html"

    summary_items = "\n".join(
        f"<li><strong>{escape(name)}</strong>: {count}</li>"
        for name, count in sorted(class_counts.items())
    ) or "<li>No logged detections yet.</li>"

    timeline_rows = []
    for snapshot in snapshots:
        record = records_by_snapshot.get(snapshot.resolve(), {})
        timestamp = record.get("timestamp") or datetime.fromtimestamp(snapshot.stat().st_mtime).isoformat(timespec="seconds")
        run_id = record.get("run_id") or snapshot.parent.parent.name
        classes = ", ".join(classes_for_record(record)) or "manual/unknown"
        mode = record.get("mode", "unknown")
        small = "yes" if record.get("small_object_mode") else "no"
        rel = "../" + snapshot.relative_to(PROJECT_DIR).as_posix()
        timeline_rows.append(
            "<tr>"
            f"<td>{escape(timestamp)}</td>"
            f"<td>{escape(str(run_id))}</td>"
            f"<td>{escape(classes)}</td>"
            f"<td>{escape(str(mode))}</td>"
            f"<td>{small}</td>"
            f"<td><a href=\"{escape(rel)}\">{escape(snapshot.name)}</a></td>"
            "</tr>"
        )

    cards = []
    for snapshot in snapshots:
        record = records_by_snapshot.get(snapshot.resolve(), {})
        classes = ", ".join(classes_for_record(record)) or "manual/unknown"
        timestamp = record.get("timestamp") or datetime.fromtimestamp(snapshot.stat().st_mtime).isoformat(timespec="seconds")
        run_id = record.get("run_id") or snapshot.parent.parent.name
        rel = "../" + snapshot.relative_to(PROJECT_DIR).as_posix()
        cards.append(
            "<article class=\"card\">"
            f"<a href=\"{escape(rel)}\"><img src=\"{escape(rel)}\" alt=\"{escape(snapshot.name)}\"></a>"
            f"<h3>{escape(classes)}</h3>"
            f"<p>{escape(timestamp)} | run {escape(str(run_id))}</p>"
            f"<p class=\"file\">{escape(snapshot.name)}</p>"
            "</article>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Drone AI Flight Report</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #101214; color: #f2f2f2; }}
    header {{ padding: 28px 34px; background: #1b2024; border-bottom: 1px solid #30363c; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    h2 {{ margin-top: 28px; }}
    main {{ padding: 0 34px 36px; }}
    .stats {{ display: flex; gap: 18px; flex-wrap: wrap; margin-top: 18px; }}
    .stat {{ background: #1b2024; border: 1px solid #30363c; padding: 14px 16px; border-radius: 8px; min-width: 140px; }}
    .stat strong {{ display: block; font-size: 26px; }}
    ul {{ line-height: 1.8; }}
    table {{ width: 100%; border-collapse: collapse; background: #171b1f; }}
    th, td {{ border-bottom: 1px solid #30363c; padding: 10px; text-align: left; }}
    th {{ background: #242a30; }}
    a {{ color: #8fd3ff; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }}
    .card {{ background: #171b1f; border: 1px solid #30363c; border-radius: 8px; overflow: hidden; }}
    .card img {{ width: 100%; height: 180px; object-fit: cover; display: block; background: #000; }}
    .card h3 {{ margin: 12px 12px 4px; font-size: 16px; }}
    .card p {{ margin: 6px 12px 12px; color: #b8c0c8; }}
    .file {{ font-size: 12px; word-break: break-all; }}
  </style>
</head>
<body>
  <header>
    <h1>Drone AI Flight Report</h1>
    <div>Generated {escape(created_at.isoformat(timespec="seconds"))}</div>
  </header>
  <main>
    <section class="stats">
      <div class="stat"><strong>{len(snapshots)}</strong>Snapshots</div>
      <div class="stat"><strong>{len(records)}</strong>Logged Events</div>
      <div class="stat"><strong>{sum(class_counts.values())}</strong>Detections</div>
      <div class="stat"><strong>{len(run_dirs)}</strong>Runs</div>
    </section>

    <h2>Summary</h2>
    <ul>{summary_items}</ul>

    <h2>Timeline</h2>
    <table>
      <thead><tr><th>Time</th><th>Run</th><th>Classes</th><th>Mode</th><th>Small Object</th><th>Snapshot</th></tr></thead>
      <tbody>{''.join(timeline_rows) or '<tr><td colspan="6">No snapshots found.</td></tr>'}</tbody>
    </table>

    <h2>Snapshots</h2>
    <section class="grid">{''.join(cards) or '<p>No snapshots found.</p>'}</section>
  </main>
</body>
</html>
"""
    report_path.write_text(html, encoding="utf-8")
    prune_old_reports()
    return report_path


if __name__ == "__main__":
    path = build_report()
    print(f"Report created: {path}")
