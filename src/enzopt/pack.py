"""``enzopt pack <run folder>``: one zip with the input data, settings copy, figures, CSVs, Hopsan .hmf models and the
validation reports of a run (for the competition submission and the paper archive)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

SKIP_NAMES = {"progress.json", "CANCEL"}
SKIP_SUFFIXES = (".tmp",)


def pack_run(run_dir: Path, output: Path | None = None) -> Path:
    run_dir = Path(run_dir).resolve()
    man = run_dir / "manifest.json"
    if not man.exists():
        raise FileNotFoundError(f"실행 폴더가 아님(manifest.json 없음): {run_dir}")
    m = json.loads(man.read_text(encoding="utf-8"))
    out = Path(output) if output else run_dir.with_name(f"{m['project']['name']}_{run_dir.name}.zip")
    files = []
    for f in sorted(run_dir.rglob("*")):
        if not f.is_file() or f.name in SKIP_NAMES or ".tmp" in f.name or f.suffix in SKIP_SUFFIXES:
            continue
        if f.resolve() == out.resolve():
            continue
        files.append(f)
    contents = {
        "run": run_dir.name,
        "project": m["project"]["name"],
        "status": m.get("status"),
        "files": [str(f.relative_to(run_dir)).replace("\\", "/") for f in files],
    }
    root = f"{m['project']['name']}_{run_dir.name}"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f"{root}/{f.relative_to(run_dir).as_posix()}")
        z.writestr(f"{root}/PACK_CONTENTS.json", json.dumps(contents, indent=1, ensure_ascii=False))
    return out
