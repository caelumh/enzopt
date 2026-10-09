"""enzopt evaluation harness (evals/EVAL.md made executable). Frozen after milestone 1: do not edit.

    uv run python evals/run.py --tier quick     # gates + quick criteria (≤ 5 min)
    uv run python evals/run.py --tier long      # gates + all criteria + Q1 (≤ 30 min)

Gates  G1 `uv sync --locked` · G2 ruff check + format --check · G3 `pytest -m "not long"` · G4 AC-01/AC-02 did not
       turn from pass to fail compared with the previous results.tsv row.
Criteria AC-01..AC-20 are mapped to pytest node-id prefixes (EVAL.md "검사 방법"); AC-10 runs `pytest tests/core`
       in the KIMM-CAD compatible env (.venv-kimmcad). A criterion passes when at least one matching test ran and
       every matching test passed (skips count as not passed). Criteria without tests yet count as failed.
Metric Q1 = wall time of `enzopt run projects/example_ldh_synth` in full mode without cache (long tier only).
Score  70 × (passed weight / 47) + 30 × min(1, 15 min / Q1). The quick tier reuses the last long result for
       long-only criteria and Q1 (evals/last_long.json).
Writes evals/last.json (+ evals/last_long.json for the long tier) and appends one row to evals/results.tsv.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"
WORK = EVALS / "_work"
RESULTS = EVALS / "results.tsv"
LAST = EVALS / "last.json"
LAST_LONG = EVALS / "last_long.json"
EXAMPLE = ROOT / "projects" / "example_ldh_synth"
COMPAT = ROOT / ".venv-kimmcad"
COMPAT_PINS = ["numpy==1.26.4", "scipy==1.13.1", "matplotlib==3.9.1", "pytest"]
BUDGET = {"quick": 300.0, "long": 1800.0}
Q1_TARGET_S = 15 * 60.0

# id: (weight, tiers in which it is evaluated, node-id prefixes)
CRITERIA = {
    "AC-01": (3, ("quick", "long"), ["tests/test_regression.py::test_batch149"]),
    "AC-02": (3, ("quick", "long"), ["tests/test_regression.py::test_design"]),
    "AC-03": (3, ("quick",), ["tests/test_crnt.py"]),
    "AC-04": (3, ("quick",), ["tests/test_io.py"]),
    "AC-05": (3, ("long",), ["tests/test_pinn.py::test_realistic_holdout"]),
    "AC-06": (3, ("quick",), ["tests/test_e2e.py::test_e2e"]),
    "AC-07": (2, ("quick",), ["tests/test_validate.py"]),
    "AC-08": (2, ("long",), ["tests/test_pinn.py::test_gpu_vs_cpu"]),
    "AC-09": (2, ("quick",), ["tests/test_runs.py"]),
    "AC-10": (2, ("quick",), ["@compat:tests/core"]),
    "AC-11": (2, ("long",), ["tests/test_hopsan.py"]),
    "AC-12": (2, ("quick",), ["tests/test_figures.py"]),
    "AC-13": (2, ("quick",), ["tests/test_e2e.py::test_non_wr"]),
    "AC-14": (1, ("quick",), ["tests/test_pack.py"]),
    "AC-15": (1, ("quick",), ["tests/test_doctor.py"]),
    "AC-16": (3, ("long",), ["tests/test_baseline_fit.py"]),
    "AC-17": (3, ("long",), ["tests/test_control.py"]),
    "AC-18": (2, ("long",), ["tests/test_design_check.py"]),
    "AC-19": (3, ("quick",), ["tests/test_dashboard.py::test_dashboard"]),
    "AC-20": (2, ("quick",), ["tests/test_dashboard.py::test_detached_run"]),
}
TOTAL_WEIGHT = sum(w for w, _, _ in CRITERIA.values())  # 47
REGRESSION = ("AC-01", "AC-02")


def sh(cmd, env=None, timeout=None, cwd=ROOT) -> tuple[int, str, float]:
    t0 = time.time()
    e = dict(os.environ)
    e["PYTHONUTF8"] = "1"
    if env:
        e.update(env)
    try:
        r = subprocess.run(
            cmd, cwd=cwd, env=e, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout
        )
        return r.returncode, (r.stdout or "") + (r.stderr or ""), time.time() - t0
    except subprocess.TimeoutExpired as ex:
        return 124, f"timeout after {timeout} s: {ex}", time.time() - t0
    except OSError as ex:
        return 127, str(ex), time.time() - t0


def junit_cases(xml_path: Path) -> list[dict]:
    if not xml_path.exists():
        return []
    out = []
    for tc in ET.parse(xml_path).getroot().iter("testcase"):
        cls, name = tc.get("classname", ""), tc.get("name", "")
        parts = cls.split(".")
        # module path from classname (drop a trailing class name when present)
        mod = parts
        for i in range(len(parts), 0, -1):
            if (ROOT / ("/".join(parts[:i]) + ".py")).exists():
                mod = parts[:i]
                break
        node = "/".join(mod) + ".py::" + name
        status = "passed"
        if tc.find("failure") is not None or tc.find("error") is not None:
            status = "failed"
        elif tc.find("skipped") is not None:
            status = "skipped"
        out.append({"node": node, "status": status, "time": float(tc.get("time", 0) or 0)})
    return out


def ensure_compat() -> tuple[bool, str]:
    py = COMPAT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not py.exists():
        rc, out, _ = sh(["uv", "venv", str(COMPAT), "--python", "3.12", "-q"])
        if rc != 0:
            return False, out
    rc, out, _ = sh(["uv", "pip", "install", "--python", str(py), "-q", *COMPAT_PINS])
    if rc != 0:
        return False, out
    rc, out, _ = sh(
        [
            str(py),
            "-c",
            "import numpy, scipy, matplotlib, sys; print(sys.version.split()[0], numpy.__version__,"
            " scipy.__version__, matplotlib.__version__)",
        ]
    )
    return rc == 0, out.strip()


def run_compat_tests() -> dict:
    ok, info = ensure_compat()
    if not ok:
        return {"status": "failed", "detail": f"compat env: {info[-500:]}"}
    if not (ROOT / "tests" / "core").exists():
        return {"status": "missing", "detail": "tests/core 없음"}
    py = COMPAT / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    xml = WORK / "compat.xml"
    xml.unlink(missing_ok=True)
    rc, out, sec = sh(
        [str(py), "-m", "pytest", "tests/core", "-q", "-p", "no:cacheprovider", "-o", "addopts=", f"--junitxml={xml}"],
        env={"PYTHONPATH": str(ROOT / "src")},
        timeout=900,
    )
    cases = junit_cases(xml)
    st = "passed" if rc == 0 and cases and all(c["status"] == "passed" for c in cases) else "failed"
    return {"status": st, "detail": f"{info} · {len(cases)} tests · {sec:.0f} s", "tail": out[-1500:]}


def _matches(node: str, prefix: str) -> bool:
    """File prefix ``tests/x.py`` matches every test in it; ``tests/x.py::test_a`` matches test names starting with
    ``test_a`` (parametrized ids and ``test_a_*`` variants included)."""
    if prefix.endswith(".py"):
        return node.startswith(prefix + "::")
    return node.startswith(prefix)


def evaluate(cases: list[dict], tier: str, compat: dict | None, prev_long: dict | None) -> dict:
    res = {}
    for ac, (w, tiers, prefixes) in CRITERIA.items():
        if tier == "quick" and "quick" not in tiers:
            prev = (prev_long or {}).get("criteria", {}).get(ac, {})
            res[ac] = {"weight": w, "passed": bool(prev.get("passed")), "source": "last_long", "n": prev.get("n", 0)}
            continue
        if prefixes[0].startswith("@compat:"):
            st = (compat or {}).get("status", "missing")
            res[ac] = {
                "weight": w,
                "passed": st == "passed",
                "source": tier,
                "n": 0 if st == "missing" else 1,
                "detail": (compat or {}).get("detail"),
            }
            continue
        hit = [c for c in cases if any(_matches(c["node"], p) for p in prefixes)]
        passed = bool(hit) and all(c["status"] == "passed" for c in hit)
        res[ac] = {
            "weight": w,
            "passed": passed,
            "source": tier,
            "n": len(hit),
            "failed": [c["node"] for c in hit if c["status"] != "passed"],
        }
    return res


def previous_row() -> dict | None:
    if not RESULTS.exists():
        return None
    lines = [ln for ln in RESULTS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        return None
    head = lines[0].split("\t")
    return dict(zip(head, lines[-1].split("\t"), strict=False))


def git_commit() -> tuple[str, bool]:
    rc, out, _ = sh(["git", "rev-parse", "--short", "HEAD"])
    commit = out.strip() if rc == 0 else "-"
    rc, out, _ = sh(["git", "status", "--porcelain"])
    return commit, bool(out.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["quick", "long"], default="quick")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    tier = a.tier
    WORK.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log = {}

    # ---------------------------------------------------------------- gates G1–G2
    gates = {}
    rc, out, _ = sh(["uv", "sync", "--locked"], timeout=900)
    gates["G1"] = rc == 0
    log["G1"] = out[-800:]
    rc1, out1, _ = sh(["uv", "run", "--frozen", "ruff", "check", "."])
    rc2, out2, _ = sh(["uv", "run", "--frozen", "ruff", "format", "--check", "."])
    gates["G2"] = rc1 == 0 and rc2 == 0
    log["G2"] = out1[-800:] + out2[-800:]

    # ---------------------------------------------------------------- Q1 (long): full run, no cache
    q1 = None
    env = {}
    if tier == "long":
        runs_root = WORK / "runs_q1"
        t0 = time.time()
        rc, out, _ = sh(
            [sys.executable, "-m", "enzopt", "run", str(EXAMPLE), "--no-cache", "--runs-root", str(runs_root)],
            timeout=3600,
        )
        q1_s = time.time() - t0
        run_dir = None
        for line in out.splitlines():
            if line.startswith("실행 폴더:"):
                run_dir = line.split(":", 1)[1].strip()
        status = None
        if run_dir and Path(run_dir, "manifest.json").exists():
            status = json.loads(Path(run_dir, "manifest.json").read_text(encoding="utf-8")).get("status")
        q1 = q1_s if status == "completed" else None
        log["Q1"] = {"seconds": q1_s, "status": status, "run_dir": run_dir, "tail": out[-1500:]}
        if run_dir:
            env["ENZOPT_LONG_RUN_DIR"] = run_dir

    # ---------------------------------------------------------------- pytest (G3 + criteria)
    xml = WORK / f"{tier}.xml"
    xml.unlink(missing_ok=True)
    marker = ["-m", "not long"] if tier == "quick" else []
    rc, out, sec = sh([sys.executable, "-m", "pytest", "-q", *marker, f"--junitxml={xml}"], env=env, timeout=3600)
    cases = junit_cases(xml)
    quick_cases = [c for c in cases]  # in the long tier G3 is judged on the quick subset below
    log["pytest"] = {"rc": rc, "seconds": sec, "tail": out[-3000:]}
    if tier == "quick":
        gates["G3"] = rc == 0 and bool(cases)
    else:
        rcq, outq, _ = sh([sys.executable, "-m", "pytest", "-q", "-m", "not long", "--co"], timeout=600)
        quick_ids = set()
        for line in outq.splitlines():
            if "::" in line:
                quick_ids.add(line.strip().replace("\\", "/"))
        quick_cases = [c for c in cases if c["node"] in quick_ids] or cases
        gates["G3"] = bool(quick_cases) and all(c["status"] in ("passed", "skipped") for c in quick_cases) and rcq == 0

    compat = run_compat_tests()
    log["compat"] = compat
    prev_long = json.loads(LAST_LONG.read_text(encoding="utf-8")) if LAST_LONG.exists() else None
    crit = evaluate(cases, tier, compat, prev_long)

    # ---------------------------------------------------------------- G4 regression guard
    prev = previous_row()
    g4 = True
    if prev:
        before = set(filter(None, prev.get("passed", "").split(",")))
        for ac in REGRESSION:
            if ac in before and not crit[ac]["passed"]:
                g4 = False
    gates["G4"] = g4

    # ---------------------------------------------------------------- score
    if tier == "quick":
        q1 = (prev_long or {}).get("Q1_seconds")
    passed_w = sum(v["weight"] for v in crit.values() if v["passed"])
    crit_score = passed_w / TOTAL_WEIGHT
    metric = min(1.0, Q1_TARGET_S / q1) if q1 else 0.0
    score = 70 * crit_score + 30 * metric
    elapsed = time.time() - t_start
    commit, dirty = git_commit()
    passed = [k for k, v in crit.items() if v["passed"]]
    failed = [k for k, v in crit.items() if not v["passed"]]
    result = {
        "time": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "tier": tier,
        "commit": commit,
        "dirty": dirty,
        "gates": gates,
        "gates_ok": all(gates.values()),
        "criteria": crit,
        "criteria_score": crit_score,
        "Q1_seconds": q1,
        "metric": metric,
        "score": score,
        "elapsed_s": elapsed,
        "over_budget": elapsed > BUDGET[tier],
        "log": log,
    }
    LAST.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    if tier == "long":
        LAST_LONG.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    head = [
        "time",
        "tier",
        "commit",
        "dirty",
        "G1",
        "G2",
        "G3",
        "G4",
        "criteria_score",
        "Q1_s",
        "score",
        "passed",
        "failed",
        "elapsed_s",
        "note",
    ]
    row = [result["time"], tier, commit, str(dirty)] + ["P" if gates[g] else "F" for g in ("G1", "G2", "G3", "G4")]
    row += [
        f"{crit_score:.4f}",
        f"{q1:.0f}" if q1 else "-",
        f"{score:.2f}",
        ",".join(passed),
        ",".join(failed),
        f"{elapsed:.0f}",
        a.note,
    ]
    new = not RESULTS.exists()
    with open(RESULTS, "a", encoding="utf-8", newline="\n") as fh:
        if new:
            fh.write("\t".join(head) + "\n")
        fh.write("\t".join(row) + "\n")

    # ---------------------------------------------------------------- report
    print(
        f"enzopt 평가 ({tier}) · commit {commit}{' (수정 중)' if dirty else ''} · {elapsed:.0f} s"
        f"{' — 예산 초과' if result['over_budget'] else ''}"
    )
    print("관문: " + "  ".join(f"{g} {'통과' if ok else '실패'}" for g, ok in gates.items()))
    for ac, v in crit.items():
        src = "" if v["source"] == tier else " (직전 긴 평가)"
        mark = "통과" if v["passed"] else ("테스트 없음" if v.get("n", 0) == 0 else "실패")
        print(f"  {ac} [{v['weight']}] {mark}{src}" + (f" — {', '.join(v['failed'][:3])}" if v.get("failed") else ""))
    print(
        f"기준 점수 {crit_score:.3f} ({passed_w}/{TOTAL_WEIGHT}) · "
        f"Q1 {f'{q1:.0f} s' if q1 else '-'} · 지표 {metric:.3f}"
    )
    print(f"점수 {score:.2f}")
    return 0 if result["gates_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
