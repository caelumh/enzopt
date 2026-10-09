"""AC-15: `enzopt doctor` checks Python/package versions, CUDA and the Hopsan path, and names missing parts."""

from conftest import run_cli

from enzopt import doctor


def test_doctor_reports_components():
    checks = {c.name: c for c in doctor.run_checks(need_hopsan=False)}
    for name in ("python", "numpy", "scipy", "matplotlib", "torch", "cuda", "hopsan", "disk"):
        assert name in checks
    assert checks["numpy"].ok and checks["scipy"].ok


def test_doctor_cli_ok():
    r = run_cli("doctor")
    assert r.returncode == 0, r.stdout + r.stderr
    for name in ("python", "numpy", "scipy", "torch", "cuda", "hopsan"):
        assert name in r.stdout


def test_doctor_names_missing_hopsan(tmp_path):
    r = run_cli("doctor", "--hopsan", env={"HOPSANCLI": str(tmp_path / "no_such_hopsancli.exe")})
    assert r.returncode == 1
    assert "빠진 구성 요소" in r.stdout and "hopsan" in r.stdout.split("빠진 구성 요소")[-1]


def test_doctor_names_missing_package(monkeypatch):
    real = doctor._version
    monkeypatch.setattr(doctor, "_version", lambda m: None if m == "scipy" else real(m))
    checks = doctor.run_checks()
    assert doctor.missing(checks) == ["scipy"]
