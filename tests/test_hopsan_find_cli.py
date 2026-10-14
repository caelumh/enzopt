"""find_cli(): executable paths with spaces in HOPSANCLI and KIMM-SYS under HOPSAN_HOME."""

import os

import pytest

from enzopt.hopsan.cli import HopsanNotFound, find_cli


def _fake_exe(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return str(path)


@pytest.fixture
def spaced_exe(tmp_path):
    return _fake_exe(tmp_path / "Program Files (x86)" / "KIMM_SYS" / "bin" / "KIMM_SYS_cli.exe")


def test_hopsancli_unquoted_path_with_spaces(monkeypatch, spaced_exe):
    monkeypatch.setenv("HOPSANCLI", spaced_exe)
    assert find_cli() == [spaced_exe]


def test_hopsancli_quoted_path_with_spaces(monkeypatch, spaced_exe):
    monkeypatch.setenv("HOPSANCLI", f'  "{spaced_exe}"  ')
    assert find_cli() == [spaced_exe]


def test_hopsancli_quoted_path_with_extra_args(monkeypatch, spaced_exe):
    monkeypatch.setenv("HOPSANCLI", f'"{spaced_exe}" --extra')
    assert find_cli() == [spaced_exe, "--extra"]


def test_hopsancli_missing_path_still_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("HOPSANCLI", f'"{tmp_path / "Program Files" / "no_such_cli.exe"}"')
    with pytest.raises(HopsanNotFound):
        find_cli()


def test_hopsan_home_with_only_kimm_sys_cli(monkeypatch, tmp_path):
    home = tmp_path / "KIMM_SYS"
    exe = _fake_exe(home / "bin" / "KIMM_SYS_cli.exe")
    monkeypatch.delenv("HOPSANCLI", raising=False)
    monkeypatch.setenv("HOPSAN_HOME", str(home))
    assert find_cli() == [exe]


def test_hopsan_home_prefers_hopsancli(monkeypatch, tmp_path):
    home = tmp_path / "Hopsan"
    _fake_exe(home / "bin" / "KIMM_SYS_cli.exe")
    hopsancli = _fake_exe(home / "bin" / ("hopsancli.exe" if os.name == "nt" else "hopsancli"))
    monkeypatch.delenv("HOPSANCLI", raising=False)
    monkeypatch.setenv("HOPSAN_HOME", str(home))
    assert find_cli() == [hopsancli]
