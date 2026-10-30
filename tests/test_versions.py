"""Tests of spaclone._versions: show_versions, check_install, find_tool and the extras table."""

import stat
import sys
from pathlib import Path

import pytest

from spaclone import _versions as V


def _pyproject() -> dict:
    path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not path.exists():
        pytest.skip("pyproject.toml not next to the tests (installed test suite)")
    import tomllib

    return tomllib.loads(path.read_text())


def test_extras_fallback_mirrors_pyproject():
    extras = _pyproject()["project"]["optional-dependencies"]
    umbrella = {k: v for k, v in extras.items() if all(s.startswith("spaclone[") for s in v)}
    plain = {k: v for k, v in extras.items() if k not in umbrella}
    assert {k: sorted(v) for k, v in plain.items()} == {k: sorted(v) for k, v in V.EXTRAS_FALLBACK.items()}
    for k, v in umbrella.items():
        inner = sorted(x.strip() for s in v for x in s.split("[", 1)[1].rstrip("]").split(","))
        assert inner == sorted(V.UMBRELLA_EXTRAS[k]), k
    assert set(V.UMBRELLA_EXTRAS) == set(umbrella)


def test_every_extra_is_reported():
    reported = {f.extra for f in V.FEATURES if f.extra}
    assert set(V.EXTRAS_FALLBACK) <= reported
    assert {f.name for f in V.FEATURES if f.kind != "python"} == {
        "IgBLAST",
        "Change-O",
        "IgPhyML",
        "R + Immcantation",
        "R + sciCSR",
    }


def test_check_install_report_shape():
    rep = V.check_install(deep=False, external=False)
    t = rep.table
    assert list(t.columns) == ["feature", "kind", "available", "version", "extra", "enable", "used_by", "detail"]
    assert len(t) == len(V.FEATURES)
    py = t[t.kind == "python"]
    assert set(py.available) <= {"yes", "no", "partial"}
    assert py.enable.str.startswith("pip install ").all()
    assert set(t[t.kind != "python"].available) == {"not checked"}
    text = repr(rep)
    assert "optional features available" in text
    assert rep.missing.equals(t[t.available != "yes"])
    assert "<table" in rep._repr_html_()


def test_check_install_reports_broken_package(tmp_path, monkeypatch):
    (tmp_path / "spaclone_fake_broken.py").write_text("raise RuntimeError('built against another numpy')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    feat = V.Feature("fake", "python", ("spaclone_fake_broken",), "nothing", extra="images")
    monkeypatch.setattr(V, "FEATURES", (feat,))
    row = V.check_install(external=False).table.iloc[0]
    assert row.available == "broken" and "RuntimeError" in row.detail
    sys.modules.pop("spaclone_fake_broken", None)
    assert V.check_install(deep=False, external=False).table.iloc[0].available == "yes"


def _script(path: Path, body: str) -> Path:
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_external_tools_and_r_packages(tmp_path, monkeypatch):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    _script(bin_ / "igblastn", "echo 'igblastn: 1.22.0'\n")
    _script(bin_ / "igphyml", "exit 0\n")
    for n in ("MakeDb.py", "DefineClones.py", "CreateGermlines.py"):  # found, but its Python cannot import changeo
        _script(bin_ / n, "echo 'ModuleNotFoundError: No module named changeo' >&2; exit 1\n")
    _script(bin_ / "Rscript", "printf 'alakazam\\t1.5.0\\t\\nshazam\\tNA\\t\\nsciCSR\\t0.4.0\\t\\n'\n")
    monkeypatch.setenv("SPACLONE_IGBLAST", str(bin_))
    monkeypatch.setenv("SPACLONE_RSCRIPT", str(bin_ / "Rscript"))
    monkeypatch.setenv("SPACLONE_CHANGEO", str(bin_))
    monkeypatch.setenv("SPACLONE_IGPHYML", str(bin_ / "igphyml"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(sys, "executable", str(tmp_path / "empty" / "python"))  # no programs next to Python
    assert V.find_tool("igblastn") == bin_ / "igblastn"
    assert V.find_tool("igblastn", tmp_path / "nowhere") == bin_ / "igblastn"  # falls back to the variable
    t = V.check_install(deep=False).table.set_index("feature")
    assert t.loc["IgBLAST", "available"] == "yes" and t.loc["IgBLAST", "version"] == "1.22.0"
    assert t.loc["Change-O", "available"] == "broken" and "No module named changeo" in t.loc["Change-O", "detail"]
    assert t.loc["IgPhyML", "available"] == "yes"
    assert t.loc["R + Immcantation", "available"] == "partial"
    assert "missing: shazam" in t.loc["R + Immcantation", "detail"]
    assert t.loc["R + sciCSR", "available"] == "yes"


def test_show_versions(capsys):
    info = V.show_versions(external=False, as_dict=True)
    assert {"python", "platform", "spaclone", "dependencies", "optional", "external"} <= set(info)
    assert info["dependencies"]["numpy"] is not None
    assert V.show_versions(external=False) is None
    out = capsys.readouterr().out
    assert "spaclone" in out and "numpy" in out and "optional" in out
