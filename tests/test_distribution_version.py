"""Version resolution must survive the wheel context — no pyproject.toml
on disk, metadata under the `iai-pme` distribution name. The source
checkout reads pyproject first, which masked exactly this defect on the
published wheel (`iai --version` -> `0+unknown`).
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError

import iai_mcp


def _fake_version(known: dict):
    def _version(dist: str) -> str:
        if dist in known:
            return known[dist]
        raise PackageNotFoundError(dist)
    return _version


def test_wheel_metadata_resolves_under_new_name(monkeypatch):
    monkeypatch.setattr(iai_mcp, "version", _fake_version({"iai-pme": "9.9.9"}))
    assert iai_mcp._distribution_version() == "9.9.9"


def test_pre_rename_editable_install_still_resolves(monkeypatch):
    monkeypatch.setattr(iai_mcp, "version", _fake_version({"iai-mcp": "2.6.1"}))
    assert iai_mcp._distribution_version() == "2.6.1"


def test_no_metadata_at_all_degrades_to_sentinel(monkeypatch):
    monkeypatch.setattr(iai_mcp, "version", _fake_version({}))
    assert iai_mcp._distribution_version() == "0+unknown"


def test_this_environment_resolves_a_real_version():
    # The dev venv carries metadata under one of the two names; the chain
    # must find it without touching pyproject.
    assert iai_mcp._distribution_version() != "0+unknown"
