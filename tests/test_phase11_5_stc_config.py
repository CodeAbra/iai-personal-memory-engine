from __future__ import annotations

import dataclasses

import pytest

from iai_mcp.daemon import (
    StcConfig,
    _load_stc_config,
)

@pytest.fixture(autouse=True)
def _isolate_stc_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "IAI_MCP_PERI_EVENT_BUFFER_SIZE",
        "IAI_MCP_PERI_EVENT_WINDOW_SEC",
        "IAI_MCP_STC_STRONG_EVENT_TYPES",
        "IAI_MCP_STC_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)

def test_T1_stcconfig_is_frozen_dataclass_with_four_typed_fields() -> None:
    assert dataclasses.is_dataclass(StcConfig)
    field_names = [f.name for f in dataclasses.fields(StcConfig)]
    assert field_names == [
        "peri_event_buffer_size",
        "peri_event_window_sec",
        "strong_event_types",
        "dry_run",
    ], field_names

    cfg = StcConfig(
        peri_event_buffer_size=20,
        peri_event_window_sec=1800,
        strong_event_types=frozenset({"memory_capture"}),
        dry_run=False,
    )
    assert cfg.peri_event_buffer_size == 20
    assert cfg.peri_event_window_sec == 1800
    assert cfg.strong_event_types == frozenset({"memory_capture"})
    assert cfg.dry_run is False

def test_T1_stcconfig_is_immutable_frozen() -> None:
    cfg = StcConfig(
        peri_event_buffer_size=20,
        peri_event_window_sec=1800,
        strong_event_types=frozenset({"memory_capture"}),
        dry_run=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.peri_event_buffer_size = 99  # type: ignore[misc]

def test_T2_defaults_when_no_env_vars_set_under_pytest() -> None:
    cfg = _load_stc_config()
    assert cfg.peri_event_buffer_size == 20
    assert cfg.peri_event_window_sec == 1800
    assert cfg.strong_event_types == frozenset({
        "memory_capture", "error_trace", "user_correction",
    })
    assert cfg.dry_run is True

def test_T2_explicit_dry_run_false_overrides_pytest_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_STC_DRY_RUN", "false")
    cfg = _load_stc_config()
    assert cfg.dry_run is False

def test_T2_dry_run_vocab_accepts_synonyms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for v in ("true", "1", "yes", "on", "TRUE", "Yes"):
        monkeypatch.setenv("IAI_MCP_STC_DRY_RUN", v)
        assert _load_stc_config().dry_run is True
    for v in ("false", "0", "no", "off", "FALSE", ""):
        monkeypatch.setenv("IAI_MCP_STC_DRY_RUN", v)
        assert _load_stc_config().dry_run is False

def test_T2_strong_event_types_csv_strip_and_lowercase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "IAI_MCP_STC_STRONG_EVENT_TYPES",
        " Memory_Capture , ERROR_TRACE,  user_correction ",
    )
    cfg = _load_stc_config()
    assert cfg.strong_event_types == frozenset({
        "memory_capture", "error_trace", "user_correction",
    })
    assert isinstance(cfg.strong_event_types, frozenset)

def test_T2_call_on_demand_rereads_env_between_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_PERI_EVENT_BUFFER_SIZE", "5")
    assert _load_stc_config().peri_event_buffer_size == 5
    monkeypatch.setenv("IAI_MCP_PERI_EVENT_BUFFER_SIZE", "100")
    assert _load_stc_config().peri_event_buffer_size == 100

@pytest.mark.parametrize(
    "env_var, bad_value",
    [
        ("IAI_MCP_PERI_EVENT_BUFFER_SIZE", "not-an-int"),
        ("IAI_MCP_PERI_EVENT_BUFFER_SIZE", "0"),
        ("IAI_MCP_PERI_EVENT_BUFFER_SIZE", "-1"),
        ("IAI_MCP_PERI_EVENT_BUFFER_SIZE", "1001"),
        ("IAI_MCP_PERI_EVENT_BUFFER_SIZE", "99999"),
        ("IAI_MCP_PERI_EVENT_WINDOW_SEC", "not-an-int"),
        ("IAI_MCP_PERI_EVENT_WINDOW_SEC", "0"),
        ("IAI_MCP_PERI_EVENT_WINDOW_SEC", "-100"),
        ("IAI_MCP_PERI_EVENT_WINDOW_SEC", "86401"),
        ("IAI_MCP_STC_STRONG_EVENT_TYPES", ""),
        ("IAI_MCP_STC_STRONG_EVENT_TYPES", ","),
        ("IAI_MCP_STC_STRONG_EVENT_TYPES", "a,,b"),
        ("IAI_MCP_STC_STRONG_EVENT_TYPES", "  ,  "),
        ("IAI_MCP_STC_DRY_RUN", "banana"),
        ("IAI_MCP_STC_DRY_RUN", "maybe"),
    ],
)
def test_T2_R5_invalid_env_var_raises_ValueError_naming_the_var(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str,
) -> None:
    monkeypatch.setenv(env_var, bad_value)
    with pytest.raises(ValueError, match=env_var):
        _load_stc_config()
