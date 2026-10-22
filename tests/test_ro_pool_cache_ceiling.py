"""Regression tests for the aggregate RO pager footprint ceiling.

`resolve_ro_slot_cache_kib` clamps the per-slot RO pager cache so that
`per_slot_kib * pool_size` cannot exceed `RO_AGGREGATE_CEILING_KIB`, closing
the gap where `IAI_MCP_RO_CONN_CACHE_SIZE_KIB` and `IAI_MCP_RO_POOL_SIZE` are
independently operator-configurable. Above `pool_size` roughly 64, the
per-slot `MIN_CACHE_KIB` floor wins over the aggregate ceiling instead of
resolving toward 0 (which the engine reads as an unbounded cache).
"""
from __future__ import annotations

import logging

import pytest

from iai_mcp.lillibrain.constants import (
    MIN_CACHE_KIB,
    RO_AGGREGATE_CEILING_KIB,
    RO_CONN_CACHE_KIB,
    resolve_ro_slot_cache_kib,
)

_DEFAULT_POOL_SIZE = 8


def test_default_pool_size_yields_unclamped_default_per_slot_kib(monkeypatch):
    monkeypatch.delenv("IAI_MCP_RO_CONN_CACHE_SIZE_KIB", raising=False)
    result = resolve_ro_slot_cache_kib(_DEFAULT_POOL_SIZE)
    assert result == RO_CONN_CACHE_KIB == 8192
    assert result * _DEFAULT_POOL_SIZE <= RO_AGGREGATE_CEILING_KIB


def test_oversized_env_value_is_clamped_to_keep_aggregate_bounded(monkeypatch):
    # 64 MiB/slot * 8 slots would be 512 MiB, well past the ceiling.
    monkeypatch.setenv("IAI_MCP_RO_CONN_CACHE_SIZE_KIB", "65536")
    result = resolve_ro_slot_cache_kib(_DEFAULT_POOL_SIZE)
    assert result * _DEFAULT_POOL_SIZE <= RO_AGGREGATE_CEILING_KIB
    assert result == RO_AGGREGATE_CEILING_KIB // _DEFAULT_POOL_SIZE
    assert result < 65536


def test_clamp_emits_warning_log(monkeypatch, caplog):
    monkeypatch.setenv("IAI_MCP_RO_CONN_CACHE_SIZE_KIB", "65536")
    with caplog.at_level(logging.WARNING, logger="iai_mcp.lillibrain.constants"):
        resolve_ro_slot_cache_kib(_DEFAULT_POOL_SIZE)
    assert any("clamped" in record.message for record in caplog.records)


def test_unclamped_value_emits_no_warning(monkeypatch, caplog):
    monkeypatch.delenv("IAI_MCP_RO_CONN_CACHE_SIZE_KIB", raising=False)
    with caplog.at_level(logging.WARNING, logger="iai_mcp.lillibrain.constants"):
        resolve_ro_slot_cache_kib(_DEFAULT_POOL_SIZE)
    assert not any("clamped" in record.message for record in caplog.records)


@pytest.mark.parametrize("pool_size", [8, 64, 65, 128, 1000, 300000])
def test_oversized_env_value_never_drops_below_floor(monkeypatch, pool_size):
    monkeypatch.setenv("IAI_MCP_RO_CONN_CACHE_SIZE_KIB", "65536")
    result = resolve_ro_slot_cache_kib(pool_size)
    assert result != 0
    assert result >= MIN_CACHE_KIB


@pytest.mark.parametrize("pool_size", [8, 64, 65, 128, 1000, 300000])
def test_default_env_never_drops_below_floor(monkeypatch, pool_size):
    monkeypatch.delenv("IAI_MCP_RO_CONN_CACHE_SIZE_KIB", raising=False)
    result = resolve_ro_slot_cache_kib(pool_size)
    assert result != 0
    assert result >= MIN_CACHE_KIB
    if pool_size == 8:
        assert result == 8192


def test_warns_when_aggregate_over_budget_even_at_floor(monkeypatch, caplog):
    # per_slot_ceiling (2048) < resolved (4096, itself the MIN_CACHE_KIB
    # floor of a sub-floor env request) -- the floor and the ceiling clamp
    # to the same returned value, but the aggregate is still over budget
    # and must still warn.
    monkeypatch.setenv("IAI_MCP_RO_CONN_CACHE_SIZE_KIB", "1024")
    with caplog.at_level(logging.WARNING, logger="iai_mcp.lillibrain.constants"):
        result = resolve_ro_slot_cache_kib(128)
    assert result == MIN_CACHE_KIB
    assert any("clamped" in record.message for record in caplog.records)
