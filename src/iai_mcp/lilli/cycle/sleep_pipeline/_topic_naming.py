from __future__ import annotations

import logging
from typing import Any, Callable
from uuid import UUID

from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep, _utc_now_iso
from iai_mcp.lilli.profile.community_names import (
    derive_community_name,
    derive_second_term,
    disambiguate,
    load_community_names,
    neutral_name,
    save_community_names,
)

logger = logging.getLogger(__name__)

_DECRYPT_BATCH = 400


def _empty_result() -> "dict[str, Any]":
    return {"communities_seen": 0, "names_persisted": 0}


class _UnionFind:
    def __init__(self, items: "list[str]") -> None:
        self._parent = {x: x for x in items}

    def find(self, x: str) -> str:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra


def _cluster_by_shared_members(
    members_by_cid: "dict[str, set[str]]",
) -> "list[list[str]]":
    """cids that share at least one member id are the SAME real topic under
    two different UUID stamps -- the load-bearing bridge between the stored
    and gate groupings. Without this, two cids that both derive "jazz" from
    overlapping-but-not-identical membership are treated as an ordinary
    collision and get pulled apart into different compound names, even
    though they name the same underlying community."""
    uf = _UnionFind(list(members_by_cid.keys()))
    member_owner: "dict[str, str]" = {}
    for cid, members in members_by_cid.items():
        for mid in members:
            owner = member_owner.get(mid)
            if owner is None:
                member_owner[mid] = cid
            else:
                uf.union(owner, cid)
    clusters: "dict[str, list[str]]" = {}
    for cid in members_by_cid:
        clusters.setdefault(uf.find(cid), []).append(cid)
    return list(clusters.values())


def step_community_naming(
    self, interrupt_check: "Callable[[], bool] | None",
) -> "tuple[bool, dict[str, Any]]":
    if self._check_interrupt(SleepStep.COMMUNITY_NAMING, 0, interrupt_check):
        return False, {}

    store = self._store
    if store is None:
        return True, _empty_result()

    try:
        from iai_mcp import core

        # Defensive warm: this step normally runs right after
        # RECALL_INDEX_REBUILD's own warm-up, but a standalone invocation
        # must still see a warm index rather than an empty one.
        try:
            store.lexical_search("community naming warm-up", k=1)
        except Exception as exc:  # noqa: BLE001 -- no index means no naming this cycle
            logger.debug("community_naming lexical warm-up failed: %s", exc)
            return True, _empty_result()

        idx = getattr(store, "_lexical_idx", None)
        if idx is None:
            return True, _empty_result()

        postings = idx.iter_token_postings()
        corpus_df = {tok: len(bucket) for tok, bucket in postings}
        n_docs = int(getattr(idx, "_n_docs", 0) or 0)

        # Both id namespaces feed ONE members map, merged by community id
        # first (a cid string shared by both groupings is one entry) --
        # the stored `record.community_id` and the live gate assignment are
        # independently-stamped UUID spaces, not guaranteed disjoint.
        members_by_cid: "dict[str, set[str]]" = {}

        try:
            for row in store.iter_record_columns(
                ["id", "community_id"], where="tombstoned_at IS NULL",
            ):
                rid_raw = row.get("id")
                if rid_raw is None:
                    continue
                rid = str(rid_raw)
                cid_raw = row.get("community_id")
                if cid_raw:
                    members_by_cid.setdefault(str(cid_raw), set()).add(rid)
        except Exception as exc:  # noqa: BLE001 -- degrade to gate-only naming
            logger.debug("community_naming stored-grouping scan failed: %s", exc)

        try:
            from iai_mcp import runtime_graph_cache

            assignment = runtime_graph_cache.load_recall_structural(store)[0]
            node_to_community = getattr(assignment, "node_to_community", None) or {}
            for node, comm in node_to_community.items():
                members_by_cid.setdefault(str(comm), set()).add(str(node))
        except Exception as exc:  # noqa: BLE001 -- degrade to stored-only naming
            logger.debug("community_naming gate-grouping load failed: %s", exc)

        if not members_by_cid:
            return True, _empty_result()

        # Decrypt every member's surface once, chunked and interrupt-aware,
        # so a foreground recall never waits behind the whole corpus.
        all_member_ids: "set[str]" = set()
        for members in members_by_cid.values():
            all_member_ids.update(members)
        ordered_ids = sorted(all_member_ids)

        surfaces: "dict[str, str]" = {}
        for i in range(0, len(ordered_ids), _DECRYPT_BATCH):
            if self._check_interrupt(SleepStep.COMMUNITY_NAMING, i, interrupt_check):
                return False, {}
            chunk = ordered_ids[i : i + _DECRYPT_BATCH]
            uuid_chunk: "list[UUID]" = []
            for x in chunk:
                try:
                    uuid_chunk.append(UUID(x))
                except (ValueError, AttributeError, TypeError):
                    continue
            batch = store.get_batch(uuid_chunk)
            for rid, rec in batch.items():
                surfaces[str(rid)] = rec.literal_surface or ""

        prior_blob = load_community_names(store) or {}
        prior_base = prior_blob.get("base_index") or {}
        prior_reverse = prior_blob.get("reverse_index") or {}

        # cids that share a member are the SAME topic in two namespaces --
        # one name is derived per CLUSTER, not per cid, so the namespace
        # bridge is structural, never an accident of matching top tokens.
        clusters = _cluster_by_shared_members(members_by_cid)

        base_by_root: "dict[str, str]" = {}
        second_by_root: "dict[str, str | None]" = {}
        provenance_by_root: "dict[str, dict]" = {}
        root_of_cid: "dict[str, str]" = {}
        derived_night = _utc_now_iso()

        for cluster_idx, cluster_cids in enumerate(clusters):
            if self._check_interrupt(
                SleepStep.COMMUNITY_NAMING, cluster_idx, interrupt_check,
            ):
                return False, {}
            root = min(cluster_cids)
            for cid in cluster_cids:
                root_of_cid[cid] = root

            cluster_members: "set[str]" = set()
            for cid in cluster_cids:
                cluster_members |= members_by_cid[cid]
            cluster_surfaces = [surfaces.get(mid, "") for mid in cluster_members]

            prior_candidate = next(
                (
                    prior_base.get(cid)
                    for cid in sorted(cluster_cids)
                    if prior_base.get(cid)
                ),
                None,
            )
            try:
                base_name = derive_community_name(
                    cluster_surfaces, corpus_df, n_docs, prior_name=prior_candidate,
                )
            except Exception as exc:  # noqa: BLE001 -- one bad cluster must not abort the night
                logger.debug(
                    "community_naming derivation failed for cluster %s: %s",
                    root, exc,
                )
                base_name = None

            if base_name is None:
                base_by_root[root] = neutral_name(root)
                second_by_root[root] = None
            else:
                base_by_root[root] = base_name
                # Sticky disambiguation term: only when the SAME cid kept
                # the SAME base name across nights does its prior compound
                # name (`music-jazz`) supply a candidate second term --
                # anchored on the known base string, never a bare split,
                # since neither the base nor the fallback `topic-<id8>`
                # shape can contain the disambiguator's own hyphen rule.
                prior_second_candidate: "str | None" = None
                prefix = f"{base_name}-"
                for cid in sorted(cluster_cids):
                    if prior_base.get(cid) != base_name:
                        continue
                    prior_full = prior_reverse.get(cid)
                    if prior_full and prior_full.startswith(prefix):
                        prior_second_candidate = prior_full[len(prefix):]
                        break
                try:
                    second_by_root[root] = derive_second_term(
                        cluster_surfaces, corpus_df, n_docs, exclude=base_name,
                        prior_name=prior_second_candidate,
                    )
                except Exception:  # noqa: BLE001 -- de-dup input is best-effort
                    second_by_root[root] = None

            provenance_by_root[root] = {
                "community_ids": sorted(cluster_cids),
                "member_count": len(cluster_members),
                "derived_night": derived_night,
            }

        final_by_root = disambiguate(base_by_root, second_by_root)

        reverse_index: "dict[str, str]" = {}
        base_index: "dict[str, str]" = {}
        provenance: "dict[str, dict]" = {}
        for cid, root in root_of_cid.items():
            reverse_index[cid] = final_by_root[root]
            base_index[cid] = base_by_root[root]
        for root, name in final_by_root.items():
            provenance[name] = provenance_by_root[root]

        persisted = save_community_names(
            store,
            reverse_index=reverse_index,
            provenance=provenance,
            base_index=base_index,
        )
        if persisted:
            core.set_community_names(reverse_index)

        return True, {
            "communities_seen": len(members_by_cid),
            "names_persisted": len(reverse_index) if persisted else 0,
        }

    except Exception as exc:  # noqa: BLE001 -- step must not crash the pipeline
        logger.warning(
            "community_naming step failed: %s", exc, exc_info=True,
        )
        return True, {"error": str(exc)[:200], **_empty_result()}
