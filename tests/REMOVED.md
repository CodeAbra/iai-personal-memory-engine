# Removed Tests Audit Trail

Tests removed as part of the Hippo storage backend swap (v8.0). Each
entry lists the file path, the symbols it tested, and the reason for
removal.

## test_lance_storage_maintenance.py (removed v8.0)

Tested `optimize_lance_storage()` and the `lance_storage_optimized` event
kind. Both replaced by `optimize_hippo_storage()` and the `hippo_compacted`
event kind. New coverage in `tests/test_hippo_storage.py`.

## test_self_heal_lance_versions.py (removed v8.0)

Tested `_maybe_self_heal_version_pileup` — the sleep-pipeline predicate that
detected runaway LanceDB `_versions/*.manifest` accumulation and triggered
forced compaction. Hippo (SQLite + hnswlib) has no version manifests; the
predicate and its surrounding code were deleted from the sleep pipeline in v8.0.

## test_daemon_startup_bounded_optimize.py (removed in earlier plan)

Tested `_run_bounded_startup_optimize` and the `IAI_MCP_SKIP_STARTUP_OPTIMIZE`
environment knob. Both were deleted as part of the Hippo migration; startup
compaction no longer exists because Hippo has no version manifests to drain.
Note: this file was already deleted before this audit trail was written.
New boot-health coverage is in
`tests/test_hippo_storage.py::test_boot_integrity_rebuild_when_hnsw_missing`.

## test_optimize_step_drains_edges_lance.py (removed v8.0)

Tested the v7.6.5 hot-fix self-heal-bypass behavior: when `edges.lance`
version counts exceeded 2x the base threshold the staleness check was skipped.
Hippo has no version pile-up scenario; the bypass code path and the associated
staleness predicate were deleted in v8.0.

## test_concurrency.py — ProcessLock API tests removed

The pure process-lifecycle-lock tests were removed: shared-vs-exclusive,
exclusive-then-exclusive, flock fd-close safety, multi-holder blocking, SIGKILL
auto-release, the `holds_exclusive_nb` cooperative-yield probe, and the
concurrent-exclusive mutual-exclusion test — plus the spawn-child helpers they
relied on. The lock-file-0o600 half of the permissions test was also removed.
Reason: the process-lifecycle lock was removed — a vestigial lifecycle lock with
zero live acquires; the storage lock is the sole contention authority. The
surviving socket control-plane tests (status round-trip, injected dispatcher,
stale-socket cleanup, socket-0o600) were rewritten to drop the lock plumbing.

## test_daemon.py — ProcessLock shutdown/yield tests removed

The shutdown test that asserted the daemon closed a process-lifecycle lock on
exit, and the cooperative-yield probe test that exercised `holds_exclusive_nb`,
were removed. Reason: the daemon no longer constructs or closes a
process-lifecycle lock, and the in-process yield gate is gone; shutdown
cleanliness is now owned by the lifecycle marker release.

## test_identity_audit.py — ProcessLock-monkeypatch half removed

The runtime half that monkeypatched the process-lifecycle lock's acquire methods
to a raiser (to prove the identity audit never took the lock) was removed: the
premise is moot once the class is gone. The static source-scan guard
(identity_audit.py does not import the concurrency lock) is retained and passes.

## Recovery

Each removed file is preserved in git history. To restore any of them:

    git log --diff-filter=D --summary -- tests/test_X.py
    git checkout <pre-removal-sha> -- tests/test_X.py

For tests removed from a file that still exists, restore from the pre-removal
commit:

    git show <pre-removal-sha>:tests/test_X.py
