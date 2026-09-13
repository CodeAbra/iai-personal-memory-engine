def continuity_update(next_action, focus, session_id, store, goal=None):
    from iai_mcp import working_tier

    # A caller-supplied empty string is a real clear, not "unset" --
    # isinstance("", str) is True, so this is distinct from the
    # is-None gate above and must reach the continuity-file guard
    # as an authorized downgrade (a retracted focus must not persist).
    # goal is never part of this computation -- an empty/whitespace goal is
    # a no-op (working_tier.update_task's own gate), not a clear, so it must
    # never authorize the continuity file's downgrade path.
    explicit_clear = focus == "" or next_action == ""
    working_tier.update_task(
        goal=goal,
        next_action=next_action,
        focus=focus,
        session_id=session_id or "-",
        store=store,
        explicit_clear=explicit_clear,
    )
