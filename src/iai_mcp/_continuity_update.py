def continuity_update(next_action, focus, session_id, store):
    from iai_mcp import working_tier

    # A caller-supplied empty string is a real clear, not "unset" --
    # isinstance("", str) is True, so this is distinct from the
    # is-None gate above and must reach the continuity-file guard
    # as an authorized downgrade (a retracted focus must not persist).
    explicit_clear = focus == "" or next_action == ""
    working_tier.update_task(
        next_action=next_action,
        focus=focus,
        session_id=session_id or "-",
        store=store,
        explicit_clear=explicit_clear,
    )
