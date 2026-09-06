---
name: iai-directive
description: Record, list, or remove a standing order the user typed themselves
argument-hint: "<standing order text> | list | remove <id>"
---

This command has three forms, dispatched on `$ARGUMENTS`. Do not
paraphrase or summarize the user's text before passing it; forward it
verbatim as the quoted argument. Each form is a thin wrapper over its own
CLI/chat mechanism -- it adds no privilege that mechanism does not already
grant.

## Set a standing order

If `$ARGUMENTS` is not `list` and does not start with `remove `, run the
following command exactly, substituting the user's typed rule for
`$ARGUMENTS`, and report its output:

```bash
iai capture --directive "$ARGUMENTS"
```

## List live standing orders

If `$ARGUMENTS` is exactly `list`, run:

```bash
iai directive list
```

and report its output verbatim.

## Remove a standing order by id

If `$ARGUMENTS` is `remove <id>`, do not run any command and do not send
the removal marker yourself -- only a genuine turn the user themselves
types can retire a directive. Instead, tell the user to type (or paste)
the following line as their own next message, substituting the given
`<id>`:

```
remove directive: <id>
```

This mirrors how setting a directive relies on the user's own typed
`standing directive:` prefix -- the removal marker must likewise
originate from the user's own message, never from an assistant reply.
