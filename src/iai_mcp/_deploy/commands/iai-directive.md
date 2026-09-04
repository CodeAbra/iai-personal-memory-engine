---
name: iai-directive
description: Record a standing order the user typed themselves, as an explicit memory directive
argument-hint: "<standing order text>"
---

Run the following command exactly, substituting the user's typed rule for
`$ARGUMENTS`, and report its output:

```bash
iai capture --directive "$ARGUMENTS"
```

This is a thin wrapper over the CLI mechanism itself -- it adds no
privilege the CLI does not already grant. Do not paraphrase or summarize
the user's text before passing it; forward it verbatim as the quoted
argument.
