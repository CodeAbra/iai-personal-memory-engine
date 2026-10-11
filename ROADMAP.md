# Roadmap

This is what I'm planning to work on next for **iai-memory**. It's a short list, and I'd rather keep it short than pad it with things I'm not sure about. Everything else is open for discussion, and help is welcome, especially on anything tagged **`help wanted`**.

No dates. It's a small project and I don't want to promise a timeline I can't keep.

## Now — Windows port

The daemon and hooks run on macOS, where I develop and test them. Linux and Windows lean on the community. Windows in particular has some rough edges right now: the daemon boot, the socket setup, the way hook commands get quoted. I want a version that just works on a fresh Windows 11 machine.

I don't run Windows myself, so this one really needs a contributor to drive it. If that's you, the open Windows issues are a good place to see what's broken: #109, #113, #118, #119, #121.

- **Status:** `help wanted`

## Next — Multilingual recall

Recall works best in English today. I want it to work as well when your memories live in another language.

Part of that is already in your hands. iai-memory can run any embedding model you point it at through a local endpoint (`IAI_MCP_EMBED_PROVIDER=http`), so you can swap in a model that understands your language right now. A strong multilingual model covers a lot of ground on its own, so the model itself usually isn't the missing piece.

What's missing is a way to know it actually works in your language. That's where a **gold set** helps: a handful of real questions paired with the memory that should come back. With one, I can measure recall in that language and keep it from sliding. If your language is where recall falls apart, a gold set is the most useful thing you can send.

- **Status:** planned · `help wanted`
- **Good first contribution:** a gold set for your language, plus the embedder config that worked for you

## Ideas

Nothing committed here yet. If there's something you want iai-memory to do, open an issue or start a discussion. That's how things move up into Now and Next.

---

*Want to take something on? Leave a comment on its issue first, so two people don't build the same thing.*
