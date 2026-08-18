# Handoff → Code

## Origen

Sesión de Chat en Claude Desktop, 17-18 de agosto de 2026. Jornada larga:
arrancó revisando un harness de terceros (descartado), siguió con el
diagnóstico y fix de tres bugs reales de arranque en Windows, una
verificación exhaustiva del propio README contra el comportamiento real,
el diseño (a nivel conversación) de un sistema de memoria alternativo con
Obsidian + LightRAG, y terminó investigando por qué la captura ambiental
no llega a Cowork. Este handoff cubre el estado consolidado de todo lo que
queda accionable sobre el fork en sí, no el diseño del sistema alternativo
(ese vive solo en la conversación, no en código todavía).

## Contexto

Tres bugs de arranque en Windows, diagnosticados con traceback real y
`py-spy`, ya corregidos y **mergeados en `main`**: guard de `SIGHUP`
(`daemon/__init__.py`), resolución de intérprete Python + guard de `fcntl`
en el hook de captura (dos copias sincronizadas), y guard de
`asyncio.start_unix_server` en `SocketServer.serve()` con un
`done_callback` que ahora expone excepciones en vez de morir en silencio.
Verificado end-to-end en la máquina real: `daemon status` limpio,
`capture`/`recall` funcionando vía daemon.

Aparte de eso, se hizo una verificación línea por línea del README contra
el comportamiento real (no solo lectura), y se investigó a fondo por qué
`iai-mcp cowork install` nunca logra activarse en esta instalación de
Desktop. Ambas líneas produjeron hallazgos reportados como issues, algunos
con trabajo de código pendiente, uno cerrado como no-accionable.

## Hallazgos verificados

- [Seguro] Los tres bugs de Windows están resueltos y mergeados: commits
  `7e576682`, `c94e996f`, `b31a489b`, `1107640`, `23727ff`, todos en
  `main`.
- [Seguro] `doctor` corre **30 checks reales**, el README documenta solo
  27: faltan `(ii) store embed identity`, `(aa) capture-state hygiene`,
  `(bb) nightly insight mint` en la tabla.
- [Seguro] La sección "Deployment surface" del README afirma "MCP
  transport is a Unix domain socket. No TCP listener, no bind address, no
  auth surface to misconfigure", lo cual es falso en Windows: hay TCP
  loopback + token (`.daemon.port`, `.daemon.token`, implementado en
  `_ipc.py`), sin que el README documente la excepción.
- [Seguro] `capture-hooks status` reporta `Claude Desktop: ... WIRED` de
  forma engañosa: solo confirma que `iai-mcp` está registrado como
  servidor MCP en `claude_desktop_config.json`, no que haya captura
  ambiental. Probado en vivo: una sesión larga en el Chat tab de Desktop
  dejó **cero registros nuevos** en el store. El Chat tab no tiene ningún
  mecanismo de hooks, estructuralmente.
- [Seguro] Cowork sí tiene un mecanismo de hooks real y separado
  (`iai-mcp cowork install/status`), pero `_looks_like_cowork_home()`
  busca `cowork_settings.json`, `cowork_plugins/` o `.claude.json`, y
  **ninguno de los tres existe** en esta versión de Desktop, confirmado
  tras un ciclo completo de cierre + reapertura + sesión Cowork nueva.
  `rpm/manifest.json` muestra que el mecanismo real de plugins en esta
  versión es centralizado (marketplace con ID emitido por servidor), no
  un archivo local.
- [Seguro] La función manual de Desktop "Agregar marketplace → Agregar
  desde un repositorio" falla con el mismo error tanto para el fork
  (`oscampo/iai-personal-memory-engine`) como para el repo oficial
  (`CodeAbra/iai-personal-memory-engine`, el que el propio README
  documenta como camino canónico). Como el repo oficial falla igual,
  **la causa no está en el código de este proyecto**, es un problema de
  la propia función de Desktop, fuera del alcance de un fix aquí.

## Conclusión

Dos issues tienen trabajo de código concreto y ofrecido, listo para
implementar. Uno quedó cerrado como no-accionable (documentado con
evidencia, no requiere más trabajo de este lado).

## Próximo paso concreto

1. **Issue `#113`** (parte de documentación, los bugs de código ya están
   resueltos): actualizar la tabla de `doctor` en el README para incluir
   `(ii)`, `(aa)`, `(bb)`; corregir la sección "Deployment surface" para
   aclarar que Windows usa TCP loopback + token, no Unix socket puro.
2. **Issue `#114`**: dividir la línea `Claude Desktop: ... WIRED` de
   `capture-hooks status` en dos, una para "MCP registrado" (Chat +
   Cowork) y otra para "hooks de captura ambiental activos" (solo
   Cowork, delegando a `cowork status`), y quitar "Desktop also wired"
   del resumen `status: ACTIVE` salvo que Cowork esté realmente activo.
3. **Issue `#119`**: sin trabajo de código. Ya documentado que la causa es
   externa al proyecto (función de Desktop). Dejar como referencia para
   quien llegue con el mismo síntoma.
4. Considerar armar el PR de los puntos 1 y 2 contra `CodeAbra/main`
   directamente (ambos issues ya ofrecieron mandar PR), no solo dejarlo en
   el fork.

## Destino Code

- **Repo:** `oscampo/iai-personal-memory-engine` (fork de
  `CodeAbra/iai-personal-memory-engine`), rama `main`, checkout local en
  `C:\Users\user\source\iai-personal-memory-engine`.
- **Issues relacionados:** `#113` (bugs de Windows resueltos + docs
  pendientes), `#114` (wording de `capture-hooks status`), `#119`
  (Cowork, cerrado como no-accionable, sin trabajo pendiente).
- **Archivos relevantes para el trabajo pendiente:**
  - `README.md` — sección `## Doctor` (tabla de checks) y
    `### Deployment surface`, ambas necesitan corrección de texto.
  - El módulo que arma el string de `capture-hooks status` (no
    identificado por ruta exacta en esta sesión, localizarlo primero;
    probablemente cerca de `_cowork.py` o `capture_hooks.py` bajo
    `src/iai_mcp/cli/`).
- **Resultado esperado de la siguiente sesión de Code:** los dos fixes de
  documentación/wording implementados y, si tiene sentido, un PR armado
  contra `CodeAbra/iai-personal-memory-engine` (no solo el fork), dado que
  ambos issues ya ofrecieron ese camino.
