# Handoff → Code

## Origen

Conversación en Chat, sesión del 17 de agosto de 2026. Empezó revisando un
repo de terceros (`valera-studio-group/valera-studio-project`, un harness de
memoria paginada para Claude Code CLI) como posible alternativa mientras se
resolvía un problema real: iai-mcp no arrancaba en Windows.

## Contexto

Ayer (sesión previa, no en este hilo) se diagnosticó que el daemon de
iai-mcp crasheaba al bootear en Windows 11 con un traceback de
`AttributeError: module 'signal' has no attribute 'SIGHUP'`. Tras aplicar un
parche local, el daemon llegó más lejos pero se quedó atascado (CPU casi
plana, sin nuevo traceback, sin socket abierto) durante varios minutos, sin
llegar a confirmarse healthy. Esa sesión se detuvo ahí, sin escalar más a
ciegas.

En este hilo se comparó el fork (`oscampo/iai-personal-memory-engine`)
contra el proyecto original (`CodeAbra/iai-personal-memory-engine`) y se
encontraron y corrigieron dos bugs reales, ya subidos a `main` del fork.

## Hallazgos verificados

- [Seguro] El fork de Oscar no tenía ningún commit propio: estaba al día
  con `upstream/main` pero además 5 commits *atrás* (Release 3.0.2 y
  siguientes). El parche de `SIGHUP` de la sesión de ayer nunca llegó a
  commitearse, vivía solo (si acaso) en el working tree local de la máquina
  Windows.
- [Seguro] El bug de `SIGHUP` existe también en `upstream/main`, sin
  arreglar: no es un problema del fork ni del entorno de Oscar, es un bug
  real y reportable en `CodeAbra/iai-personal-memory-engine`.
- [Seguro] El bug real: en `src/iai_mcp/daemon/__init__.py`, la función
  `_install_boot_signal_trace` construía la tupla
  `(signal.SIGTERM, signal.SIGINT, signal.SIGHUP)` directamente en la
  cláusula `for`, evaluada antes de que el `try/except` interno pudiera
  protegerla. Eso crashea en Windows (sin `SIGHUP`) antes de que los
  handlers de shutdown graceful, unas líneas más abajo, lleguen a
  instalarse (esos sí ya usaban `getattr(signal, "SIGHUP", None)`).
- [Seguro] Segundo bug, idéntico en dos copias que deben mantenerse
  sincronizadas (`plugin/hooks/iai-mcp-turn-capture.sh` y
  `src/iai_mcp/_deploy/hooks/iai-mcp-turn-capture.sh`): el hook de captura
  por turno invocaba `/usr/bin/python3` con ruta fija (no existe en Git
  Bash de Windows) e importaba `fcntl` sin guard (módulo POSIX-only). Esto
  causaba `rc=127` en cada turno en Windows, es decir, la captura ambiental
  ("nunca digo 'recuérdalo'") estaba rota de fondo en esa plataforma.
- [Seguro] El transporte del daemon para Windows **sí existe** en el
  código: `src/iai_mcp/_ipc.py` implementa TCP loopback
  (`127.0.0.1:<puerto efímero>`) con autenticación por token vía `icacls`,
  agregado en el commit `6bfc5d9` ("Add Windows support..."), ya en `main`
  desde antes del Release 3.0.0. Esto contradice la lectura inicial de que
  "el soporte Windows no existía"; lo que se vio ayer (atascado sin socket)
  probablemente era el daemon quedándose en otro punto antes de llegar a
  bindear esa capa, no una ausencia de la capa misma.
- [Probable] El atascamiento visto ayer (CPU plana, sin traceback) podría
  desaparecer solo con el fix de `SIGHUP` si el daemon simplemente no había
  llegado tan lejos antes. No confirmado: no se ha vuelto a correr en la
  máquina Windows real desde que se subieron los parches.

## Commits aplicados (ya en `main` de `oscampo/iai-personal-memory-engine`)

1. `7e576682` — guard de `SIGHUP` en `_install_boot_signal_trace`
   (`src/iai_mcp/daemon/__init__.py`).
2. `c94e996f` — resolución de intérprete Python + guard de `fcntl` en
   `plugin/hooks/iai-mcp-turn-capture.sh`.
3. `b31a489b` — mismo fix, copia sincronizada en
   `src/iai_mcp/_deploy/hooks/iai-mcp-turn-capture.sh`.

## Conclusión

El camino correcto sigue siendo terminar el port a Windows sobre el fork
propio, no adoptar una herramienta externa (Valera Studio Harness, un
wrapper de Claude Code CLI sin componente MCP, evaluado y descartado en
este mismo hilo por no cubrir el caso de uso donde el MCP de iai-mcp ya
funciona). Los dos bugs identificados con traceback en mano ya están
corregidos y subidos. Queda por confirmar si eso basta para un boot limpio.

## Próximo paso concreto

En la máquina Windows: `git pull` sobre el fork, reinstalar
(`pip install .` desde el checkout, o reemplazar el paquete en el venv ya
activo), y repetir `iai-mcp daemon install` / arranque, con el mismo
protocolo de diagnóstico de ayer (`Get-CimInstance` para confirmar
proceso, revisar `task-stderr.log` y el log del daemon). Si vuelve a
atascarse en el mismo punto (sin traceback, sin socket, CPU plana), el
siguiente paso es leer `_ipc.py` y el arranque del `SocketServer` con el
traceback real en mano, en vez de suponer.

## Destino Code

- **Repo:** `oscampo/iai-personal-memory-engine` (fork de
  `CodeAbra/iai-personal-memory-engine`), rama `main`.
- **Módulos relevantes al problema de Windows:**
  - `src/iai_mcp/daemon/__init__.py` — boot del daemon, FSM, señales
    (ya parcheado en el punto de `SIGHUP`).
  - `src/iai_mcp/_ipc.py` — capa de transporte multiplataforma (Unix
    socket en POSIX, TCP loopback + token en Windows). Punto de partida
    si el atascamiento persiste.
  - `plugin/hooks/iai-mcp-turn-capture.sh` y
    `src/iai_mcp/_deploy/hooks/iai-mcp-turn-capture.sh` — hooks de
    captura ambiental (ya parcheados, mantener sincronizados a mano si se
    tocan de nuevo).
  - `src/iai_mcp/_flock.py` — el shim de locking POSIX/Windows ya
    existente en el paquete, usado como referencia para el fix del hook
    (el hook no pudo importarlo directamente porque corre bajo el Python
    del sistema, no el venv).
- **Resultado esperado de la siguiente sesión de Code:** correr el daemon
  en Windows real, capturar el traceback o confirmar arranque limpio, y si
  hace falta, localizar y corregir el siguiente punto de fallo en
  `_ipc.py` o en el arranque de `SocketServer`.
