# Handoff → Cowork

## Origen

Conversación en Chat, sesión del 17 de agosto de 2026 (continuación de la
sesión que ya resolvió el boot de Windows, ver commits `7e576682`,
`c94e996f`, `b31a489b`, `1107640`, `23727ff`, todos en `main`, todos
verificados end-to-end en la máquina real). Ese problema quedó cerrado.
Este handoff es sobre uno nuevo, distinto: activar captura ambiental real
en Claude Desktop.

## Contexto

Se diseñó (todavía a nivel de conversación, sin implementar) un sistema
para reemplazar el rol de "memoria de largo plazo portable" de iai-pme por
un vault de Obsidian sincronizado por git + LightRAG, dejando iai-pme
únicamente como motor de recall semántico en vivo dentro de una sesión.
Ese rediseño no es el foco de este handoff, se retoma después.

Durante la verificación se descubrió que `capture-hooks status` reporta
`Claude Desktop: ... WIRED` de forma engañosa: eso solo confirma que
`iai-mcp` está registrado como servidor MCP (`claude_desktop_config.json`),
no que exista captura ambiental automática. Se probó en vivo: dos horas de
conversación real en el Chat tab, cero registros nuevos en el store
después. El Chat tab no tiene ningún mecanismo de hooks, estructuralmente,
a diferencia de Claude Code.

Se encontró que **Cowork sí tiene un mecanismo de hooks real**, vía el
subcomando `iai-mcp cowork install/status`, que engancha
`recall_hooks`/`capture_hooks` en el `cowork_settings.json` de cada
"Cowork home". Se corrió `iai-mcp cowork install` en la máquina de casa:
quedó el plugin "staged" en `~/.iai-mcp/claude-plugin`, pero no se pudo
conectar porque **nunca se había abierto una sesión de Cowork en esta
máquina** (`no Cowork homes found`).

## Hallazgos verificados

- [Seguro] `claude_desktop_config.json` solo contiene `mcpServers`
  (`github`, `iai-mcp`), sin ninguna sección de hooks. Confirmado leyendo
  el archivo completo.
- [Seguro] `episodes_recent` tras una sesión larga de Chat tab: 0 registros
  nuevos. Solo los de pruebas manuales anteriores seguían ahí.
- [Seguro] `iai-mcp cowork status` inicial: `status: INACTIVE`, `Cowork
  homes: none found (desktop app absent or Cowork unused)`.
- [Seguro] `iai-mcp cowork install` corrido: materializó el plugin
  correctamente, pero no encontró ningún "Cowork home" donde engancharlo.
  El mensaje del propio comando indica el paso que falta: *"rerun install
  after the first Cowork session."*
- Reportado en GitHub, dos issues abiertos hoy en
  `CodeAbra/iai-personal-memory-engine`:
  - `#113` — los tres bugs de Windows (ya resueltos en este fork) más dos
    discrepancias de documentación (claim de Unix-socket-only falso en
    Windows, tabla de `doctor` desactualizada: 27 documentados vs 30
    reales).
  - `#114` — específico de este hallazgo: `capture-hooks status` conflating
    "MCP registrado en Desktop" con "hooks activos", cuando en realidad
    solo Cowork tiene hooks reales y estaban INACTIVE.

## Conclusión

No hay arreglo de configuración para captura ambiental en el Chat tab, es
un límite estructural del producto. Cowork sí tiene el mecanismo correcto,
solo falta activarlo con una sesión real.

## Próximo paso concreto

1. Confirmar que esta sesión de Cowork cuenta como "abrir Cowork por primera
   vez" en esta máquina (o que ya existe un Cowork home previo).
2. Correr `iai-mcp cowork install` de nuevo, ahora que el home debería
   existir.
3. Verificar con `iai-mcp cowork status` que pase de `INACTIVE` a `ACTIVE`.
4. Prueba real, no solo status: después de un par de turnos en esta misma
   sesión de Cowork, consultar `episodes_recent` (vía las tools MCP, o
   `iai last` por CLI) y confirmar que aparece contenido nuevo de esta
   conversación, sin haber llamado `memory_capture` a mano. Eso es lo único
   que prueba que el hook realmente corrió y no solo que el plugin quedó
   instalado.
5. Si algo falla, capturar el error real (no asumir), de la misma forma
   que se hizo esta semana con los bugs de Windows: traceback completo,
   `iai-mcp doctor`, y el estado exacto de `cowork_settings.json` del home
   correspondiente antes de tocar nada.

## Destino Cowork

- **Objetivo de esta sesión:** activar y verificar hooks de captura
  ambiental reales en Cowork (no solo confirmar que las tools MCP están
  disponibles, eso ya se sabe que funciona).
- **Comando clave:** `iai-mcp cowork install` /
  `iai-mcp cowork status` (venv: `C:\Users\user\.iai-pme-venv\Scripts\`,
  ya está en el `PATH` del usuario, no hace falta ruta completa en una
  terminal nueva).
- **Archivos relevantes:** el `cowork_settings.json` de cada Cowork home
  (ruta exacta todavía no confirmada, la reporta el propio comando
  `install` una vez que encuentra el home), y
  `C:\Users\user\.iai-mcp\claude-plugin` (marketplace ya materializado).
- **Resultado esperado:** `cowork status` → `ACTIVE`, y un `episodes_recent`
  después de un par de turnos de esta sesión mostrando contenido capturado
  solo, sin intervención manual.
