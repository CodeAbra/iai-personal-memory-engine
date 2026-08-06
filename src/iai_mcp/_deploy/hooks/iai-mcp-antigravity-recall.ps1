$inputJson = $input | Out-String | ConvertFrom-Json -ErrorAction SilentlyContinue
if (-not $inputJson) { exit 0 }
$sessionId = $inputJson.conversationId
if (-not $sessionId) { $sessionId = $inputJson.conversation_id }
if (-not $sessionId) { $sessionId = $inputJson.session_id }
if (-not $sessionId) { exit 0 }
$cli = "C:\iai-personal-memory-engine\.venv\Scripts\iai-mcp.exe"
if (-not (Test-Path $cli)) { exit 0 }
& $cli session-start --session-id $sessionId
