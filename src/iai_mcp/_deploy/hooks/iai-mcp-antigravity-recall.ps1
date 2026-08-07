$inputJson = $input | Out-String | ConvertFrom-Json -ErrorAction SilentlyContinue
if (-not $inputJson) { exit 0 }
$sessionId = $inputJson.conversationId
if (-not $sessionId) { $sessionId = $inputJson.conversation_id }
if (-not $sessionId) { $sessionId = $inputJson.session_id }
if (-not $sessionId) { exit 0 }
$cli = $null
$command = Get-Command iai-mcp -ErrorAction SilentlyContinue
if ($command) { $cli = $command.Source }
if (-not $cli) {
    $pipxPath = [System.IO.Path]::Combine($env:USERPROFILE, ".local", "pipx", "venvs", "iai-pme", "Scripts", "iai-mcp.exe")
    if (Test-Path $pipxPath) { $cli = $pipxPath }
}
if (-not $cli) { exit 0 }
& $cli session-start --session-id $sessionId

