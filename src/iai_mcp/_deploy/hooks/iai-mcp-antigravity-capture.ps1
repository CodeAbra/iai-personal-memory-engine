$inputJson = $input | Out-String | ConvertFrom-Json -ErrorAction SilentlyContinue
if (-not $inputJson) { exit 0 }

$sessionId = $inputJson.conversationId
if (-not $sessionId) { $sessionId = $inputJson.conversation_id }
if (-not $sessionId) { $sessionId = $inputJson.session_id }

$transcriptPath = $inputJson.transcriptPath
if (-not $transcriptPath) { $transcriptPath = $inputJson.transcript_path }

if ($transcriptPath -and $transcriptPath.EndsWith("transcript.jsonl")) {
    $fullPath = $transcriptPath.Substring(0, $transcriptPath.Length - "transcript.jsonl".Length) + "transcript_full.jsonl"
    if (Test-Path $fullPath) {
        $transcriptPath = $fullPath
    }
}

if (-not $transcriptPath -or -not (Test-Path $transcriptPath)) {
    exit 0
}

$cli = $null
$command = Get-Command iai-mcp -ErrorAction SilentlyContinue
if ($command) { $cli = $command.Source }
if (-not $cli) {
    $pipxPath = [System.IO.Path]::Combine($env:USERPROFILE, ".local", "pipx", "venvs", "iai-pme", "Scripts", "iai-mcp.exe")
    if (Test-Path $pipxPath) { $cli = $pipxPath }
}
if (-not $cli) { exit 0 }

& $cli capture-turn-deferred --session-id $sessionId --transcript-path $transcriptPath --max-turns-per-call 1000

$deferredDir = [System.IO.Path]::Combine($env:USERPROFILE, ".iai-mcp", ".deferred-captures")
$liveFile = [System.IO.Path]::Combine($deferredDir, "$sessionId.live.jsonl")
if (Test-Path $liveFile) {
    $ts = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $pidId = $PID
    $newName = "$sessionId.live-$ts-$pidId.jsonl"
    Rename-Item -Path $liveFile -NewName $newName -ErrorAction SilentlyContinue
}

