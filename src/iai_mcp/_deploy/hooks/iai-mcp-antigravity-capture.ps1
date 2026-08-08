$inputJson = $input | Out-String | ConvertFrom-Json -ErrorAction SilentlyContinue
if (-not $inputJson) { exit 0 }

$sessionId = $inputJson.conversationId
if (-not $sessionId) { $sessionId = $inputJson.conversation_id }
if (-not $sessionId) { $sessionId = $inputJson.session_id }
if (-not $sessionId) { exit 0 }

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
# Host-controlled text lands on a joined command line below; a value
# carrying a double-quote could splice extra arguments. Refuse it.
if ($sessionId -match '"' -or $transcriptPath -match '"') {
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

# Bounded like the .sh path's `timeout 30`: a hung CLI must not hang the
# host's Stop hook. Start-Process joins -ArgumentList WITHOUT quoting, so
# any value that may contain spaces must carry its own quotes.
$proc = Start-Process -FilePath $cli -ArgumentList @(
    "capture-turn-deferred",
    "--session-id", "`"$sessionId`"",
    "--transcript-path", "`"$transcriptPath`"",
    "--max-turns-per-call", "1000"
) -NoNewWindow -PassThru
if (-not $proc.WaitForExit(30000)) {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
}

$deferredDir = [System.IO.Path]::Combine($env:USERPROFILE, ".iai-mcp", ".deferred-captures")
$liveFile = [System.IO.Path]::Combine($deferredDir, "$sessionId.live.jsonl")
if (Test-Path $liveFile) {
    $ts = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
    $pidId = $PID
    $newName = "$sessionId.live-$ts-$pidId.jsonl"
    Rename-Item -Path $liveFile -NewName $newName -ErrorAction SilentlyContinue
}

