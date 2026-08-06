$ErrorActionPreference = "SilentlyContinue"
$brainDir = [System.IO.Path]::Combine($env:USERPROFILE, ".gemini", "antigravity", "brain")
$captureScript = [System.IO.Path]::Combine($env:USERPROFILE, ".gemini", "config", "iai-hooks", "antigravity-capture.ps1")
$stateDir = [System.IO.Path]::Combine($env:USERPROFILE, ".iai-mcp", ".capture-state")

if (-not (Test-Path $brainDir) -or -not (Test-Path $captureScript)) { exit 0 }

Get-ChildItem -Path $brainDir | Where-Object { $_.PSIsContainer -and $_.Name -match "^[0-9a-f\-]{36}$" } | ForEach-Object {
    $cid = $_.Name
    $tpath = [System.IO.Path]::Combine($_.FullName, ".system_generated", "logs", "transcript.jsonl")
    $fullPath = [System.IO.Path]::Combine($_.FullName, ".system_generated", "logs", "transcript_full.jsonl")
    
    $targetFile = $null
    if (Test-Path $fullPath) { $targetFile = Get-Item $fullPath }
    elseif (Test-Path $tpath) { $targetFile = Get-Item $tpath }
    
    if ($targetFile) {
        $offsetFile = [System.IO.Path]::Combine($stateDir, "$cid.offset")
        $shouldProcess = $true
        
        if (Test-Path $offsetFile) {
            $offsetItem = Get-Item $offsetFile
            if ($targetFile.LastWriteTime -le $offsetItem.LastWriteTime) {
                $shouldProcess = $false
            }
        }
        
        if ($shouldProcess) {
            $json = @{ conversationId = $cid; transcriptPath = $targetFile.FullName } | ConvertTo-Json -Compress
            $json | powershell.exe -ExecutionPolicy Bypass -NoProfile -File $captureScript -ErrorAction SilentlyContinue
        }
    }
}
