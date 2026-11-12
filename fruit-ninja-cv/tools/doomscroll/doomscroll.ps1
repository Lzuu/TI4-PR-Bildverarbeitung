# Opens a short-video page in a NEW window of the default browser while an AI agent
# (Claude Code, Codex, ...) works, and closes exactly that window when it is done.
#
#   doomscroll.ps1 open    -> new window with YouTube Shorts / Instagram Reels / TikTok
#   doomscroll.ps1 close   -> closes only that window
#
# The window runs in its own browser profile (%LOCALAPPDATA%\ClaudeScroll\profile), so
# the normal browser and its tabs are never touched. Close = end every process whose
# command line contains that profile path.
param([Parameter(Position = 0)][ValidateSet('open', 'close')][string]$Action = 'open')

$ErrorActionPreference = 'SilentlyContinue'
$root = Join-Path $env:LOCALAPPDATA 'ClaudeScroll'
$profileDir = Join-Path $root 'profile'

function Get-ScrollProcesses {
    # all browser processes started with our profile directory
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -and $_.CommandLine.Contains($profileDir) -and $_.ProcessId -ne $PID }
}

if ($Action -eq 'close') {
    foreach ($p in Get-ScrollProcesses) {
        & taskkill.exe /PID $p.ProcessId /T /F *> $null      # incl. child processes
    }
    exit 0
}

# --- open ---------------------------------------------------------------------------
if (Get-ScrollProcesses) { exit 0 }                           # already open (another session)

$urls = @('https://www.youtube.com/shorts',
          'https://www.instagram.com/reels/',
          'https://www.tiktok.com/foryou')

# default browser from the registry
$progId = (Get-ItemProperty 'HKCU:\Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice').ProgId
$cmd = (Get-ItemProperty "Registry::HKEY_CLASSES_ROOT\$progId\shell\open\command").'(default)'
$exe = $null
if ($cmd -match '^\s*"([^"]+)"') { $exe = $Matches[1] }
elseif ($cmd -match '^\s*(\S+)') { $exe = $Matches[1] }
if (-not $exe -or -not (Test-Path $exe)) {
    $exe = "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe"   # fallback
}

New-Item -ItemType Directory -Force -Path $profileDir | Out-Null
$name = [IO.Path]::GetFileNameWithoutExtension($exe).ToLower()

# Firefox always gets YouTube Shorts, every other browser a random page
if ($name -match 'firefox') { $url = 'https://www.youtube.com/shorts' }
else { $url = $urls | Get-Random }

if ($name -match 'firefox') {
    $browserArgs = @('-no-remote', '-new-instance', '-profile', "`"$profileDir`"", '-new-window', $url)
} else {                                                       # Chrome, Edge, Brave, Vivaldi, Opera
    $browserArgs = @("--user-data-dir=`"$profileDir`"", '--new-window', '--no-first-run',
                     '--no-default-browser-check', $url)
}
Start-Process -FilePath $exe -ArgumentList $browserArgs
exit 0
