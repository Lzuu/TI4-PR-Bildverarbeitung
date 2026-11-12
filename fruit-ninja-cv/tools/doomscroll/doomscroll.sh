#!/bin/bash
# macOS version of doomscroll.ps1: opens a short-video page in a NEW window of the
# default browser while an AI agent works, and closes exactly that window afterwards.
#
#   doomscroll.sh open    -> new window with YouTube Shorts / Instagram Reels / TikTok
#   doomscroll.sh close   -> closes only that window
#
# Chrome, Edge, Brave, Vivaldi, Opera, Firefox: started with their own profile
# directory, so only that window's processes are ended (your normal windows and tabs
# are never touched). Safari (default on most Macs) cannot be started with its own
# profile: its new window is opened and closed through AppleScript by window id. The
# first time, macOS asks once whether the terminal / Claude may control Safari
# (System Settings > Privacy & Security > Automation).

ROOT="$HOME/.claude/claude-scroll"
PROFILE="$ROOT/profile"          # no spaces: it is used as a pkill -f pattern
WINFILE="$ROOT/safari-window-id"

URLS=("https://www.youtube.com/shorts"
      "https://www.instagram.com/reels/"
      "https://www.tiktok.com/foryou")

# Bundle id of the https handler from `defaults read ...LSHandlers` (stdin), lower-case.
parse_default_browser() {
    awk '
        { line = $0; gsub(/^[ \t]+|[ \t]+$/, "", line) }
        line == "{" { role = "" }
        /LSHandlerRoleAll/ {
            if (match($0, /"[^"]+"/)) role = substr($0, RSTART + 1, RLENGTH - 2)
        }
        /LSHandlerURLScheme/ && /https;/ {
            if (role != "" && role != "-") { print tolower(role); exit }
        }
    '
}

# The page to open: Firefox always gets YouTube Shorts, everything else a random one.
pick_url() {
    if [ "$1" = "firefox" ]; then
        echo "${URLS[0]}"
    else
        echo "${URLS[RANDOM % ${#URLS[@]}]}"
    fi
}

# "<kind>|<app name>" for a bundle id; kind: chromium | firefox | safari
browser_info() {
    case "$1" in
        com.google.chrome)          echo "chromium|Google Chrome" ;;
        com.microsoft.edgemac)      echo "chromium|Microsoft Edge" ;;
        com.brave.browser)          echo "chromium|Brave Browser" ;;
        com.vivaldi.vivaldi)        echo "chromium|Vivaldi" ;;
        com.operasoftware.opera)    echo "chromium|Opera" ;;
        org.mozilla.firefox)        echo "firefox|Firefox" ;;
        *)                          echo "safari|Safari" ;;   # Safari and anything unknown
    esac
}

scroll_running() {
    pgrep -f -- "$PROFILE" >/dev/null 2>&1 && return 0
    if [ -s "$WINFILE" ] && pgrep -x Safari >/dev/null 2>&1; then
        local id
        id=$(cat "$WINFILE")
        [ "$(osascript -e "tell application \"Safari\" to count (every window whose id is $id)" 2>/dev/null)" = "1" ] && return 0
    fi
    return 1
}

do_open() {
    scroll_running && return 0                       # already open (another session)
    mkdir -p "$ROOT"
    local bundle info kind app url
    bundle=$(defaults read com.apple.LaunchServices/com.apple.launchservices.secure LSHandlers 2>/dev/null | parse_default_browser)
    info=$(browser_info "$bundle")
    kind="${info%%|*}"
    app="${info#*|}"
    url=$(pick_url "$kind")
    case "$kind" in
        chromium)
            open -n -a "$app" --args "--user-data-dir=$PROFILE" --new-window --no-first-run \
                 --no-default-browser-check "$url" ;;
        firefox)
            open -n -a "$app" --args -no-remote -new-instance -profile "$PROFILE" -new-window "$url" ;;
        *)
            osascript >"$WINFILE" 2>/dev/null <<EOF
tell application "Safari"
    make new document with properties {URL:"$url"}
    activate
    return id of front window
end tell
EOF
            ;;
    esac
}

do_close() {
    pkill -f -- "$PROFILE" >/dev/null 2>&1          # Chromium family and Firefox
    if [ -s "$WINFILE" ]; then
        if pgrep -x Safari >/dev/null 2>&1; then
            osascript -e "tell application \"Safari\" to close (every window whose id is $(cat "$WINFILE"))" >/dev/null 2>&1
        fi
        rm -f "$WINFILE"
    fi
}

main() {
    case "${1:-open}" in
        open)  do_open ;;
        close) do_close ;;
        *)     echo "usage: doomscroll.sh open|close" >&2; return 2 ;;
    esac
    return 0
}

# run only when executed (not when sourced, e.g. by tests)
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main "$@"
fi
