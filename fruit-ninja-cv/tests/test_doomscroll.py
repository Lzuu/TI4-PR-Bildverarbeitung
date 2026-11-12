"""Short-video window while an agent works (#48) -- headless, no browser is started."""

import shutil
import subprocess
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parent.parent / "tools" / "doomscroll"
SH = TOOLS / "doomscroll.sh"
PS1 = TOOLS / "doomscroll.ps1"


def _defaults(https_handler=None, scheme="https"):
    """Text like `defaults read com.apple.LaunchServices/... LSHandlers`."""
    role = f'        LSHandlerRoleAll = "{https_handler}";\n' if https_handler else ""
    return ('(\n        {\n        LSHandlerPreferredVersions =         {\n'
            '            LSHandlerRoleAll = "-";\n        };\n'
            f'{role}        LSHandlerURLScheme = {scheme};\n    }}\n)')


def _bash(script):
    """Run ``script`` in bash inside the tools folder; skip if no usable bash exists."""
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("no bash available")
    probe = subprocess.run([bash, "-c", "source ./doomscroll.sh && echo ok"], cwd=TOOLS,
                           capture_output=True, text=True)
    if probe.stdout.strip() != "ok":
        pytest.skip("bash cannot run the script here (e.g. WSL without a distribution)")
    return lambda text="": subprocess.run([bash, "-c", f"source ./doomscroll.sh; {script}"],
                                          cwd=TOOLS, input=text, capture_output=True,
                                          text=True).stdout.strip()


@pytest.mark.parametrize("bundle", ["com.google.chrome", "org.mozilla.firefox",
                                    "com.microsoft.edgemac"])
def test_default_browser_is_read_from_the_https_handler(bundle):
    assert _bash("parse_default_browser")(_defaults(bundle)) == bundle


def test_http_only_entry_is_not_the_https_browser():
    assert _bash("parse_default_browser")(_defaults("com.google.chrome", "http")) == ""


@pytest.mark.parametrize("text", ["", _defaults(None)])
def test_no_https_handler_means_no_browser_found(text):
    assert _bash("parse_default_browser")(text) == ""


@pytest.mark.parametrize("bundle,expected", [
    ("com.google.chrome", "chromium|Google Chrome"),
    ("com.microsoft.edgemac", "chromium|Microsoft Edge"),
    ("com.brave.browser", "chromium|Brave Browser"),
    ("org.mozilla.firefox", "firefox|Firefox"),
    ("com.apple.safari", "safari|Safari"),
    ("", "safari|Safari"),                      # nothing set: macOS default
    ("company.thebrowser.browser", "safari|Safari"),   # unknown: Safari path
])
def test_browser_kind_and_app_name(bundle, expected):
    assert _bash(f'browser_info "{bundle}"')() == expected


def test_unknown_action_is_rejected():
    run = _bash('main foo 2>/dev/null; echo "exit $?"')
    assert run().endswith("exit 2")


def test_shell_script_has_unix_line_endings():
    # a CRLF script cannot run on macOS ("bad interpreter"); .gitattributes enforces LF
    assert b"\r" not in SH.read_bytes()
    assert SH.read_bytes().startswith(b"#!/bin/bash\n")


def test_windows_script_closes_only_its_own_profile():
    text = PS1.read_text(encoding="utf-8")
    assert "ClaudeScroll" in text and "Contains($profileDir)" in text
    assert "Stop-Process -Name" not in text and "taskkill.exe /IM" not in text
