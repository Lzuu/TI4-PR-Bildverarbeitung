# Short-Video-Fenster, solange ein Agent arbeitet

Sobald du bei Claude Code eine Nachricht abschickst, öffnet sich ein neues Fenster
des Standardbrowsers mit YouTube Shorts, Instagram Reels oder TikTok (zufällig).
**Ist Firefox der Standardbrowser, öffnet sich immer YouTube Shorts.**
Ist der Agent fertig, schließt sich genau dieses Fenster wieder. Das gehört nicht zum
Spiel, jede Person richtet es auf ihrem eigenen Rechner ein (Issue #48).

| Datei | System |
|---|---|
| `doomscroll.ps1` | Windows (PowerShell) |
| `doomscroll.sh` | macOS (bash) |

Beide kennen zwei Aktionen: `open` und `close`.

## Wie das Fenster gezielt geschlossen wird
- **Windows, Chrome/Edge/Brave/Firefox, macOS Chrome/Edge/Brave/Vivaldi/Opera/Firefox:**
  Das Fenster läuft in einem eigenen Browserprofil (`%LOCALAPPDATA%\ClaudeScroll\profile`
  bzw. `~/.claude/claude-scroll/profile`). `close` beendet nur Prozesse mit diesem
  Profilpfad. Dein normaler Browser mit seinen Tabs bleibt unberührt.
- **macOS, Safari (Standard):** Safari kann kein eigenes Profil starten. Das Fenster
  wird per AppleScript geöffnet und über seine Fenster-ID wieder geschlossen. Beim
  ersten Mal fragt macOS, ob das Terminal bzw. Claude Safari steuern darf
  (Systemeinstellungen > Datenschutz & Sicherheit > Automation).

## Einrichten
Die Skripte aus diesem Ordner liegen schon im Repo. Du trägst nur die Hooks in deine
**Benutzer-Konfiguration** ein (`~/.claude/settings.json`, gilt für alle Projekte).
Danach `/hooks` öffnen oder Claude neu starten. Pfad zum Skript = dein Klon dieses Repos.

**Windows** (Pfad anpassen, in JSON mit doppelten Backslashes):
```json
"hooks": {
  "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "powershell.exe",
    "args": ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "C:\PFAD\fruit-ninja-cv\tools\doomscroll\doomscroll.ps1", "open"], "timeout": 15}]}],
  "Stop":        [{"hooks": [{"type": "command", "command": "powershell.exe",
    "args": ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "C:\PFAD\fruit-ninja-cv\tools\doomscroll\doomscroll.ps1", "close"], "timeout": 15}]}],
  "StopFailure": [{"hooks": [{"type": "command", "command": "powershell.exe",
    "args": ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "C:\PFAD\fruit-ninja-cv\tools\doomscroll\doomscroll.ps1", "close"], "timeout": 15}]}],
  "SessionEnd":  [{"hooks": [{"type": "command", "command": "powershell.exe",
    "args": ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "C:\PFAD\fruit-ninja-cv\tools\doomscroll\doomscroll.ps1", "close"], "timeout": 15}]}]
}
```

**macOS** (Pfad anpassen):
```json
"hooks": {
  "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "bash \"$HOME/PFAD/fruit-ninja-cv/tools/doomscroll/doomscroll.sh\" open", "timeout": 15}]}],
  "Stop":        [{"hooks": [{"type": "command", "command": "bash \"$HOME/PFAD/fruit-ninja-cv/tools/doomscroll/doomscroll.sh\" close", "timeout": 15}]}],
  "StopFailure": [{"hooks": [{"type": "command", "command": "bash \"$HOME/PFAD/fruit-ninja-cv/tools/doomscroll/doomscroll.sh\" close", "timeout": 15}]}],
  "SessionEnd":  [{"hooks": [{"type": "command", "command": "bash \"$HOME/PFAD/fruit-ninja-cv/tools/doomscroll/doomscroll.sh\" close", "timeout": 15}]}]
}
```

Von Hand testen: `... open`, dann `... close`.

## Grenzen
- Mehrere Claude-Sitzungen gleichzeitig teilen sich ein Fenster; es schließt, sobald
  eine davon fertig ist.
- Fragt der Agent nach einer Berechtigung, bleibt das Fenster offen.
- **Codex:** nicht eingerichtet. Die Skripte sind unabhängig von Claude; `open` und
  `close` lassen sich an Codex' Start- und Ende-Hooks hängen.
- **Die macOS-Version ist noch nicht auf einem echten Mac geprüft.** Getestet sind
  die Erkennung des Standardbrowsers, die Zuordnung und die Fehlerfälle
  (`tests/test_doomscroll.py`); ob Öffnen und Schließen funktionieren, zeigt erst ein
  Test am Mac.
- **Entfernen:** den `hooks`-Block aus `settings.json` löschen.
