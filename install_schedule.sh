#!/bin/bash
# Makes this Mac run daily.py automatically (11:00, with catch-up tries at 14:00,
# 17:00 and at login). Extra tries do nothing once the day's run is finished.
# Undo with ./uninstall_schedule.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
LABEL="com.nbacrystalball.daily"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
SYSLOG_DIR="$HOME/Library/Logs/NBACrystalBall"
PY="$DIR/.venv/bin/python"

[ -x "$PY" ] || { echo "Missing $PY - create the virtual environment first."; exit 1; }
mkdir -p "$HOME/Library/LaunchAgents" "$SYSLOG_DIR" "$DIR/logs"
touch "$DIR/logs/daily.log"
BEFORE=$(wc -l < "$DIR/logs/daily.log")

cat > "$PLIST" <<PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array><string>$PY</string><string>$DIR/daily.py</string></array>
  <key>WorkingDirectory</key><string>$DIR</string>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Hour</key><integer>11</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>14</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Hour</key><integer>17</integer><key>Minute</key><integer>0</integer></dict>
  </array>
  <key>RunAtLoad</key><true/>
  <key>StandardOutPath</key><string>$SYSLOG_DIR/launchd.log</string>
  <key>StandardErrorPath</key><string>$SYSLOG_DIR/launchd.log</string>
</dict>
</plist>
PLIST_EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Installed. A first run has started; waiting for it to finish..."

for _ in $(seq 1 60); do
  sleep 3
  if tail -n +"$((BEFORE + 1))" "$DIR/logs/daily.log" | grep -q "=== Run finished\|=== Run failed\|Already finished"; then
    tail -n +"$((BEFORE + 1))" "$DIR/logs/daily.log" | tail -n 12
    echo
    echo "The schedule is working. Log: $DIR/logs/daily.log"
    exit 0
  fi
done

echo "The scheduled run did not report back. Output from macOS:"
tail -n 15 "$SYSLOG_DIR/launchd.log" 2>/dev/null || echo "(nothing logged)"
echo
echo "If it says 'Operation not permitted', macOS is blocking background access to"
echo "your Documents folder. Open System Settings > Privacy & Security > Full Disk"
echo "Access, add this program, then run this script again:"
echo "  $(readlink -f "$PY" 2>/dev/null || echo "$PY")"
exit 1
