#!/bin/bash
# Stops the automatic daily run installed by ./install_schedule.sh
LABEL="com.nbacrystalball.daily"
launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/$LABEL.plist"
echo "Automatic daily run removed."
