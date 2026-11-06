#!/bin/bash
# Run a command ON THE BOX detached from the ssh session (the WiFi drops long ones):
#   tools/probe/detach.sh <log> <cmd> [args...]
# from the repo root on the box (tools/box.sh run 'tools/probe/detach.sh ~/b70/logs/<topic>/x.log ...').
# Logs go under ~/b70/logs/<topic>/, never loose in the box's $HOME; the log's directory is
# made here. The log ends in "ALLDONE rc=<status>". Poll it.
log="$1"
shift
mkdir -p "$(dirname "$log")"
cd "$(dirname "$0")/../.."
setsid nohup bash -c '"$@"; echo "ALLDONE rc=$?"' detach "$@" > "$log" 2>&1 < /dev/null &
echo "started, log $log"
