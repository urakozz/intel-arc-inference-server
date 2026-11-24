#!/bin/bash
# Run a command detached from the terminal / ssh session (a dropped connection does not stop it):
#   tools/probe/detach.sh <log> <cmd> [args...]
# from the repo root (e.g. tools/probe/detach.sh $B70_DATA/logs/<topic>/x.log ...).
# Logs go under $B70_DATA/logs/<topic>/, never loose in $HOME; the log's directory is
# made here. The log ends in "ALLDONE rc=<status>". Poll it.
log="$1"
shift
mkdir -p "$(dirname "$log")"
cd "$(dirname "$0")/../.."
setsid nohup bash -c '"$@"; echo "ALLDONE rc=$?"' detach "$@" > "$log" 2>&1 < /dev/null &
echo "started, log $log"
