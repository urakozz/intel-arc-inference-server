#!/bin/bash
# tools/probe/offload_links.sh - spec 22 P0.1 (plan 22a Task 2 Step 4): the PCIe links both B70s train at, every
# bridge above each up to its root port, the CPU and the host memory channels.
#
#   offload_links.sh [--under-load S] [--channels N]          on the box (lspci, /sys, lscpu, sudo -n dmidecode)
#   offload_links.sh --lspci-file F [--dmidecode-file F]      parse saved text instead (the fixture: tools/probe/testdata)
#   offload_links.sh --self-test                              the parser over tools/probe/testdata/lspci_b70.txt
#
# For 0000:04:00.0 and 0000:08:00.0: the chain of bridges from the root port down (/sys/bus/pci/devices/<bdf>'s
# resolved path), each one's `lspci -vv -s <bdf>` LnkCap / LnkSta as generation and width, idle and - with
# --under-load S - while `probe_offload hold --dev both --seconds S` copies (sampled at S / 2; $PROBE_OFFLOAD or
# build/tools/probe/probe_offload; run it under the GPU lock). The card's link is the one between its root port and
# the first device below it (the card's on-board switch upstream port): the endpoint behind the switch reports an
# internal link, printed but not used. Then `lscpu`, `sudo -n dmidecode -t memory` (the command to run by hand when
# sudo -n is refused), the derived host RAM ceiling = channels x MT/s x 8 B (channels: --channels, else the populated
# DIMMs capped at 4 - the T5810's Xeon E5 v3 / v4 has 4 channels, ASSUMED unless given).
#
# Prints `link <dev> bdf=.. role=root|bridge|endpoint cap=Gen<g> x<w> sta=Gen<g> x<w> state=idle|load`, one line
# `links: dev0 Gen<g> x<w>, dev1 Gen<g> x<w>` (per state) and `STOP: <card> trains below Gen3 x8` when one does
# (spec 22 §3: stop and report). Exit 0 (an unreadable link - lspci -vv hides LnkSta without root - prints an
# `UNKNOWN:` line for the record and the hand command), 3 on a STOP line, 2 on a usage error.
set -u
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
cards="0000:04:00.0 0000:08:00.0"
lspci_file="" dmi_file="" load=0 channels="" self=0
while [ $# -gt 0 ]; do
  case "$1" in
    --under-load) load=${2:?--under-load needs seconds}; shift 2 ;;
    --channels) channels=${2:?--channels needs a count}; shift 2 ;;
    --lspci-file) lspci_file=${2:?}; shift 2 ;;
    --dmidecode-file) dmi_file=${2:?}; shift 2 ;;
    --self-test) self=1; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "offload_links: unknown argument $1" >&2; exit 2 ;;
  esac
done

# GT/s -> PCIe generation
gen_of() {
  case "$1" in
    2.5) echo 1 ;; 5) echo 2 ;; 8) echo 3 ;; 16) echo 4 ;; 32) echo 5 ;; 64) echo 6 ;; *) echo 0 ;;
  esac
}
short() { echo "${1#0000:}"; }

# The bridge chain above a card, root port first, the card last.
chain() {
  if [ -n "$lspci_file" ]; then
    grep "^# chain $1 " "$lspci_file" | head -1 | cut -d' ' -f4-
  else
    readlink -f "/sys/bus/pci/devices/$1" 2>/dev/null | tr '/' '\n' |
      grep -E '^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9a-f]$' | tr '\n' ' '
  fi
}
# One device's lspci -vv text.
lspci_of() {
  if [ -n "$lspci_file" ]; then
    awk -v b="$(short "$1")" '/^[^#\t ]/ { on = (index($0, b " ") == 1) } on' "$lspci_file"
  else   # the capabilities (LnkCap / LnkSta) need root: sudo -n first, then plain lspci
    sudo -n lspci -vv -s "$1" 2>/dev/null || lspci -vv -s "$1" 2>/dev/null
  fi
}
# "<gen> <width>" of a LnkCap / LnkSta line ("Speed 8GT/s (downgraded), Width x16 (downgraded)").
parse_link() {
  sed -n "s/.*$2:[[:space:]]*\(.*\)/\1/p" <<< "$1" | head -1 |
    sed -n 's/.*Speed \([0-9.]*\)GT\/s[^,]*, Width x\([0-9]*\).*/\1 \2/p'
}

# links <state>: every card's chain and its link; sets $summary, returns 3 on a STOP.
links() {
  local state=$1 rc=0 summary="links:" i=0
  for card in $cards; do
    local ch role n=0 g=0 w=0 tag="dev$i"
    ch=$(chain "$card")
    if [ -z "$ch" ]; then
      echo "link $tag bdf=$card MISSING (no such device or sysfs path)"
      summary="$summary $tag missing,"
      rc=3
      i=$((i + 1))
      continue
    fi
    local nlinks
    nlinks=$(echo "$ch" | wc -w | tr -d ' ')
    for b in $ch; do
      n=$((n + 1))
      if [ "$n" = 1 ]; then role=root; elif [ "$n" = "$nlinks" ]; then role=endpoint; else role=bridge; fi
      local txt cap sta capg capw stag staw
      txt=$(lspci_of "$b")
      cap=$(parse_link "$txt" LnkCap) sta=$(parse_link "$txt" LnkSta)
      capg=$(gen_of "${cap% *}") capw=${cap#* } stag=$(gen_of "${sta% *}") staw=${sta#* }
      if [ -z "$sta" ]; then
        echo "link $tag bdf=$b role=$role no LnkSta (lspci -vv needs root for the capabilities: run it with sudo)"
        continue
      fi
      echo "link $tag bdf=$b role=$role cap=Gen$capg x$capw sta=Gen$stag x$staw state=$state"
      # the card's link: the root port's LnkSta and the device below it (both ends; the lower is what trains)
      if [ "$n" -le 2 ] && [ "$role" != endpoint ]; then
        if [ "$g" = 0 ] || [ "$stag" -lt "$g" ]; then g=$stag; fi
        if [ "$w" = 0 ] || [ "$staw" -lt "$w" ]; then w=$staw; fi
      fi
    done
    summary="$summary $tag Gen$g x$w,"
    if [ "$g" = 0 ] || [ "$w" = 0 ]; then
      # not a STOP and not a pass of the rule: the record names it; run `sudo lspci -vv -s <bdf>` by hand
      echo "UNKNOWN: $tag ($card) - its link was not readable (lspci -vv without root hides LnkSta: sudo -n refused)"
    elif [ "$g" -lt 3 ] || [ "$w" -lt 8 ]; then
      echo "STOP: $tag ($card) trains below Gen3 x8 (Gen$g x$w, state $state) - spec 22 §3: stop and report"
      rc=3
    fi
    i=$((i + 1))
  done
  echo "${summary%,}"
  return $rc
}

memory() {
  local txt
  if [ -n "$dmi_file" ]; then
    txt=$(cat "$dmi_file")
  elif [ -n "$lspci_file" ]; then
    return 0
  elif ! txt=$(sudo -n dmidecode -t memory 2>/dev/null); then
    echo "dmidecode: sudo -n refused - run by hand and add to the record: sudo dmidecode -t memory"
    return 0
  fi
  local dimms mts n
  dimms=$(awk '/^Memory Device/ { on = 1 } on && /^[[:space:]]+Size:/ { if ($0 !~ /No Module|Not Installed|Unknown/) n++; on = 0 } END { print n + 0 }' <<< "$txt")
  mts=$(sed -nE 's/^[[:space:]]*Configured (Memory|Clock) Speed: ([0-9]+) MT\/s.*/\2/p' <<< "$txt" | sort -n | head -1)
  [ -z "$mts" ] && mts=$(sed -nE 's/^[[:space:]]*Speed: ([0-9]+) MT\/s.*/\1/p' <<< "$txt" | sort -n | head -1)
  n=${channels:-$((dimms < 4 ? dimms : 4))}
  local src="--channels"
  [ -z "$channels" ] && src="ASSUMED: populated DIMMs capped at the E5 v3 / v4's 4"
  echo "memory: dimms=$dimms configured_mts=${mts:-unknown} channels=$n ($src)"
  if [ -n "$mts" ] && [ "$n" -gt 0 ]; then
    echo "memory: host RAM ceiling (derived) = $n x $mts MT/s x 8 B = $(awk -v n="$n" -v m="$mts" 'BEGIN { printf "%.1f", n * m * 8 / 1000 }') GB/s"
  fi
}

if [ $self = 1 ]; then
  fx=$here/testdata/lspci_b70.txt
  lspci_file=$fx
  out=$(links idle); rc=$?
  want=$(sed -n 's/^# expect //p' "$fx")
  got=$(grep '^links:' <<< "$out")
  ok=1
  [ "$got" = "$want" ] || { echo "self-test: links line '$got', expected '$want'"; ok=0; }
  [ "$rc" = 0 ] || { echo "self-test: exit $rc on a fixture with no STOP"; ok=0; }
  grep -q 'role=endpoint cap=Gen1 x1' <<< "$out" || { echo "self-test: the internal endpoint link not printed"; ok=0; }
  # the STOP path: dev1's root port at x4 (a temporary copy of the fixture)
  tmp=$(mktemp "${TMPDIR:-/tmp}/offload_links.XXXXXX")
  awk '/^00:03.0 / { on = 1 } /^$/ { on = 0 } { if (on && /LnkSta:/) sub(/Width x8/, "Width x4"); print }' "$fx" > "$tmp"
  lspci_file=$tmp
  out2=$(links idle); rc2=$?
  rm -f "$tmp"
  grep -q '^STOP: dev1 (0000:08:00.0) trains below Gen3 x8 (Gen3 x4' <<< "$out2" || { echo "self-test: no STOP line at x4"; ok=0; }
  [ "$rc2" = 3 ] || { echo "self-test: exit $rc2 at x4 (want 3)"; ok=0; }
  # the memory parser over a two-DIMM sample
  dmi=$(mktemp "${TMPDIR:-/tmp}/offload_links_dmi.XXXXXX")
  printf 'Memory Device\n\tSize: 32 GB\n\tSpeed: 2400 MT/s\n\tConfigured Memory Speed: 2133 MT/s\n\nMemory Device\n\tSize: No Module Installed\n\nMemory Device\n\tSize: 32 GB\n\tSpeed: 2400 MT/s\n\tConfigured Memory Speed: 2133 MT/s\n' > "$dmi"
  dmi_file=$dmi
  m=$(memory)
  rm -f "$dmi"
  grep -q 'dimms=2 configured_mts=2133 channels=2' <<< "$m" || { echo "self-test: memory '$m'"; ok=0; }
  grep -q '= 2 x 2133 MT/s x 8 B = 34.1 GB/s' <<< "$m" || { echo "self-test: ceiling '$m'"; ok=0; }
  # without root lspci hides LnkCap / LnkSta: the world-readable sysfs link files answer instead (a fake tree here,
  # dev1's root port at Gen3 x4 - the STOP must still be seen)
  tmp=$(mktemp "${TMPDIR:-/tmp}/offload_links_nolnk.XXXXXX")
  sfs=$(mktemp -d "${TMPDIR:-/tmp}/offload_links_sysfs.XXXXXX")
  grep -v -E 'LnkCap:|LnkSta:' "$fx" > "$tmp"
  for b in 0000:00:02.0:8.0:16:8.0:16 0000:02:00.0:8.0:16:16.0:16 0000:00:03.0:8.0:4:8.0:16 0000:06:00.0:8.0:8:16.0:16; do
    IFS=: read -r d1 d2 d3 cs cw ms mw <<< "$b"
    mkdir -p "$sfs/$d1:$d2:$d3"
    echo "$cs GT/s PCIe" > "$sfs/$d1:$d2:$d3/current_link_speed"
    echo "$cw" > "$sfs/$d1:$d2:$d3/current_link_width"
    echo "$ms GT/s PCIe" > "$sfs/$d1:$d2:$d3/max_link_speed"
    echo "$mw" > "$sfs/$d1:$d2:$d3/max_link_width"
  done
  lspci_file=$tmp
  out3=$(OFFLOAD_SYSFS=$sfs links idle); rc3=$?
  rm -rf "$tmp" "$sfs"
  grep -q '^links: dev0 Gen3 x16, dev1 Gen3 x4' <<< "$out3" || { echo "self-test: the sysfs fallback read '$(grep '^links:' <<< "$out3")'"; ok=0; }
  grep -q 'role=root cap=Gen3 x16 sta=Gen3 x4 state=idle (sysfs)' <<< "$out3" || { echo "self-test: no sysfs-marked line"; ok=0; }
  [ "$rc3" = 3 ] || { echo "self-test: exit $rc3 on the sysfs x4 (want 3)"; ok=0; }
  echo "$out" | sed 's/^/  /'
  if [ $ok = 1 ]; then echo "offload_links self-test: OK"; exit 0; fi
  echo "offload_links self-test: FAIL"; exit 1
fi

rc=0
links idle; r=$?
[ "$r" = 3 ] && rc=3
if [ -z "$lspci_file" ]; then
  echo "lscpu:"
  lscpu 2>/dev/null | grep -E 'Model name|Socket|Core\(s\)|Thread\(s\)|NUMA node\(s\)|MHz' | sed 's/^/  /'
fi
memory
if [ "$load" != 0 ] && [ -z "$lspci_file" ]; then
  probe=${PROBE_OFFLOAD:-$root/build/tools/probe/probe_offload}
  if [ ! -x "$probe" ]; then
    echo "under-load: SKIP no $probe (build probe_offload, or set PROBE_OFFLOAD)"
  else
    cap=$((load + 120))
    if command -v timeout > /dev/null; then
      timeout "$cap" "$probe" hold --dev both --seconds "$load" > "${TMPDIR:-/tmp}/offload_links_hold.$$" 2>&1 &
    else
      "$probe" hold --dev both --seconds "$load" > "${TMPDIR:-/tmp}/offload_links_hold.$$" 2>&1 &
    fi
    pid=$!
    sleep "$(awk -v s="$load" 'BEGIN { printf "%.1f", s / 2 }')"
    links load; r=$?
    [ "$r" = 3 ] && rc=3
    wait "$pid"; hrc=$?
    sed 's/^/  hold: /' "${TMPDIR:-/tmp}/offload_links_hold.$$"
    rm -f "${TMPDIR:-/tmp}/offload_links_hold.$$"
    echo "under-load: probe_offload hold exit $hrc"
  fi
fi
exit $rc
