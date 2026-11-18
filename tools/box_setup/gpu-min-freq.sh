#!/usr/bin/env bash
# Lets every xe GPU clock down to its hardware minimum (rpn_freq, 400 MHz on a B70) while awake but idle.
# The driver's default minimum on the B70 is 1200 MHz. Under load the firmware still raises the clock
# to max_freq, so only an awake, quiet card is affected; a card in C6 is powered down either way.
# Run as root at boot by gpu-min-freq.service; waits up to 60 s for the driver to bring the cards up.
set -euo pipefail
for _ in $(seq 60); do
  compgen -G '/sys/class/drm/card*/device/tile*/gt*/freq0/min_freq' > /dev/null && break
  sleep 1
done
for f in /sys/class/drm/card*/device/tile*/gt*/freq0; do
  [ -w "$f/min_freq" ] || continue
  rpn=$(cat "$f/rpn_freq")
  echo "$rpn" > "$f/min_freq"
  echo "$f: min_freq $(cat "$f/min_freq") MHz"
done
