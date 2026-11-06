#!/usr/bin/env bash
# The box's model storage tiers, ON THE BOX: NVMe (hot) / 8 TB (cold) / 3 TB (archive).
#   tools/hf_tier.sh status               every model: tier, size, copies; free space per disk
#   tools/hf_tier.sh hot <model>...       copy to the NVMe, verify, the cache entry = real folder
#   tools/hf_tier.sh cold <model>...      8 TB copy complete + verified, the entry = link to it
#   tools/hf_tier.sh archive <model>...   the same on the 3 TB disk (parked models)
#   tools/hf_tier.sh sync <model>...      only make the 8 TB copy complete (entry untouched)
# A model is models--Org--Name or Org/Name. Options (before or after the command): --hot DIR,
# --cold DIR, --archive DIR, --hot-trash DIR, --min-free-gb N (what `hot` leaves free, 40),
# --jobs N (parallel verify readers, 3). Nothing is deleted: what a swap replaces goes to
# <disk>/trash/<date>/hf-tier/ (~/b70/trash, ~/b70/hdd/trash, ~/b70/hdd3/trash) for the
# operator. A model a running process has open is refused (exit 3). Big copies take hours:
# run detached, e.g.
#   mkdir -p ~/b70/logs/hf-tier && setsid nohup tools/hf_tier.sh hot Org/Name \
#     > ~/b70/logs/hf-tier/$(date +%F-%H%M)-hot.log 2>&1 < /dev/null &
# The logic and its rules are in tools/hf_tier.py; tools/test_hf_tier.py tests them on a fake hub.
set -euo pipefail
exec python3 "$(dirname "${BASH_SOURCE[0]:-$0}")/hf_tier.py" "$@"
