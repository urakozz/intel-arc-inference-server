# Sourced by the box helpers: BOX (the box's ssh target, user@host) from the environment, else from the
# git-ignored .env in the main checkout's root - one file that every worktree reads too. No default: a wrong
# address should fail loudly rather than quietly talk to the wrong machine.
if [ -z "${BOX:-}" ]; then
  _box_env="$(git -C "$(dirname "${BASH_SOURCE[0]:-$0}")" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)/../.env"
  # shellcheck source=/dev/null
  [ -r "$_box_env" ] && . "$_box_env"
  unset _box_env
fi
