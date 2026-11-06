#!/bin/bash
# ctest's launcher for every test (CMAKE_TEST_LAUNCHER, set in the top CMakeLists.txt): runs the
# test as it is. With B70_TRACE_DIR set (tools/box_validate and tools/box_check.sh set it), it runs
# the test under strace and keeps the list of files the test - and every process it starts - looked
# at, plus the environment it saw: tools/box_validate/reuse.py turns that into the test's inputs, so
# a later run whose inputs are byte-identical takes the old PASS instead of running the test again.
# strace stops only the traced file and exec calls (seccomp-bpf); the GPU work is untouched.
if [ -z "${B70_TRACE_DIR:-}" ] || ! command -v strace > /dev/null; then exec "$@"; fi
name=${B70_TEST_NAME:-unnamed}
mkdir -p "$B70_TRACE_DIR"
env | grep -E '^(B70_|ZE_|HF_|ONEAPI_|SYCL_|IGC_|NEO|OCL_|LD_LIBRARY_PATH=|LD_PRELOAD=)' \
  | grep -vE '^(B70_TRACE_DIR|B70_TEST_NAME)=' | LC_ALL=C sort > "$B70_TRACE_DIR/$name.env"
pwd > "$B70_TRACE_DIR/$name.cwd"
exec strace -f -qq --seccomp-bpf --kill-on-exit -s 4096 \
  -e trace=open,openat,openat2,creat,execve,execveat,stat,lstat,newfstatat,statx,access,faccessat,faccessat2,readlink,readlinkat,chdir,fchdir,truncate,rename,renameat,renameat2,unlink,unlinkat,mkdir,mkdirat,symlink,symlinkat,link,linkat \
  -o "$B70_TRACE_DIR/$name.trace" "$@"
