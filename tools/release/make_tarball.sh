#!/bin/sh
# Assembles the Linux release tarball beside the Docker image:
# intel-inference-server-<version>-linux-x86_64.tar.gz, extracting to intel-inference-server-<version>/.
# <version> is the project's version in CMakeLists.txt.
#
#   tools/release/make_tarball.sh <build dir> <output dir>
#
# Run it in the Dockerfile's builder stage (Ubuntu 26.04, GCC 16.2), so the glibc floor is that release's:
#
#   docker build --target tarball -t urakozz/intel-arc-inference-server:tarball .
#   docker create --name tarball urakozz/intel-arc-inference-server:tarball
#   docker cp tarball:/out/. <output dir>/ && docker rm tarball
set -eu

build=${1:?usage: make_tarball.sh <build dir> <output dir>}
out=${2:?usage: make_tarball.sh <build dir> <output dir>}
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)

version=$(sed -n 's/^project(b70_inference_server VERSION \([0-9][0-9.]*\).*/\1/p' "$root/CMakeLists.txt")
[ -n "$version" ] || { echo "make_tarball.sh: no project version in $root/CMakeLists.txt" >&2; exit 1; }
name="intel-inference-server-$version"

stage=$(mktemp -d)
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/$name/bin" "$stage/$name/kernels" "$stage/$name/lib"

cp "$build/src/cli/b70-serve" "$build/src/cli/b70-decode" "$stage/$name/bin/"
cp -a "$build/kernels/." "$stage/$name/kernels/"
# GCC 16's libstdc++ and libgcc_s, which the binaries find through their $ORIGIN/../lib rpath.
cxx=${CXX:-/usr/local/bin/g++-16.2}
libdir=$(dirname "$("$cxx" -print-file-name=libstdc++.so)")
cp -a "$libdir"/libstdc++.so.6* "$libdir/libgcc_s.so.1" "$stage/$name/lib/"
cp "$root/tools/release/README.md" "$stage/$name/README.md"
cp "$root/LICENSE" "$root/LICENSING.md" "$stage/$name/"

mkdir -p "$out"
tar -czf "$out/$name-linux-x86_64.tar.gz" -C "$stage" "$name"
echo "$out/$name-linux-x86_64.tar.gz"
