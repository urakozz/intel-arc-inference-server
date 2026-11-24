# syntax=docker/dockerfile:1
#
# b70-serve for Intel Arc Pro B70 cards, in three stages: GCC 16.2 built from source, a builder with the Intel compute
# toolchain, CMake and Rust, and a slim runtime with the Level Zero runtime libraries, GCC 16.2's libstdc++, the
# binaries and the compiled device kernels.
#
#   docker build -t urakozz/intel-arc-inference-server .
#
#   docker run --rm --device /dev/dri --group-add "$(getent group render | cut -d: -f3)" \
#       -v ~/.cache/huggingface:/root/.cache/huggingface -p 8000:8000 \
#       urakozz/intel-arc-inference-server urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --prefix-split-last --kv-cache int8
#
# - The host needs the `xe` kernel driver; the image brings the user-space runtime, not the driver.
# - `--device /dev/dri` hands over every card. `--group-add` with the host's render group id lets the server open the
#   render nodes (`stat -c %g /dev/dri/renderD128` names it). The fast single session on two cards needs
#   peer-to-peer between them, native from kernel 7.4.
# - The engine never downloads: fetch the model into the Hugging Face cache on the host first (see the README) and
#   mount the cache as above. The server listens on port 8000. The entrypoint is b70-serve, so the arguments after
#   the image name are the model and its flags.

ARG UBUNTU=ubuntu:26.04
ARG GCC_VERSION=16.2.0
ARG RUST_VERSION=1.93.1

# ---- GCC 16.2 ------------------------------------------------------------------------------------------------------
# Built from the GNU source tarball with the configuration of the development box. No distribution ships GCC 16.2
# yet and the build needs it (C++26 with -freflection and contracts). It installs as /usr/local/bin/g++-16.2, the
# path the top-level CMakeLists.txt looks for.
FROM ${UBUNTU} AS gcc
ARG GCC_VERSION
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl xz-utils bzip2 file flex bison texinfo \
    && apt-get clean && find /var/lib/apt/lists -type f -delete
WORKDIR /gcc
RUN curl -fsSL "https://ftp.gnu.org/gnu/gcc/gcc-${GCC_VERSION}/gcc-${GCC_VERSION}.tar.xz" | tar -xJ \
    && cd "gcc-${GCC_VERSION}" && ./contrib/download_prerequisites
RUN mkdir build && cd build \
    && "../gcc-${GCC_VERSION}/configure" --prefix=/usr/local --enable-languages=c,c++ --disable-multilib \
           --disable-bootstrap --program-suffix=-16.2 \
    && make -j"$(nproc)" && make install-strip DESTDIR=/gcc-root

# ---- Builder -------------------------------------------------------------------------------------------------------
FROM ${UBUNTU} AS builder
ARG RUST_VERSION
# Intel's compute stack from the graphics PPA, pinned to what the development box runs: compute runtime
# 26.40.40196.15 (ocloc and the Level Zero driver, which pull in the matching IGC) and Level Zero loader 1.34.0.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl git cmake ninja-build pkg-config python3 build-essential ccache \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        intel-ocloc=26.40.40196.15-1~26.04~ppa1 \
        libze-intel-gpu1=26.40.40196.15-1~26.04~ppa1 \
        libze1=1.34.0-1~26.04~ppa1 \
        libze-dev=1.34.0-1~26.04~ppa1 \
    && apt-get clean && find /var/lib/apt/lists -type f -delete
COPY --from=gcc /gcc-root/usr/local /usr/local
RUN curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain "${RUST_VERSION}"
ENV PATH=/root/.cargo/bin:${PATH}
WORKDIR /opt/b70
COPY . .
# The kernel directory's path is compiled into the binaries, so the runtime stage keeps /opt/b70/build/kernels.
# Every kernel is compiled fresh; ccache lives in a BuildKit cache mount, so a rebuild recompiles only changed C++.
ENV B70_OCLOC_CACHE=0 CCACHE_DIR=/root/.cache/ccache
RUN --mount=type=cache,id=b70-ccache,target=/root/.cache/ccache \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    && cmake --build build -j"$(nproc)"

# ---- Runtime -------------------------------------------------------------------------------------------------------
FROM ${UBUNTU} AS runtime
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libze-intel-gpu1=26.40.40196.15-1~26.04~ppa1 \
        libze1=1.34.0-1~26.04~ppa1 \
    && apt-get purge -y software-properties-common && apt-get autoremove -y \
    && apt-get clean && find /var/lib/apt/lists -type f -delete
# GCC 16.2's libstdc++ and libgcc_s are newer than the system's; the binaries find them through the rpath the build
# links in (/usr/local/lib64).
COPY --from=gcc /gcc-root/usr/local/lib64/libstdc++.so.6* /gcc-root/usr/local/lib64/libgcc_s.so.1 /usr/local/lib64/
COPY --from=builder /opt/b70/build/src/cli/b70-serve /opt/b70/build/src/cli/b70-decode /opt/b70/build/src/cli/
COPY --from=builder /opt/b70/build/kernels /opt/b70/build/kernels
ENV HF_HOME=/root/.cache/huggingface
EXPOSE 8000
ENTRYPOINT ["/opt/b70/build/src/cli/b70-serve"]
