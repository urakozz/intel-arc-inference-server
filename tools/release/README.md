# intel-inference-server (Linux tarball)

An inference engine for Intel Arc Pro B70 graphics cards, as a plain tarball - no Docker. Extract it anywhere
and run `bin/b70-serve` (an OpenAI-compatible server) or `bin/b70-decode` (a single-model benchmark) directly.

```sh
tar -xzf intel-inference-server-1.2.0-linux-x86_64.tar.gz
cd intel-inference-server-1.2.0
```

## What is inside

```
intel-inference-server-1.2.0/
├── bin/
│   ├── b70-serve     # the server: OpenAI-compatible API on port 8000
│   └── b70-decode    # decode-step benchmark for one checkpoint
├── kernels/          # the compiled device kernels the binaries load
├── lib/              # GCC 16's libstdc++ and libgcc_s (found via the binaries' rpath)
├── README.md
├── LICENSE
└── LICENSING.md
```

## What the host must provide

The tarball does not bring the GPU stack; it must match the kernel driver on the host, so install these from
the host's packages:

- the `xe` kernel driver (the kernel module for Intel Arc Pro B70);
- Intel's compute runtime - the Level Zero GPU driver `libze_intel_gpu` with its compiler, version
  `26.40.40196.15` or newer (Ubuntu: the `libze-intel-gpu1` package from Intel's graphics PPA,
  `ppa:kobuk-team/intel-graphics`, or your distro's package);
- the Level Zero loader, `libze_loader.so.1` (`libze1`), version `1.34.0` or newer.

On Ubuntu 26.04:

```sh
sudo add-apt-repository -y ppa:kobuk-team/intel-graphics
sudo apt install libze-intel-gpu1 libze1
```

The binaries need glibc 2.43 or newer (they are built on Ubuntu 26.04) and access to the render nodes
(`stat -c %g /dev/dri/renderD128` names the group to run as, e.g. `render`; the fast two-card session also
needs peer-to-peer between the cards, native from kernel 7.4).

## Running

The engine never downloads a model: fetch it into the Hugging Face cache first (see the project README for the
`hf download` lines), then start the server with the cache pointed at:

```sh
export HF_HOME=$HOME/.cache/huggingface   # where `hf download` put the model
./bin/b70-serve urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ --prefix-split-last --kv-cache int8
```

`b70-serve --help` and `b70-decode --help` list every flag. Cards are picked with `ZE_AFFINITY_MASK`
(e.g. `ZE_AFFINITY_MASK=0,1` for both).
