# Benchmarks

Every speed the [README](../README.md) quotes, and the rest, per model and setup. Measured on the owner's PCIe 3.0
workstation (DDR4-2133, two Intel Arc Pro B70 with peer-to-peer). Greedy decoding; served through `b70-serve` unless a
row says otherwise. How the sets are built and timed: [17-running-models.md](17-running-models.md#how-to-read-the-numbers).

**Prefill** is the model reading your prompt, **decode** is the model writing its answer

### Qwen3.8-Flash-Next

Prefill, t/s:

| setup | 32k-token prompt | 128k-token prompt |
|---|---:|---:|
| two cards `pp` | **3,415** | **2,918** |
| two cards `tp` | 2,234 | 1,980 |

A short prompt (43 tokens) gets its first token in 0.30 s on `pp` and on `tp`

Generation, t/s:

| setup | code | prose | tool calls |
|---|---:|---:|---:|
| MTP, two cards `pp` | 79 | 63 | 98-123 |
| MTP, two cards `tp` | **99** | **64** | **84-126** |
| MTP, two cards `tp`, PCIe 5.0 x16 + DDR5 | ~125-155 | | |

`--kv-cache int8` halves Flash-Next's context (12.8 KiB a position against 24.2 KiB) at the same speed: generation and
prefill within 2 % of bf16 in both modes. On `pp`, `--max-len 131072 --kv-cache int8` decodes as fast as bf16 at
65,536

Flash-Next needs at least 128 GB of system RAM: the experts that do not fit VRAM, its n-gram table and the prefix
cache take about 104 GB of it

Measured on a PCIe 3.0 workstation with DDR4-2133 and P2P. PCIe 5.0 x16, DDR5 RAM, AVX-512 CPU speed it up

### Inkling 2 Mini

| setup | code | prose | tool calls |
|---|---:|---:|---:|
| one card | 220.5 | 287.9 | 176.4 |
| two cards `pp` | 207.7 | 281.3 | 177.3 |
| two cards `tp` | 274.8 | 376.6 | 219.1 |
| 2 requests, two cards `replica` (total) | 351 | 452 | 286 |

| prefill (`l0-int8`, int8 cache), t/s | 16k | 32k | 64k |
|---|---:|---:|---:|
| one card | 4,506 | 3,939 | 3,137 |
| two cards `pp` | **7,194** | **6,908** | **5,842** |
| two cards `tp` | 6,384 | 5,794 | 4,801 |

### Qwen3.8-27B

Prefill, t/s:

| setup | 4k prompt | 16k prompt | 32k prompt | 64k prompt | 128k prompt |
|---|---:|---:|---:|---:|---:|
| one card, MTP on (the default server) | 1,908 | 1,771 | 1,634 | 1,395 | 1,072 |
| one card, MTP on, `--prefill-backend l0-int8` (opt-in) | **2,800** | **2,564** | **2,346** | **1,985** | **1,500** |
| two cards, `replica`, 2 requests (total), `--prefill-backend l0-int8` (opt-in) | **4,610** | - | ~4,000 | - | - |

Generation, t/s (several requests: the total; llama-benchy tg128 after 4k / 32k / 128k tokens of context, median of 3):

| setup | code | prose | tool calls | llama-benchy |
|---|---:|---:|---:|---:|
| MTP, 1 request, one card | 135 | 50 | 102 | 67 / 58 / 51 |
| MTP, 1 request, two cards `tp` | 138 | 74 | 115 | 95 / 84 / **69** |
| MTP, 4 requests, one card | 237 | 140 | 122 | |
| MTP, 2 requests, two cards `replica` | 168 | 99 | 144 | |
| MTP, 4 requests, two cards `replica` | 283 | 172 | 168 | |
| MTP, 8 requests, two cards `replica` | 435 | 268 | 206 | |
| DFlash, 1 request, one card | **167** | 67 | 118 | 94 / 83 / 55 (`--kv-cache int8`) |
| DFlash, 1 request, two cards `tp` | 182 | 89 | 153 | **119** / **107** / 68 |
| DFlash, 2 requests, one card | 233 | 98 | 114 | |
| DFlash, 4 requests, one card | **343** | 156 | 128 | |
| DFlash, 2 requests, two cards `replica` | 304 | 126 | 151 | |
| DFlash, 4 requests, two cards `replica` | 433 | 189 | 199 | |
| DFlash, 8 requests, two cards `replica` | **626** | **281** | **211** | |

DFlash is `--spec dflash --draft-model urakozz/Qwen3.8-27B-DFlash2-W4A16-g64-AutoRound`

### Ornith 1.5 35B-A3B

Prefill, t/s:

| setup | 4k prompt | 16k prompt | 32k prompt | 64k prompt | 128k prompt | 262k prompt |
|---|---:|---:|---:|---:|---:|---:|
| one card | 11,223 | 8,800 | 7,420 | 5,428 | 3,544 | 2,096 |
| two cards `replica`, 2 requests (total) | **19,117** | **13,728** | - | - | - | - |

Generation with MTP, t/s (llama-benchy as above):

| setup | code | prose | tool calls | llama-benchy |
|---|---:|---:|---:|---:|
| 1 request, one card | 320 | 341 | 287 | **196** / 126 / 62 |
| 2 requests, two cards `replica` (total) | **581** | **605** | **453** | |
| 4 requests, one card | ~430 | | | |

