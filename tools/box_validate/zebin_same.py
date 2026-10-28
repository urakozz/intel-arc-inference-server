#!/usr/bin/env python3
"""G0's second look at a kernel binary that differs (tools/box_validate/g0_compare.sh): the same ELF sections with
the same contents once each tree's root path is replaced by one token?

    zebin_same.py <baseline .bin> <under-test .bin> <baseline tree root> <tree root under test>

ocloc keeps the build options in the zebin, so a command line naming an absolute include directory (Bolt's
`-I<tree>/src/kernels/bolt`) differs between two checkouts by the path alone - and with it every offset in the ELF
headers. Compared here: the section names in order and each section's bytes with the root replaced; the headers are
not compared (their offsets follow from the sections). Exit 0 same, 1 different, 2 not ELF64.
"""
import struct
import sys


def sections(path: str, root: str):
    data = open(path, "rb").read()
    if data[:4] != b"\x7fELF" or data[4] != 2:
        sys.exit(2)
    shoff, = struct.unpack_from("<Q", data, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    hdr = [struct.unpack_from("<IIQQQQIIQQ", data, shoff + i * shentsize) for i in range(shnum)]
    strtab = hdr[shstrndx]
    names = data[strtab[4]:strtab[4] + strtab[5]]
    out = []
    for h in hdr:
        name = names[h[0]:names.index(b"\0", h[0])]
        body = b"" if h[1] == 8 else data[h[4]:h[4] + h[5]]   # SHT_NOBITS has no bytes
        out.append((name, h[1], body.replace(root.encode(), b"<tree>")))
    return out


if __name__ == "__main__":
    a, b, ra, rb = sys.argv[1:5]
    sys.exit(0 if sections(a, ra) == sections(b, rb) else 1)
