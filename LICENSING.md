# Licensing

`SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial`

This project is dual-licensed. You pick one of two licenses:

1. **GNU Affero General Public License, version 3 only** (AGPL-3.0-only). The full text is in [LICENSE](LICENSE).
   It is free and open to everyone.
2. **A commercial license**, for companies that cannot or do not want to meet the AGPL's obligations.
   [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) is the template such an agreement starts from.

Copyright (C) Jürgen Kozyrev.

This page is a plain-language summary, not legal advice. Where it and the license text disagree, the license text
wins.

## What the AGPL means for you

- **Running it for yourself is fine.** You can use and change the engine on your own machines without publishing
  anything, as long as you are the only one using your changed version.
- **Distributing it means sharing the source.** If you give the software to someone else, changed or not, you give
  them the complete source under the AGPL too.
- **Serving it over a network means sharing the source.** If other people use your modified version over a
  network (for example through the OpenAI-compatible server), you must offer them the complete source
  of that modified version under the AGPL. This is the part that sets the AGPL apart from the GPL.
- **What you build on it follows the same license.** A work that includes or is derived from this code is covered by
  the AGPL as a whole.

Programs that only talk to the server over its HTTP API are separate programs and are not affected.

## Who needs the commercial license

You need it if you want to do any of the following without publishing your source under the AGPL:

- ship the engine, or something built from it, inside a closed-source product;
- run a hosted service on a modified version and keep the changes private;
- embed it in proprietary hardware or software, such as an appliance or a workstation image.

## What the commercial license gives

- The right to use, modify and distribute the software without the AGPL's obligations, on the terms agreed.
- Optionally, support or consulting, agreed per deal.

## How to ask

Open an issue in this repository or contact the author through the GitHub profile.

## Earlier versions

Versions published before 2026-11-17 were released under the Apache License 2.0. Those copies stay under Apache-2.0
for everyone who has them. The dual license applies to that date's version and everything after it.

## Third-party code

Code from other projects keeps its own license. Neither the AGPL nor the commercial license changes that.

| Path | Project | License |
|---|---|---|
| `third_party/httplib/` | cpp-httplib | MIT |
| `third_party/json/` | nlohmann/json | MIT |
| `third_party/minja/` | minja (with local patches, listed in `third_party/VERSIONS`) | MIT |
| `tools/oracle/third_party/kolibri1/` | a transformers port of Aleph Alpha's vLLM plugin | Apache-2.0 |

A file that names another license in its own `SPDX-License-Identifier` header stays under that license. Today these are
`tools/oracle/dflash_ref.py` and `tools/oracle/kolibri_ref.py`, which follow Apache-2.0 reference code closely.

Model weights are not part of this repository and carry the licenses of their publishers.
