# Licenses of the Java sources

`episodes.jsonl` contains short excerpts of Java source code: the helper method a developer introduced, and the
enclosing methods of the sites at which the commit called it (before and after the commit). They come from 14 public
repositories at 50 commits and reached this repository through the public SWE-Refactor archive
(Xu, Yang and Chen, 2026, `xu2026swerefactor`; arXiv 2602.03712; Zenodo 10.5281/zenodo.17196850, version 3 is
10.5281/zenodo.22001546). The Zenodo records of both versions declare the license **CC BY 4.0**; that license
covers the archive as distributed. The excerpts stay under the license of the repository they were taken from, listed
below. The MIT license of this repository's code does not apply to `data/java/episodes.jsonl`.

How the licenses were determined (13 October 2026): for every repository and every recorded commit, the license file(s)
were fetched from the repository at that commit and identified by their text. Each repository has a single license
text across all its commits, except PMD, whose `LICENSE` file differs between its two commits (the BSD-style
text at the top is the same). The SHA-256 prefix names the file that was read. File headers of individual source files
were not audited, except that a `hibernate-orm` source file at the first recorded commit was read and states
"LGPL, version 2.1 or later".

| Repository | Upstream | License | Commits | Episodes |
|---|---|---|---|---|
| checkstyle | https://github.com/checkstyle/checkstyle | LGPL-2.1 | 2 | 5 |
| commons-io | https://github.com/apache/commons-io | Apache-2.0 | 18 | 45 |
| commons-lang | https://github.com/apache/commons-lang | Apache-2.0 | 5 | 24 |
| gson | https://github.com/google/gson | Apache-2.0 | 3 | 6 |
| guava | https://github.com/google/guava | Apache-2.0 | 1 | 1 |
| hertzbeat | https://github.com/apache/hertzbeat | Apache-2.0 | 3 | 6 |
| hibernate-orm | https://github.com/hibernate/hibernate-orm | LGPL-2.1 | 3 | 3 |
| hibernate-search | https://github.com/hibernate/hibernate-search | LGPL-2.1 | 4 | 4 |
| jadx | https://github.com/skylot/jadx | Apache-2.0 | 1 | 4 |
| javaparser | https://github.com/javaparser/javaparser | LGPL-3.0 or Apache-2.0 (user's choice) | 3 | 8 |
| junit4 | https://github.com/junit-team/junit4 | EPL-1.0 | 2 | 7 |
| junit5 | https://github.com/junit-team/junit5 (now junit-team/junit-framework) | EPL-2.0 | 2 | 8 |
| pmd | https://github.com/pmd/pmd | BSD-style (PMD's own text) | 2 | 5 |
| shenyu | https://github.com/apache/shenyu | Apache-2.0 | 1 | 1 |

Notes. The excerpts of Checkstyle, Hibernate ORM and Hibernate Search are LGPL-2.1 material, the excerpts of JUnit 4
and JUnit 5 are Eclipse Public License material, and PMD's are under its own BSD-style license; the excerpts are
redistributed here as text in a data file. Every excerpt is attributed to its repository, commit and file by the
fields `repository`, `commit` and `file` of its record. JavaParser offers two licenses; the choice is the
user's.

## Per repository: commits and license evidence

### checkstyle
- Upstream: https://github.com/checkstyle/checkstyle
- License: **LGPL-2.1** (GNU Lesser General Public License v2.1).
- Commits: 21c8e04bc9 b142dda472
  - License files at these commits: `LICENSE` (SHA-256 9b872a8a070b…)

### commons-io
- Upstream: https://github.com/apache/commons-io
- License: **Apache-2.0** (Apache License 2.0).
- Commits: 153a91c127 1e73dd92a2 269f9921cb 5035e97fa0 54858b5f9f 69e3b819a1 6b0ed87f7b 6dd4c0850c 8b6d4969ff 8b7c82225a a304a37c7f cb6f3aabcf d269f164b3 d75027d357 dcf0197fbb dd63fa3b53 e95c77c48c f448f97af4
  - License files at these commits: `LICENSE.txt` (SHA-256 8c6db3404751…)

### commons-lang
- Upstream: https://github.com/apache/commons-lang
- License: **Apache-2.0** (Apache License 2.0).
- Commits: 426ebb13ee 5d146cd069 bdff802a85 c290aebffa c95ee37dff
  - License files at these commits: `LICENSE.txt` (SHA-256 cfc7749b96f6…)

### gson
- Upstream: https://github.com/google/gson
- License: **Apache-2.0** (Apache License 2.0).
- Commits: 08174e2019 46ab704221 810e356059
  - License files at these commits: `LICENSE` (SHA-256 cfc7749b96f6…)

### guava
- Upstream: https://github.com/google/guava
- License: **Apache-2.0** (Apache License 2.0).
- Commits: c28e65234e
  - License files at these commits: `LICENSE` (SHA-256 cfc7749b96f6…)

### hertzbeat
- Upstream: https://github.com/apache/hertzbeat
- License: **Apache-2.0** (Apache License 2.0).
- Commits: 477af0959f c4fa738d65 f10e5dbd06
  - License files at these commits: `LICENSE` (SHA-256 49798571faa8…)

### hibernate-orm
- Upstream: https://github.com/hibernate/hibernate-orm
- License: **LGPL-2.1** (`lgpl.txt` is the GNU Lesser General Public License v2.1; the header of a source file at the first commit says "version 2.1 or later").
- Commits: 4b33d0d067 800a3f0738 afca93158e
  - License files at these commits: `lgpl.txt` (SHA-256 dc626520dcd5…)

### hibernate-search
- Upstream: https://github.com/hibernate/hibernate-search
- License: **LGPL-2.1** (GNU Lesser General Public License v2.1).
- Commits: 9cd4135b36 a4d2e5b39b c46066ddc2 f7c9efb791
  - License files at these commits: `lgpl.txt` (SHA-256 dc626520dcd5…)

### jadx
- Upstream: https://github.com/skylot/jadx
- License: **Apache-2.0** (Apache License 2.0).
- Commits: d4ce969fb7
  - License files at these commits: `LICENSE` (SHA-256 b40930bbcf80…)

### javaparser
- Upstream: https://github.com/javaparser/javaparser
- License: **LGPL-3.0 or Apache-2.0**, the user's choice (its `LICENSE` says "available either under the terms of the LGPL License or the Apache License"; the texts are `LICENSE.LGPL`, version 3, and `LICENSE.APACHE`).
- Commits: 339d45a569 a73aba7b8c b22ebb29ab
  - License files at these commits: `LICENSE` (SHA-256 b8655bf6423a…), `LICENSE.APACHE` (SHA-256 cfc7749b96f6…), `LICENSE.GPL` (SHA-256 e73d7dbbb975…), `LICENSE.LGPL` (SHA-256 319917f5ccd0…)

### junit4
- Upstream: https://github.com/junit-team/junit4
- License: **EPL-1.0** (Eclipse Public License v1.0).
- Commits: 2adef6f30c 2db6394288
  - License files at these commits: `LICENSE-junit.txt` (SHA-256 9648bb2891b9…)

### junit5
- Upstream: https://github.com/junit-team/junit5 (now junit-team/junit-framework)
- License: **EPL-2.0** (Eclipse Public License v2.0).
- Commits: e07cbb6162 f95d4e8b09
  - License files at these commits: `LICENSE.md` (SHA-256 5aa4cd44c111…)

### pmd
- Upstream: https://github.com/pmd/pmd
- License: **BSD-style** (PMD's own license text, headed "PMD's BSD-style license", with an end-user acknowledgement clause; no SPDX identifier is claimed here).
- Commits: 18f393da3c
  - License files at these commits: `LICENSE` (SHA-256 bc61ca15a6e9…)
- Commits: 30b76285e3
  - License files at these commits: `LICENSE` (SHA-256 296dae48eae5…)

### shenyu
- Upstream: https://github.com/apache/shenyu
- License: **Apache-2.0** (Apache License 2.0).
- Commits: 34aeadeac6
  - License files at these commits: `LICENSE` (SHA-256 a0f69689eabc…)

## Not verified

All 14 repository licenses were read at the recorded commits. Two things were not checked: the license header of
each individual source file, and whether every excerpt is original to its repository (code copied into a repository
from elsewhere keeps its own license).
