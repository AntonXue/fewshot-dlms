# Provenance of the Java episodes (Section 5, Appendix C)

Section 5 of the paper asks a diffusion language model to propagate a refactoring: a developer extracted a helper
method and rewrote some call sites to use it; the model sees the helper and the rewrite of one or two sites (the
demonstrations) and must rewrite one or two other sites of the same file (the targets). This directory holds the 127
episodes of the evaluation population and the ledgers that say how they were chosen. The code that turns an episode
into the model's input is `canvas.build_java_canvas`; the episodes are read with `data.load_java_episodes()`.

| File | Contents |
|---|---|
| `episodes.jsonl` | the 127 episodes, one JSON record each (fields below) |
| `episode_cells.csv` | the 158 candidate episodes of the G4 population with their flags and which of the two populations (A, B) contain them |
| `exclusion_ledger.csv` | every point at which a candidate left population A or B, with the reason (75 rows) |
| `reference_validity.csv` | the source-validity flags of each candidate and the adjudication that decided it (158 rows) |
| `source_groups.csv` | the 338 helper-extraction groups the census started from, with their site counts |
| `LICENSES.md` | license of every upstream repository, per commit |

What is not here: the pipeline that mined the sites from the source archive and the build-and-test harness that
compiles and tests candidate patches. The episodes are the pipeline's frozen output. `tools/export_java.py` records
how the files were cut from the authors' run.

## 1. Source and sites

**Source.** The SWE-Refactor archive (`xu2026swerefactor`, CC BY 4.0; see `LICENSES.md`) lists 1,099 pure-refactoring
instances of public Java repositories, each with the whole source file before and after the commit and a refactoring
label. Instances of type Extract Method that belong to the same commit and extract the same helper form a *helper
group*; the census found 338 of them (`source_groups.csv`, 18 repositories).

**Sites.** For a group, the after-file is scanned for methods (a brace-matching scanner that ignores comments and
literals). A *site* is a method, matched by name and parameters between the before-file and the after-file, that
gained a call to the helper and is not the helper itself. Its changed region is the smallest range of lines that
differs between the two versions: `span_old` before, `span_new` after. The region must be non-empty and must contain a
call to the helper. 

**The G4 rules** (the census called the population that satisfies them *G4*). A site is eligible only if its
replacement `span_new` is at most 80 LLaDA-1.5 tokens, its enclosing method is at most 200 lines after the change, and
`span_new` survives a tokenizer round trip. A group needs at least `M + J` eligible sites for the cell `M x J`. The
canvas with oracle lengths of every shipped episode has at most 2,048 tokens (the cap on the canvas; the longest is
1,797), and the longest target is 74 tokens.

**Roles and cells.** The four cells are `M x J` in `{1, 2} x {1, 2}`: M demonstrations and J targets, with D1, T1, T2, D2
as the roles. The eligible sites of a group are put in order of their line position and rotated by
`key(260929, "swe_explore", group_id) mod n`. Here `key` is the first 12 hexadecimal digits of the SHA-256 of the
`|`-joined parts, read as an integer, and the rotation is stored in the field `rotation`. D1, T1, T2 and D2 are the
first four sites of the rotated order; D2 is the third site if the group has only three eligible sites. The order
depends only on the source and the group id, not on any model output, and one group gives at most one episode per
cell. The demonstrations of `M = 2` extend those of `M = 1`, and the targets of `J = 2` extend those of `J = 1`.

## 2. From 158 candidates to 127 episodes

The G4 rules leave 158 candidate episodes (69, 33, 33 and 23 groups in the four cells). Two populations are defined
from them (`in_manifest_A`, `in_manifest_B` in `episode_cells.csv`):

- **A**, the provisional population (114 episodes): candidates without a source-validity flag, after a first
  deduplication, made at the time of the census.
- **B**, the evaluation population of the paper (127 episodes, 50 commits, 14 repositories): the candidates after
  the adjudication of the flags (rules R1, R2, R3) and the final deduplication (rules R4, R5) below. A is contained
  in B; the 13 episodes of B that are not in A are the ones re-admitted by the adjudication.

| Cell | Raw groups / commits | G4 groups | A | B episodes / commits |
|---|---|---|---|---|
| M1J1 | 86 / 66 | 69 | 50 | 52 / 50 |
| M1J2 | 45 / 38 | 33 | 23 | 28 / 26 |
| M2J1 | 45 / 38 | 33 | 26 | 28 / 26 |
| M2J2 | 30 / 24 | 23 | 15 | 19 / 17 |
| total | | 158 | 114 | 127 / 50 |

**Flags.** The census flags a candidate when (a) an identifier or literal of the reference replacement is absent from
the whole prompt (`gold_not_inferable_tokens_absent`), (b) the reference changes an expected value that a test asserts
(`expectation_change_in_gold`), (c) the oracle canvas exceeds 2,048 tokens or a target exceeds 256 tokens, or (d) the
group is a mirror duplicate. 23 of the 158 candidates carry flag (a) or (b); `reference_validity.csv` records the
flags and the decision for every candidate (the unflagged candidates of a flagged group carry the group's rule label).
The rules below were fixed from the source text alone, before the evaluation run generated any output.

**R1, missing declaration.** Exclude a candidate only if a target's reference needs an identifier or literal that (i)
is absent from the whole rendered prompt, (ii) is not part of a standard API, (iii) cannot be derived from the
demonstrated pattern or the target's own old text, and (iv) without which no behaviour-preserving rewrite exists.

**R2, expectation change.** Exclude a candidate when the reference of a demonstration or a target changes an asserted
expected value, because the commit then bundles a behaviour change with the extraction.

**R3, overload rendering.** The prompt shows the first method of the helper's name in the after-file. A candidate
whose shown overload is never called by a demonstration or a target is kept and labelled
`rendered_overload_not_called` (four episodes in B, all the `next` helper of one commons-lang commit, whose targets
call sibling overloads that the prompt does not show). The prompt format is not changed for it.

**R4, deduplication.** Keep one helper per (commit, file) unit among the candidates that survive R1 and R2: the helper
with the most eligible sites, ties broken by the first helper name in sorted order. The choice is reused in every cell
the helper qualifies for. Six units had more than one helper:

| Unit (commit, file) | Kept helper | Dropped helpers |
|---|---|---|
| c95ee37dff, `RandomUtils.java` | `secure` | `randomDouble`, `randomFloat`, `randomInt`, `randomLong` |
| c95ee37dff, `RandomStringUtils.java` | `next` | `nextAlphabetic`, `nextAlphanumeric`, `nextAscii`, `nextGraph`, `nextNumeric`, `nextPrint` |
| 08174e2019, `JsonWriterTest.java` | `assertNonFiniteDoublesExceptions` | `assertNonFiniteFloatsExceptions`, `assertNonFiniteNumbersExceptions` |
| 46ab704221, `ExposeAnnotationExclusionStrategyTest.java` | `assertExcludesField` | `assertIncludesField` |
| a73aba7b8c, `RecordDeclarationTest.java` | `assertOneRecordDeclaration` | `assertCompilationFails` |
| 2db6394288, `FailOnTimeoutTest.java` | `failAfter50Ms` | `run` |

**R5, mirror pairs.** Two Guava commits (62a888101d and cdc225474e) that carry the same helper `isLongTimeUnitBased`
count as one unit. Neither is in the G4 population (its targets exceed 80 tokens); the pair is recorded for
completeness.

**Adjudication of the flags.** Of the 23 flagged candidates, 10 are excluded and 13 are kept:

| Group | Flag | Decision | Rule | Reason |
|---|---|---|---|---|
| pmd b25c16f `assertWarningsWithFoo` (4 cells) | tokens absent | exclude | R1 | the reference inlines the string contents of constants the commit deleted; the text is nowhere in the prompt |
| junit4 2db6394 `run`, M1J2 and M2J2 | tokens absent | exclude | R1 | target t2 needs two new classes absent from the prompt; the cells with t1 only are unflagged, and R4 then keeps `failAfter50Ms` for the unit |
| zxing 3909ebe `verifyMinimalEncoding` (4 cells) | expectation change | exclude | R2 | the demonstration and target references change the asserted encoding strings; the commit bundles an encoder change |
| commons-lang c95ee37 `secure` (4 cells) | tokens absent | keep | R1, derivable | the demonstrated renaming `nextBytes` to `secure().randomBytes` makes `secure().randomDouble` inferable |
| gson 810e356 `isTestingProGuard` (2 cells) | tokens absent | keep | R1, standard API | `assumeFalse` is the JUnit counterpart of `assumeTrue` in the old text |
| hertzbeat 477af09 `genericPollDataFunction` (4 cells) | tokens absent | keep | R1, derivable | the missing name follows the demonstrated field-naming pattern; the old name is an acceptable alternative |
| junit4 2adef6f `getInjectionType`, M1J2 | tokens absent | keep | R1, non-essential | the developer's switch with a default that throws is optional; an if/else rewrite preserves behaviour |
| junit4 2db6394 `failAfter50Ms`, M1J2 and M2J2 | tokens absent | keep | R1, derivable | the message literal is the old one with 100 replaced by 50, as the helper's name says |

The `reason` column of `exclusion_ledger.csv` carries the same decisions per episode.

**The ledgers.** `exclusion_ledger.csv` has one row for every candidate that is not in a population (columns
`episode_id, cell, commit10, helper, manifest, reason`). The 44 candidates outside A are `dedupe_gateA` (21, dropped
by the deduplication made at the census) or `flag:<flags>` (23, the flagged candidates). The 31 candidates outside B
are `excluded:R1` or `excluded:R2` followed by the reason (10) or `dropped:R4 same-file unit kept <helper>` (21). So
the 158 candidates of `episode_cells.csv` are the 127 episodes of `episodes.jsonl` and the 31 rows of the B part of
the ledger. `reference_validity.csv` has the census flags, the verdict (`keep` or `exclude`), the rule and the reason
for each candidate, and `episode_cells.csv` has the population flags, the exposure, and the sites and oracle lengths
as text.

## 3. What an episode record holds

`episodes.jsonl` keeps the fields of the authors' manifest unchanged, and adds the text the canvas needs.

- Identity: `episode_id` (`<cell>_<repository prefix>_<commit7>_<helper>`), `gateA_episode_id` (the id in the census, which
  truncated the helper name; two gson helpers of one commit collided there, hence the new ids), `cell`, `M`, `J`,
  `source_group_id`, `repository`, `commit` and `commit10`, `file`, `unit` (commit and file), `helper`.
- Sites: `demo_sites` and `target_sites` (`<method>@<line>` ids), `rotation`, `n_sites_G4`; `demos` (the accepted
  edits: `span_old`, `span_new`, `j1:j2` the changed lines in the new method) and `targets_visible` (the same without
  `span_new`); `reference.targets` (the reference replacement `span_new` of each target, for grading only).
- Added for the canvas: `helper_text` (the helper, with its leading comments, as in the after-file) and for every site
  `method_before` (the enclosing method before the commit) with `i1`, `i2`, the line range of `span_old` in it. The
  lines before and after the changed region are the same in the old and the new method, so the new method is
  not stored.
- Lengths: `oracle_lengths` (LLaDA-1.5 tokens of each reference replacement), `old_lengths`, `demo_new_lengths`,
  `oracle_canvas_tokens`, `compact_tokens`.
- Hashes: `prompt_sha_oracle` and `prompt_sha_compact` (the first 12 hex digits of the SHA-1 of the JSON list of token
  ids of the canvas with oracle lengths, and of the canvas with one mask per target), `helper_text_sha`, `renderer_sha`
  (of the renderer the authors used), `prompt_sha_recomputed_ok`, `compact_invariant_to_gold` (the compact prompt does not
  change when the reference is replaced).
- Flags: `flags_gateA`, `adjudication` (verdict, rule, reason), `n_overloads`, `rendered_overload_not_called`,
  `gold_visible_in_helper_or_demo` (a target's reference text occurs, up to whitespace, in the helper or in a
  demonstration), `literal_copy_of_demo` (a target's reference equals a demonstration's, up to whitespace),
  `other_accepted_sites_visible` (other sites of the group whose rewritten text appears in the prompt),
  `test_code` (the file is under a test directory), `in_manifest_A`, `in_manifest_B`, `inclusion_reason_B`.
- Exposure: `exposure` and `exposure_strict_llada_dev` record whether the commit had appeared in earlier work of the
  authors: `LLADA_DEV` (used while developing the method on LLaDA-1.5; `+CROSS_MODEL` if also used with other models),
  `OTHER_MODEL_PRIOR` (appeared in earlier evaluations of other models), `INSPECTED_ONLY` (its text was read,
  no model output seen), `NO_KNOWN_OUTCOME`. In B, 76 episodes are `LLADA_DEV` (44) or `LLADA_DEV+CROSS_MODEL` (32),
  25 are `OTHER_MODEL_PRIOR` and 26 are `INSPECTED_ONLY`; `exposure_strict_llada_dev` is true for the first 76.

## 4. The canvas

`canvas.build_java_canvas(adapter, episode, M, J, lengths)` writes the prompt as one token sequence, in the chat frame
of the model:

```
<chat header + instruction>
// <repository> <commit10> <file>
// helper, already present in the file:
<helper_text>

// --- <File.java> : <method> ---          (one block per demonstration)
<old>
<up to 20 lines of the method before the changed lines, behind "    // ..." if cut>
<span_old>
<up to 12 lines after, before "    // ..." if cut>
</old>
<new>
<the same lines before> <span_new> <the same lines after>
</new>

// --- <File.java> : <method> ---          (one block per target)
<old> ... </old>
<new> <the lines before> <lengths[j] mask tokens> <the lines after> </new>
<end of the assistant turn>
```

`sites[j]` are the mask positions of target `j`; `E` are the positions of the demonstrations' `span_new` tokens (the
accepted edits whose log-likelihood the paper's `P0` hides); `gold` are the reference replacements, which never enter
`ids`. `lengths` is the number of mask tokens given to each target: the *oracle* regime of the paper passes
`oracle_lengths`, the *learned* regime passes lengths drawn from the length head. With `M = 0` the canvas has no
demonstration block (the zero-demonstration ablation of Appendix C, which uses `JAVA_ZERO_DEMO_INSTRUCTION` for all
`M`).

**Checks.** With the LLaDA-1.5 tokenizer, `build_java_canvas` reproduces, for all 127 episodes, the recorded
`prompt_sha_oracle` and `prompt_sha_compact` and the token counts (`tests/test_java_data.py`, skipped when the
tokenizer is not in the local cache). The 127 compact prompts (the input of the length head) have the same token ids
and mask positions as the prompt file the authors recorded (not shipped).
