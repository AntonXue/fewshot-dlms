# LLM coding-judge rubric for helper-propagation edits (revision r1, locked after one calibration pass)

## What the judge receives (sanitized context packet)
1. The **introduced helper** declaration as it appears in the accepted file (signature, Javadoc, body).
2. The **M accepted demonstrations**: for each, the enclosing method with its `<old>` body region and the accepted `<new>` region.
3. For each of the **J targets** (labelled Target 1, Target 2): the enclosing method's *old* text with the region to be rewritten marked `<old>…</old>`, followed by the **candidate** `<new>…</new>` region exactly as decoded, and up to 20 lines before / 12 after the region (the same cropping the generator saw).
4. Nothing else: no developer reference for the targets, no other sites of the file, no file after the commit, no generation metadata (method, β, allocation condition, rank, log-probabilities, token counts). Candidates with identical (packet, candidate) text are judged once (cache key = sha256 of the packet text). Target presentation order is randomized per packet; the judge never sees more than one candidate for the same episode in one call.

## Task statement given to the judge
"A developer introduced the helper shown and rewrote the demonstrated method(s) to use it. For each target, decide whether the candidate `<new>` region correctly applies the *same* refactoring to that method: the helper (or the demonstrated convention) is used appropriately for the target's own variables, arguments, receiver and control flow, and the method's observable behavior is plausibly preserved. The candidate does not need to match any particular wording; a different but equivalent rewrite is correct. Judge each target independently, then give the joint verdict."

## Verdict schema (strict JSON)
```
{"targets": [{"label": "Target 1", "verdict": "correct|incorrect|uncertain", "failure_tag": "none|wrong_helper_or_receiver|wrong_argument|changed_semantics_or_control_flow|missing_context|syntax_or_type_error|incomplete_or_unchanged|other", "reason": "<= 40 words"}],
 "joint": "correct|incorrect|uncertain", "instance_ill_posed": false|true, "ill_posed_reason": "<= 30 words or empty"}
```
- `joint` = correct only if every target is correct; incorrect if any target is incorrect; otherwise uncertain.
- `instance_ill_posed` = the packet itself does not allow a decision for *any* candidate (e.g. the target depends on a declaration the file no longer contains, or the intended behavior cannot be inferred from the helper and demonstrations). This is a property of the instance, recorded separately from candidate verdicts.
- `uncertain` is allowed and is reported separately; it must not be used to avoid a clear call.

## Decision guidance
- `incorrect` when: the old code survives unchanged; the helper is not used where the demonstration shows it should be (or a same-named static/instance method is used through the wrong receiver, e.g. `secure().nextLong(...)` where the convention is `secure().randomLong(...)`); arguments are swapped, missing or refer to identifiers that do not exist in the target's method; a condition's polarity or a branch's shape changes meaning; the region would not compile in context (unbalanced braces, undeclared identifiers, wrong types); a partial edit leaves one target untouched.
- `correct` when: the helper is called with the target's own arguments and receiver, the control flow is preserved or transformed exactly as the demonstration transforms it, and remaining statements are consistent. Equivalent alternatives (different variable names, extra whitespace, an equivalent expression, a different but valid overload) are correct.
- `uncertain` when the decision depends on information outside the packet (e.g. the helper's behavior on an edge case) and both outcomes are plausible.

## Recorded per judgment
`judge_model`, `rubric_revision`, `packet_sha256`, `candidate_sha256`, `episode_id` (source id only; no condition), raw response, parsed JSON, parse status, adjudication note (calibration only).

## Known limitations (to be reported with results)
The judge is a fallible instrument: it cannot execute code, may accept well-formed but subtly wrong arguments, and may reject unusual but valid rewrites. Calibration anchors are historical build/test outcomes (weak anchors: tests can pass wrong code and fail correct code) and hand-constructed negatives.
