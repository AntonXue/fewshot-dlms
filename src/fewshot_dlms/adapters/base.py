# ported from editdiff/experiments/multimodel/mm_adapters.py, experiments/multimodel/mm_common.py (incl. ar_sample), experiments/llada_pilot/lp_common.py, experiments/preflight/pf_common.py::ar_score_prefix and experiments/algo_investigation/dr_worker.py (incl. free_steps) @ d276fa6
"""Fixed-order decoding (Lemma 3.1 of the paper): sample and score complete infills of a canvas.

A canvas z has masked target positions T (`canvas.sites`). Fix an order sigma of T. An infill u puts
one token at every position of T, and

    P(u | z) = prod_s P_DLM(u[sigma_s] | z^(s-1)),

where z^(s-1) is the canvas with the first s - 1 tokens of u written in and every later target
position still masked. Lemma 3.1: this is a normalised distribution over complete infills. Drawing
one token per step from these factors samples from it exactly; evaluating the same factors at a
given u scores it. The loop that does both is `reveal`.

Two contexts, one reveal order:

    P1(u) = P(u | z)       the canvas as is (demonstration outputs visible)
    P0(u) = P(u | z \\ E)   the demonstration-output positions E masked in place

R = log P1 - log P0 is the example reward (scoring.py). Infills are sampled from P1; P0 only scores.

The reveal order is the authors': all target positions in ascending canvas position, one token per
step. The sites are contiguous and stored in prompt order, so this is site 1 left to right, then
site 2 left to right, and so on. A token revealed at one site is visible to every later prediction,
at its own site and at the others.

Each factor is the model's distribution at the position being revealed: an fp32 log-softmax over a
support, by default `adapter.support("ordinary")` (every real token except the mask, special tokens
and chat markers). The model files (llada.py, nemotron.py, dqwen.py) load a pinned checkpoint and
supply `Adapter.logits` and the chat framing; the loop itself is model-independent.

Causal models (Figure 4 only; Appendix B, Experimental Setup): Nemotron's causal suffix and Qwen3-8B continue a prompt
(canvas.ar_prompt_ids) left to right, one token per step, with the same fp32 log-softmax over a support
(`decode_ar`, `sample_ar`, `logprob_ar`). The suffix has the fixed oracle length; Qwen3's free-length
continuation stops at its first stop token. Their P0 prompt deletes the demonstration outputs rather than
masking them, so R = log P1 - log P0 compares the same continuation under two prompts.
"""
from dataclasses import dataclass
from dataclasses import field
import re
import time

import torch


@dataclass
class Infill:
    """One sampled infill of a canvas."""

    toks: list[list[int]]  # the tokens of each target site, in the order of canvas.sites
    ell1_sample: float  # log P1 of the infill, accumulated while sampling (the pools' ell1)
    ell1_tokens: list[float] = field(default_factory=list)  # the terms of ell1_sample, in reveal order


# ---------------------------------------------------------------------------------------------
# Sampling and scoring a canvas
# ---------------------------------------------------------------------------------------------


def sample_fixed_order(adapter, canvas, K, seed, support="ordinary", chunk=16):
    """Draw K infills from P1 by fixed-order decoding: by Lemma 3.1, exact samples of P1.

    One torch.Generator, seeded with `seed`, serves all K draws, so the seed fixes the draws for given
    logits. `support` is a kind for `adapter.support` or a boolean mask over the vocabulary. `chunk`
    is the most canvases per forward pass. Returns the K infills in draw order; `ell1_sample` and
    `ell1_tokens` are the pools' ell1 and its per-token terms (`site_sums` turns these into ell1_t).
    """
    generator = torch.Generator().manual_seed(seed)
    traces, logprobs = reveal(
        adapter,
        [context_ids(adapter, canvas, hide_E=False)],
        [reveal_order(canvas)],
        which=[0] * K,
        allowed=_allowed(adapter, support),
        generator=generator,
        chunk=chunk,
    )
    infills = []
    for trace, terms in zip(traces, logprobs):
        toks = split_by_site(canvas, trace)
        infills.append(Infill(toks=toks, ell1_sample=float(sum(terms)), ell1_tokens=terms))
    return infills


def logprob_fixed_order(adapter, canvas, toks, hide_E, chunk=16, support="ordinary"):
    """Score one infill with the same factors, its tokens written in (teacher-forced) in the same order.

    hide_E=False gives log P1 (the canvas as is), hide_E=True gives log P0 (canvas.E masked). `toks`
    is either per site, as in Infill.toks, or one flat list in reveal order, as stored in the frozen
    pools. Returns the total and the per-token terms in reveal order. With hide_E=True this is how the
    pools' ell0 was computed; with hide_E=False it replays their ell1, which was summed at sample
    time (equal up to GPU numerics). `site_sums` of the terms gives ell1_t / ell0_t.
    """
    trace = _in_reveal_order(canvas, toks)
    _, logprobs = reveal(
        adapter,
        [context_ids(adapter, canvas, hide_E)],
        [reveal_order(canvas)],
        which=[0],
        allowed=_allowed(adapter, support),
        forced=[trace],
        chunk=chunk,
    )
    terms = logprobs[0]
    return float(sum(terms)), terms


def reveal_order(canvas):
    """The fixed order sigma: every target position of the canvas, ascending."""
    return sorted(position for site in canvas.sites for position in site)


def context_ids(adapter, canvas, hide_E):
    """The canvas before the first reveal: z for P1, or z \\ E for P0 (every position of E masked)."""
    ids = list(canvas.ids)
    if hide_E:
        for position in canvas.E:
            ids[position] = adapter.mask_id
    return ids


def split_by_site(canvas, values):
    """Regroup values given in reveal order (tokens or log-prob terms) into one list per target site."""
    value_at = dict(zip(reveal_order(canvas), values))
    return [[value_at[position] for position in site] for site in canvas.sites]


def site_sums(canvas, terms):
    """Per-site sums of per-token log-prob terms: the pools' ell1_t and ell0_t (one value per target)."""
    return [float(sum(site_terms)) for site_terms in split_by_site(canvas, terms)]


# ---------------------------------------------------------------------------------------------
# The loop of Lemma 3.1
# ---------------------------------------------------------------------------------------------


def reveal(adapter, contexts, orders, which, allowed, generator=None, forced=None, chunk=16):
    """Fixed-order decoding of n = len(which) candidates at once (sampling or scoring).

    Candidate k is revealed on the token sequence contexts[ctx], ctx = which[k], in the order
    orders[ctx]. There is one context in Figures 4, 5 and 7; Section 4.3 gives each target-length
    allocation its own. At step s, candidate k's factor is the log-softmax over `allowed` (a boolean
    mask over the vocabulary, or None for all tokens) at position orders[ctx][s] of contexts[ctx]
    with the candidate's first s tokens written at orders[ctx][:s].

    Before the first forward pass the adapter is told the reveal order, once:
    adapter.begin_reveal(orders[0]). With several contexts the first target position is the same in
    all of them (only the target lengths differ), so orders[0] stands for every context.

    Sampling (forced=None): every candidate draws its next token from its factor, with `generator` (a
    torch.Generator; required, so that no draw comes from torch's global RNG: ValueError without it).
    Scoring (forced = n token lists in reveal order): the given tokens are written in instead.
    Returns the n token lists and, for each, its log-prob terms in reveal order. Their sum is log P1
    or log P0 of the candidate, depending on the context.

    Candidates on the same context that agree on every token so far see the same canvas, which is
    run once. The draws a seed reproduces come in this order: step by step; within a step, one
    multinomial call per distinct canvas, in the order of the first candidate on it; each call draws
    one token per candidate on that canvas, handed out in candidate order.
    """
    n = len(which)
    sampling = forced is None
    if sampling and generator is None:
        raise ValueError("sampling needs a seeded torch.Generator (generator=...); pass forced=... to score")
    traces = [[] for _ in range(n)] if sampling else [list(trace) for trace in forced]
    logprobs = [[] for _ in range(n)]
    adapter.begin_reveal(orders[0])
    for step in range(max(len(orders[ctx]) for ctx in set(which))):
        groups = {}
        for k in range(n):
            ctx = which[k]
            if step < len(orders[ctx]):
                groups.setdefault((ctx, tuple(traces[k][:step])), []).append(k)
        keys = list(groups)
        step_logprobs = _factor_logprobs(adapter, contexts, orders, keys, step, allowed, chunk)
        for row, key in enumerate(keys):
            members = groups[key]
            factor = step_logprobs[row]
            if sampling:
                probs = factor.exp()
                probs = torch.where(torch.isfinite(factor), probs, torch.zeros_like(probs)).cpu()
                draws = torch.multinomial(probs, len(members), replacement=True, generator=generator).tolist()
                for k, token in zip(members, draws):
                    traces[k].append(int(token))
                    logprobs[k].append(float(factor[token]))
            else:
                for k in members:
                    logprobs[k].append(float(factor[traces[k][step]]))
    return traces, logprobs


def _factor_logprobs(adapter, contexts, orders, keys, step, allowed, chunk):
    """The factor at `step` for each distinct canvas keys[i] = (context index, tokens revealed so far).

    Canvases of equal length are batched, `chunk` at a time, in the order of `keys`. Returns one row
    of fp32 log-probs per key.
    """
    by_length = {}
    for i, (ctx, _) in enumerate(keys):
        by_length.setdefault(len(contexts[ctx]), []).append(i)
    rows = [None] * len(keys)
    for indices in by_length.values():
        for start in range(0, len(indices), chunk):
            batch = indices[start:start + chunk]
            canvases = []
            positions = []
            for i in batch:
                ctx, revealed = keys[i]
                canvases.append(_fill(contexts[ctx], orders[ctx][:step], revealed))
                positions.append([orders[ctx][step]])
            logits = adapter.counted_logits(torch.tensor(canvases, dtype=torch.long), positions)
            for j, i in enumerate(batch):
                rows[i] = logits[j, 0]
    logits = torch.stack(rows, 0).float()
    if allowed is not None:
        logits = logits.masked_fill(~allowed.to(logits.device), float("-inf"))
    return torch.log_softmax(logits, -1)


def _fill(ids, positions, tokens):
    """A copy of ids with tokens[i] written at positions[i]: the canvas z^(s) after s reveals."""
    canvas = list(ids)
    for position, token in zip(positions, tokens):
        canvas[position] = token
    return canvas


# ---------------------------------------------------------------------------------------------
# Causal models: left-to-right decoding of a prompt's continuation (Figure 4)
# ---------------------------------------------------------------------------------------------


@dataclass
class ARInfill(Infill):
    """One sampled continuation of a causal model's prompt: one site, its tokens in generation order."""

    truncated: bool = False  # stopped by the token budget before any stop token (free-length decoding only)


def sample_ar(adapter, prompt_ids, K, seed, max_new, stop_ids, support=None, chunk=None):
    """Draw K continuations of the prompt from P1, token by token at temperature 1.

    A continuation ends at its first token in `stop_ids`, which it keeps (Qwen3-8B: <|im_end|> or any token
    whose text holds a newline), or after `max_new` tokens. With no stop ids every continuation has exactly
    `max_new` tokens: Nemotron's causal suffix of the oracle length, which is never truncated. One
    torch.Generator, seeded with `seed`, serves all K draws. `support` is a kind for `adapter.support` or a boolean
    mask over the vocabulary; by default the adapter's `default_support` (Qwen3-8B: "free", which admits
    <|im_end|>). `chunk` is the most sequences per forward pass; by default the adapter's `sample_chunk`, the
    frozen pools' (with both defaults, the same seed draws the stored continuations again on the authors'
    platform).

    Returns the K continuations in draw order: toks = [tokens], ell1_sample = log P1 of the tokens (the
    pools' ell1), ell1_tokens its terms, truncated = ended by the budget while stop ids were in force.
    """
    generator = torch.Generator().manual_seed(seed)
    traces, logprobs, truncated = decode_ar(
        adapter,
        prompt_ids,
        K,
        support,
        generator=generator,
        max_new=max_new,
        stop_ids=stop_ids,
        chunk=chunk,
    )
    infills = []
    for trace, terms, cut in zip(traces, logprobs, truncated):
        infills.append(ARInfill(toks=[trace], ell1_sample=float(sum(terms)), ell1_tokens=terms, truncated=cut))
    return infills


def logprob_ar(adapter, prompt_ids, toks, hide_E_prompt_ids=None, support=None):
    """Score one continuation with the factors it would be sampled from, its tokens written in (teacher forcing).

    Without hide_E_prompt_ids this is log P1 under `prompt_ids`, which replays the pools' ell1. With
    hide_E_prompt_ids, the P0 prompt (canvas.ar_prompt_ids(..., hide_E=True): the demonstration outputs
    deleted), it is log P0 under that prompt, the pools' ell0; the P0 prompt must be `prompt_ids` with tokens
    deleted, else ValueError. `toks` is ARInfill.toks or one flat list. `support` defaults to the adapter's
    `default_support`, the support the continuation was sampled over. Returns the total and the per-token terms.

    One continuation per forward pass: equal to the frozen values up to bf16 batch noise. The frozen pools
    scored all distinct draws of an episode together; `decode_ar(..., forced=...)` with the distinct draws in
    first-occurrence order forms those batches again.
    """
    trace = _flat_continuation(toks)
    context = list(prompt_ids)
    if hide_E_prompt_ids is not None:
        if not _is_deletion(hide_E_prompt_ids, prompt_ids):
            raise ValueError("hide_E_prompt_ids must be prompt_ids with some tokens deleted (P0 deletes E)")
        context = list(hide_E_prompt_ids)
    _, logprobs, _ = decode_ar(adapter, context, 1, support, forced=[trace])
    terms = logprobs[0]
    return float(sum(terms)), terms


def decode_ar(
    adapter, prompt_ids, n, support=None, generator=None, forced=None, max_new=None, stop_ids=frozenset(), chunk=None
):
    """Left-to-right decoding of n continuations of one prompt at once (sampling or scoring).

    At step s, continuation k's factor is the log-softmax over the support (a kind for `adapter.support` or a
    boolean mask over the vocabulary; by default the adapter's `default_support`) of the model's logits for the
    next token after prompt_ids + its first s tokens. `chunk` defaults to the adapter's `sample_chunk`; the frozen
    pools scored P0 with `adapter.p0_chunk`.

    Sampling (forced=None): every unfinished continuation draws its next token from its factor with `generator` (a
    torch.Generator, required: ValueError without it); it finishes on a token in `stop_ids` (kept) or once it has
    `max_new` tokens. Scoring (forced = n token lists): the given tokens
    are written in instead, and a continuation takes part in a step while it has a token for it.
    Returns the n token lists, their log-prob terms, and for each whether the budget ended it before a stop token
    (always False without stop ids, where max_new is the fixed length).

    Continuations that agree on every token so far form the same sequence, which is run once; sequences go to the
    model `chunk` at a time, in the order of their first continuation. The draws a seed reproduces: step by step;
    within a step, one multinomial call per distinct sequence in that order; each call draws one token per
    continuation on it, handed out in continuation order. Scoring the distinct draws of a sampling call, in
    first-occurrence order, repeats that call's batches.
    """
    if adapter.kind != "ar":
        raise ValueError(f"{adapter.name!r} fills canvases; left-to-right decoding needs a causal interface")
    allowed = _allowed(adapter, adapter.default_support if support is None else support)
    chunk = adapter.sample_chunk if chunk is None else chunk
    sampling = forced is None
    if sampling and (max_new is None or max_new < 1):
        raise ValueError(f"sampling needs a token budget max_new >= 1, got {max_new}")
    if sampling and generator is None:
        raise ValueError("sampling needs a seeded torch.Generator (generator=...); pass forced=... to score")
    if not sampling and len(forced) != n:
        raise ValueError(f"expected {n} forced continuations, got {len(forced)}")
    traces = [[] for _ in range(n)] if sampling else [[int(token) for token in trace] for trace in forced]
    logprobs = [[] for _ in range(n)]
    done = [False] * n
    truncated = [False] * n
    step = 0
    while True:
        if sampling:
            active = [k for k in range(n) if not done[k]]
        else:
            active = [k for k in range(n) if len(traces[k]) > step]
        if not active:
            break
        groups = {}
        for k in active:
            groups.setdefault(tuple(traces[k][:step]), []).append(k)
        keys = list(groups)
        step_logprobs = _next_token_logprobs(adapter, prompt_ids, keys, allowed, chunk)
        for row, key in enumerate(keys):
            members = groups[key]
            factor = step_logprobs[row]
            if sampling:
                probs = factor.exp()
                probs = torch.where(torch.isfinite(factor), probs, torch.zeros_like(probs)).cpu()
                draws = torch.multinomial(probs, len(members), replacement=True, generator=generator).tolist()
                for k, token in zip(members, draws):
                    traces[k].append(int(token))
                    logprobs[k].append(float(factor[token]))
                    if token in stop_ids:
                        done[k] = True
                    elif len(traces[k]) >= max_new:
                        done[k] = True
                        truncated[k] = bool(stop_ids)
            else:
                for k in members:
                    logprobs[k].append(float(factor[traces[k][step]]))
        step += 1
    return traces, logprobs, truncated


def _next_token_logprobs(adapter, prompt_ids, keys, allowed, chunk):
    """The factor of the next token after prompt_ids + keys[i], for each distinct continuation so far.

    All keys of a step have the same length, so the sequences have equal length. They are run `chunk` at a time,
    in the order of `keys`; the model is asked for the token at position len(sequence), the next one. Returns
    one row of fp32 log-probs per key.
    """
    sequences = [list(prompt_ids) + list(key) for key in keys]
    rows = []
    for start in range(0, len(sequences), chunk):
        batch = sequences[start:start + chunk]
        positions = [[len(sequence)] for sequence in batch]
        logits = adapter.counted_logits(torch.tensor(batch, dtype=torch.long), positions)
        rows.append(logits[:, 0])
    logits = torch.cat(rows, 0).float()
    if allowed is not None:
        logits = logits.masked_fill(~allowed.to(logits.device), float("-inf"))
    return torch.log_softmax(logits, -1)


def _flat_continuation(toks):
    """A continuation as one token list, from ARInfill.toks ([tokens]) or the flat list itself."""
    if toks and all(isinstance(site_toks, (list, tuple)) for site_toks in toks):
        if len(toks) != 1:
            raise ValueError(f"a causal continuation has one site, got {len(toks)}")
        return [int(token) for token in toks[0]]
    return [int(token) for token in toks]


def _is_deletion(shorter, longer):
    """True when `shorter` is `longer` with zero or more tokens deleted (an order-preserving subsequence)."""
    remaining = iter(longer)
    return len(shorter) <= len(longer) and all(any(token == other for other in remaining) for token in shorter)


# ---------------------------------------------------------------------------------------------
# Argument checks
# ---------------------------------------------------------------------------------------------


def _in_reveal_order(canvas, toks):
    """An infill as one token list in reveal order, from per-site lists or a flat list in that order."""
    order = reveal_order(canvas)
    if all(isinstance(site_toks, (list, tuple)) for site_toks in toks):
        lengths = [len(site_toks) for site_toks in toks]
        expected = [len(site) for site in canvas.sites]
        if lengths != expected:
            raise ValueError(f"expected per-site token lists of lengths {expected}, got {lengths}")
        token_at = {}
        for site, site_toks in zip(canvas.sites, toks):
            for position, token in zip(site, site_toks):
                token_at[position] = int(token)
        return [token_at[position] for position in order]
    if len(toks) != len(order):
        raise ValueError(f"expected {len(order)} tokens in reveal order, got {len(toks)}")
    return [int(token) for token in toks]


def _allowed(adapter, support):
    """The support as a boolean mask over the vocabulary: a kind for adapter.support, a mask, or None."""
    if isinstance(support, str):
        return adapter.support(support)
    return support


# ---------------------------------------------------------------------------------------------
# The model interface
# ---------------------------------------------------------------------------------------------


class Adapter:
    """A model as the reveal loop sees it: tokenizer, chat framing, support and logits.

    A subclass loads a pinned checkpoint (`model_id` at `revision`) and implements `logits` and the
    four chat methods. A subclass that keeps its Hugging Face tokenizer in `tok` and sets `V` and
    `control_ids` gets `encode`, `decode`, `marker_ids` and `support` from here.

    The loop calls `begin_reveal(order)` once per reveal, before its first `logits` call. It does
    nothing here; an adapter whose attention depends on where the targets start overrides it.
    """

    name: str = ""  # short name in records and file names, e.g. "llada", "nemotron_pt", "dqwen"
    model_id: str = ""  # Hugging Face model id
    revision: str = ""  # pinned commit of the model repository
    mask_id: int = -1  # the mask token; -1 for a left-to-right model, which has none
    kind = None  # "dlm" (fills a masked canvas) or "ar" (generates left to right)
    close_slot = False  # one-target canvas: end the turn right after the masks (the authors' slot rule)

    # Left-to-right decoding only (sample_ar, logprob_ar, decode_ar), when the call names no support or chunk: the
    # support kind of every step, and the sequences per forward pass while sampling (P1 is replayed in the same
    # batches) and while scoring P0. The causal adapters set the values their Figure 4 pools were made with.
    default_support = "ordinary"
    sample_chunk = 16
    p0_chunk = 16

    tok = None  # Hugging Face tokenizer
    V = 0  # size of the logit dimension (can exceed the tokenizer's vocabulary)
    control_ids = frozenset()  # ids that are never content: mask, end of text, padding, special tokens
    code_ids = None  # classification: letter -> id of the single token " <letter>"

    # Cost accounting, read with cost(): forward passes, sequences, tokens, seconds in the forward pass.
    n_forward = 0
    n_seq = 0
    n_tok = 0
    forward_s = 0.0

    def encode(self, text):
        """The token ids of a piece of text, without special tokens."""
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids):
        """The text of a list of token ids."""
        return self.tok.decode(ids)

    def chat_header(self, instruction):
        """Everything before the first record: the chat template with the instruction in the user turn."""
        raise NotImplementedError

    def chat_user_open(self):
        """Opens the user turn."""
        raise NotImplementedError

    def chat_user_close_assistant_open(self):
        """Closes the user turn and opens the assistant turn."""
        raise NotImplementedError

    def chat_end(self):
        """Ends the assistant turn."""
        raise NotImplementedError

    def begin_reveal(self, order):
        """Hook called once before a reveal's first forward pass, with its reveal order. No-op here.

        `order` lists the target positions in reveal order, so order[0] is the first target position.
        Nemotron in prefix/tail mode takes order[0] as the start of its bidirectional block and keeps
        it for every step of the reveal, under P1 and P0 alike.
        """
        return None

    def logits(self, batch_ids, positions):
        """The model's logits at chosen positions of each canvas: the one model-specific piece.

        batch_ids  LongTensor [B, T] of B canvases of equal length (on CPU; move it to the model)
        positions  for each canvas, the positions to read (the reveal loop reads one)
        returns    Tensor [B, P, V]: [b, i] are the logits for the token at positions[b][i]. Any shift
                   between input and output index is the adapter's business.
        """
        raise NotImplementedError

    def counted_logits(self, batch_ids, positions):
        """`logits`, counted in cost()."""
        start = time.monotonic()
        logits = self.logits(batch_ids, positions)
        if logits.is_cuda:
            torch.cuda.synchronize(logits.device)  # so that forward_s measures the forward pass
        self._count(batch_ids, start)
        return logits

    def _count(self, batch_ids, start):
        """Record one forward pass over the canvases of batch_ids that began at time `start`."""
        self.forward_s += time.monotonic() - start
        self.n_forward += 1
        self.n_seq += len(batch_ids)
        self.n_tok += sum(len(ids) for ids in batch_ids)

    def cost(self):
        """Forward passes, canvases and tokens fed, and seconds in the forward pass, so far."""
        return {
            "forwards": self.n_forward,
            "sequences": self.n_seq,
            "tokens": self.n_tok,
            "forward_s": self.forward_s,
        }

    def marker_ids(self):
        """Token ids that are special tokens or chat markers, so never content (computed once).

        A token is a marker if the tokenizer lists it as special or added, if it does not decode, if it
        decodes to the replacement character, or if its text is a tag such as <|im_end|>, <think> or
        </answer>.
        """
        if not hasattr(self, "_markers"):
            ids = set(i for i in self.tok.all_special_ids if i is not None)
            tag = re.compile(r"^\s*(<\|[^\s<>]{1,40}\|>|</?(role|tool_call|tool_response|think|answer)>)\s*$")
            for i in range(len(self.tok)):
                try:
                    text = self.tok.decode([i])
                except Exception:
                    ids.add(i)
                    continue
                if tag.match(text) or text == "�":
                    ids.add(i)
            for i in getattr(self.tok, "added_tokens_decoder", {}):
                ids.add(int(i))
            self._markers = ids
        return self._markers

    def support(self, kind, codes=None):
        """Which token ids may fill a target position, as a boolean mask over the V logits.

        "codes": exactly the ids in `codes` (single-token classification codes).
        Any other kind ("ordinary"): every id of the tokenizer except control_ids and marker_ids;
        ids beyond the tokenizer (padding rows of the output layer) are excluded too.
        """
        mask = torch.zeros(self.V, dtype=torch.bool)
        if kind == "codes":
            mask[list(codes)] = True
        else:
            mask[:len(self.tok)] = True
            mask[sorted(self.control_ids | self.marker_ids())] = False
        return mask

    def support_audit(self):
        """What the ordinary support leaves out, for a run's records."""
        allowed = self.support("ordinary")
        excluded = sorted(set(range(self.V)) - set(allowed.nonzero().flatten().tolist()))
        real = [i for i in excluded if i < len(self.tok)]
        return {
            "V_logits": self.V,
            "n_tokenizer": len(self.tok),
            "n_ordinary": int(allowed.sum()),
            "n_excluded_total": len(excluded),
            "n_excluded_beyond_tokenizer": len(excluded) - len(real),
            "excluded_real_ids": real,
            "excluded_real_tokens": [self.tok.decode([i]) for i in real][:400],
            "control_ids": sorted(self.control_ids),
            "code_ids": self.code_ids,
        }
