# ported from editdiff/experiments/length_project/lpx_alloc.py, experiments/length_project/lpx_ds.py (length_mass), experiments/length_project/lpx_model.py::{StopForward, Trunk, Branch, site_metrics}, experiments/length_pilot/lp_model.py::{rot_half, block_forward, Packed, Trunk, Branch, load_branch, predict} and experiments/length_pilot/lp_common.py::{collapse_window, L_MAX} @ d276fa6
"""Target lengths (Section 3.3, Section 4.3, Appendix B, Length Allocation): allocation modes and the learned
length head.

A masked target needs a token count before the model fills it: a canvas with L_j masks at target j can only
hold an answer of exactly L_j tokens. `allocate` gives the J target lengths of one candidate in one of three
modes:

    oracle   L_j = L*_j, the token count of the reference answer (Sections 4.1 and 4.2)
    offset   L_j = L*_j + delta, clipped to 1..lmax (the controlled offsets of Section 4.3)
    learned  L_j drawn from the length head below (Section 4.3; the Java head C of Section 5)

The length head (Appendix B, Length Allocation)
-----------------------------------------------
A frozen DLM reads a compact prompt z~ that holds one placeholder per target site. The hidden state that
enters the backbone's last transformer block feeds a trainable copy of that block and of the final
normalisation (LLaDA-1.5: block 31 of 32 and ln_f; Nemotron: layer 33 of 34 and encoder.norm; dQwen3.5:
layer 31 of 32 and model.norm); a linear categorical head reads the normalised state at each placeholder and
gives a distribution over the lengths 1..256 (index c <-> length c + 1). Sites are predicted independently
given z~. Only the copy and the head were trained: prompt-normalised cross-entropy against the exact span
lengths (mean over prompts of the mean site NLL), seed 0, AdamW with weight decay 0.01, learning rates 3e-5
for the copied block and norm and 3e-4 for the head, bf16 autocast with fp32 master weights, final checkpoint.
The same forward pass is used here: the copy and the head hold fp32 weights and run under bf16 autocast, and
the head's logits are fp32. On the authors' platform the ported LLaDA-1.5 head reproduces the frozen
evaluation's distributions to about 1e-6 in total variation (see LengthHead.predict_examples for the one
numerical precaution this takes).

Which canvas the head sees: `compact_prompt(canvas, mask_id)`. It is the canvas as built by canvas.py (chat
framing, instruction, every demonstration with its output visible, as in P1 -- nothing at E is masked -- and
every target input) with the run of masks of each target site replaced by ONE mask token, the placeholder;
positions are renumbered from 0. It is therefore the same for every allocation of a canvas (oracle, offset or
learned), and the reference answers never enter it. For the canvases of Sections 4.2 and 4.3 it equals
`build_canvas(adapter, family, M, J, [1] * J).ids`, which is how the authors rendered the head's prompts (the
production renderer with one mask per target); tests/test_gpu_head.py checks it token for token through the
SHA-256 of the frozen prompts. The Java head C reads the same construction applied to `build_java_canvas`
(Section 5). The head reads its distribution at the J placeholders, in site order.

The forward pass on each backbone (lpx_model.py):

    llada        Compact prompts are concatenated into one flat stream without padding (`Packed`). Every frozen
                 block except the last runs on it with the checkpoint's own submodules; only attention and RoPE
                 are rewritten (`_llada_block`): bidirectional within a prompt (flash-attn varlen, one segment
                 per prompt), none across prompts, RoPE positions restarting at 0 for every prompt. The copied
                 last block runs on the same stream.
    nemotron_pt  One prompt per forward pass, under the adapter's prefix/tail mask with the bidirectional tail
                 starting at the first placeholder. A hook captures the inputs of the last layer and stops the
                 frozen forward; the copied layer runs on them under the same mask.
    dqwen        One prompt per forward pass; the model prepends its BOS, so the placeholder at input position
                 p is read at BOS-prefixed index p (the model's own readout). The inputs of the last layer (a
                 full-attention layer) are captured the same way. Its Gated-DeltaNet kernels are autotuned
                 afresh in every process, so its predictions can differ between processes (tests/gpu_util.py).

The checkpoints are the authors' final.pt files, loaded unchanged: {"config": {..., "backbone": "llada" |
"nemotron" | "dqwen", "lmax": 256}, "branch": state dict with block.*, norm.*, head.weight [lmax, d] and
head.bias [lmax]}. The Section 4.3 heads and the Java head C (Section 5, LLaDA-1.5) share this format.

Sampling learned allocations (lpx_alloc.py)
-------------------------------------------
The head is evaluated once per compact prompt; each of the K candidates then draws its own allocation.
Candidate k, site j: L_kj = F_j^{-1}(u_kj), the inverse CDF of the site's distribution at the uniform
u_kj = SHA-256("lpx-alloc-v1" | seed | k | j) / 2^64 (`hashed_uniform`). The draws are independent across
candidates and sites, and every head draws with the same u (common random numbers across predictors). The
Section 4.3 allocations used seed = (bank, backbone, family, J) (`section43_seed`); a learned allocation never
exceeds lmax.
"""
import copy
import hashlib

import numpy as np
import torch

L_MAX = 256  # lengths are 1..L_MAX: the head's categorical support (Appendix B, Length Allocation)

MODES = ("oracle", "offset", "learned")

ALLOC_TAG = "lpx-alloc-v1"  # first part of every hashed uniform of a learned allocation (lpx_alloc.py)

# The backbone name in a head checkpoint's config, for each adapter of fewshot_dlms.adapters.
BACKBONE_OF_ADAPTER = {
    "llada": "llada",
    "nemotron_pt": "nemotron",
    "dqwen": "dqwen",
}

MAX_TOKENS = 16384  # most prompt tokens in one packed LLaDA forward pass (lp_model.predict)

# Shortest packed LLaDA stream at prediction time; a shorter pack gets a filler segment (see predict_examples).
MIN_STREAM_TOKENS = 2048
FILLER_TOKEN = 0  # token of the filler segment; it never reaches a prompt (segments do not attend to each other)


# ---------------------------------------------------------------------------------------------
# Allocation modes
# ---------------------------------------------------------------------------------------------


def allocate(mode, oracle_L, delta=0, head_sample=None, lmax=L_MAX):
    """The J target lengths of one candidate.

    oracle_L     L*_j, the token count of each reference answer (tokenized as " <output>"), in site order
    "oracle"     L_j = L*_j, the setting of Sections 4.1 and 4.2
    "offset"     L_j = max(1, min(lmax, L*_j + delta)), the controlled offsets of Section 4.3. The paper's
                 offset runs (delta in -2..+2) skipped a family when some L*_j + delta < 1 and counted it,
                 so clipping never entered the reported numbers; a run that reproduces them skips those
                 families too.
    "learned"    L_j = head_sample[j], one draw of the length head (LengthHead.sample)
    """
    if mode == "oracle":
        return [int(length) for length in oracle_L]
    if mode == "offset":
        return [max(1, min(lmax, int(length) + delta)) for length in oracle_L]
    if mode == "learned":
        if head_sample is None:
            raise ValueError("mode 'learned' needs head_sample, one length per target")
        if len(head_sample) != len(oracle_L):
            raise ValueError(f"expected {len(oracle_L)} sampled lengths (one per target), got {len(head_sample)}")
        if any(not 1 <= length <= lmax for length in head_sample):
            raise ValueError(f"sampled lengths must lie in 1..{lmax}, got {list(head_sample)}")
        return [int(length) for length in head_sample]
    raise ValueError(f"unknown allocation mode {mode!r}; expected one of {MODES}")


# ---------------------------------------------------------------------------------------------
# Sampling learned allocations
# ---------------------------------------------------------------------------------------------


def hashed_uniform(*parts):
    """A uniform number in [0, 1): the first 16 hex digits of SHA-256 of the parts joined by '|', over 2^64."""
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16) / 2**64


def inverse_cdf(probs, u):
    """The length F^{-1}(u) of a categorical over 1..len(probs): the smallest length whose CDF exceeds u.

    The probabilities are normalised first (in float64), and u is scaled by the last CDF value, as in the
    authors' sampler, so a length with zero mass is never drawn.
    """
    q = np.asarray(probs, dtype=np.float64)
    q = q / q.sum()
    cdf = np.cumsum(q)
    index = int(np.searchsorted(cdf, u * cdf[-1], side="right"))
    return min(len(q), index + 1)


def sample_allocations(probs, K, seed):
    """K allocations drawn from per-site length distributions: K lists of J lengths.

    probs  array [J, lmax]: site j's probabilities over the lengths 1..lmax, in site order
    seed   an int, or a tuple of key parts such as section43_seed(...)

    Candidate k, site j gets inverse_cdf(probs[j], u) with u = hashed_uniform(ALLOC_TAG, *seed, k, j), so the
    first K allocations of a larger K are the same, and every site and candidate has its own uniform.
    """
    seed_parts = tuple(seed) if isinstance(seed, (tuple, list)) else (seed,)
    probs = np.asarray(probs, dtype=np.float64)
    allocations = []
    for k in range(K):
        allocation = []
        for j in range(len(probs)):
            u = hashed_uniform(ALLOC_TAG, *seed_parts, k, j)
            allocation.append(inverse_cdf(probs[j], u))
        allocations.append(allocation)
    return allocations


def section43_seed(bank, backbone, episode_id, J):
    """The seed of the paper's Section 4.3 allocations: (bank, backbone, family, J).

    bank is "dev" or "report", backbone the head's name ("llada", "nemotron" or "dqwen"; LengthHead.backbone),
    episode_id the family (e.g. "list_easy_fam:report:000") and J its number of targets.
    """
    return (bank, backbone, episode_id, J)


# ---------------------------------------------------------------------------------------------
# How good a length distribution is (Section 4.3, Appendix B, Length Allocation)
# ---------------------------------------------------------------------------------------------


def length_mass(probs, lengths, radius):
    """P_r of every site: the probability mass a length distribution puts within `radius` tokens of the reference.

    probs    array [S, lmax]: site s's probabilities over the lengths 1..lmax
    lengths  array [S]: the reference length L* of each site
    Returns [S]: sum of probs[s, l - 1] over |l - L*_s| <= radius (P_0 = the mass on the exact length). Sums are taken
    over the slice of the support, in the order of the authors' analysis (lpx_ds.py), so they are bit-identical to it.
    """
    lmax = probs.shape[1]
    return np.array([probs[s, max(0, lengths[s] - 1 - radius) : min(lmax, lengths[s] + radius)].sum() for s in range(len(lengths))])


def site_metrics(probs, lengths, lmax=L_MAX):
    """Summary of per-site length distributions against the reference lengths (lpx_model.site_metrics).

    probs [S, lmax] probabilities over 1..lmax; lengths [S] reference lengths (may exceed lmax: out of support).
    Returns the site means of P_0, P_1, P_2 (mass within 0, 1, 2 tokens), the NLL of the reference (in-support sites),
    the expected absolute error E|L - L*|, how often the mode is exact, and the calibration error of the mode (ECE over
    10 confidence bins). The trainer's development evaluation (experiments/length_allocation/train_head.py) reports these.
    """
    S = len(lengths)
    support = np.arange(1, lmax + 1)
    in_support = lengths <= lmax
    clipped = np.clip(lengths, 1, lmax)

    def within(radius):
        values = []
        for s in range(S):
            if lengths[s] - radius <= lmax:
                values.append(probs[s, max(0, lengths[s] - 1 - radius) : min(lmax, lengths[s] + radius)].sum())
            else:
                values.append(0.0)
        return np.array(values)

    p0 = np.where(in_support, probs[np.arange(S), clipped - 1], 0.0)
    mode = probs.argmax(1) + 1
    confidence = probs.max(1)
    ece = 0.0
    bins = np.minimum((confidence * 10).astype(int), 9)
    for b in range(10):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(confidence[in_bin].mean() - (mode[in_bin] == lengths[in_bin]).mean())
    nll = -np.log(np.clip(p0, 1e-300, None))
    return {
        "sites": int(S),
        "out_of_support": float((~in_support).mean()),
        "P0": float(p0.mean()),
        "P1": float(within(1).mean()),
        "P2": float(within(2).mean()),
        "nll_site": float(nll[in_support].mean()) if in_support.any() else float("nan"),
        "eae": float((probs * np.abs(support[None, :] - lengths[:, None])).sum(1).mean()),
        "mode_exact": float((mode == lengths).mean()),
        "ece_mode": float(ece),
    }


# ---------------------------------------------------------------------------------------------
# The compact prompt
# ---------------------------------------------------------------------------------------------


def compact_prompt(canvas, mask_id):
    """The head's input: the canvas with each target site's masks replaced by one placeholder mask.

    Returns {"input_ids": the compact prompt, "mask_positions": the placeholder of each target, in site order}
    (the example format of the authors' code; lp_common.collapse_window). Every site must be a non-empty,
    contiguous run of mask tokens, and no mask token may sit outside the sites.
    """
    target_of_start = {}
    collapsed = set()
    for j, site in enumerate(canvas.sites):
        if not site:
            raise ValueError(f"target {j} has no positions")
        if list(site) != list(range(site[0], site[0] + len(site))):
            raise ValueError(f"target {j} is not a contiguous run of positions")
        if any(canvas.ids[position] != mask_id for position in site):
            raise ValueError(f"target {j} holds tokens other than the mask")
        target_of_start[site[0]] = j
        collapsed.update(site[1:])

    ids = []
    mask_positions = [None] * len(canvas.sites)
    for position, token in enumerate(canvas.ids):
        if position in collapsed:
            continue
        if position in target_of_start:
            mask_positions[target_of_start[position]] = len(ids)
        ids.append(int(token))
    if ids.count(mask_id) != len(canvas.sites):
        raise ValueError("the canvas has mask tokens outside its target sites")
    return {"input_ids": ids, "mask_positions": mask_positions}


# ---------------------------------------------------------------------------------------------
# LLaDA-1.5: packed variable-length forward pass (lp_model.py)
# ---------------------------------------------------------------------------------------------


def _flash_attn_varlen():
    """flash_attn_varlen_func, imported on first use: only the LLaDA-1.5 head needs flash-attn."""
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError as error:
        raise ImportError(
            "the LLaDA-1.5 length head runs the authors' packed attention, which needs flash-attn "
            "(the authors used flash-attn 2.8.3 with torch 2.7.1)"
        ) from error
    return flash_attn_varlen_func


def _rot_half(x):
    """Rotate the two halves of the last dimension: (a, b) -> (-b, a), the RoPE companion of x."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _llada_block(block, x, cu, maxlen, sin, cos, n_heads):
    """One LLaDA-1.5 block on a flat (T, d) stream of packed prompts (lp_model.block_forward).

    The block's own submodules do the norms, projections and MLP. Attention is bidirectional within each
    prompt and impossible across prompts (flash-attn varlen, cu = prompt boundaries); RoPE is applied in fp32.
    sin, cos: (T, head_dim) fp32, from positions that restart at 0 for every prompt.
    """
    flash_attn_varlen_func = _flash_attn_varlen()
    T, d = x.shape
    head_dim = d // n_heads
    h = block.attn_norm(x)
    q = block.q_proj(h).view(T, n_heads, head_dim)
    k = block.k_proj(h).view(T, n_heads, head_dim)
    v = block.v_proj(h).view(T, n_heads, head_dim)
    q_float = q.float()
    k_float = k.float()
    q = (q_float * cos[:, None] + _rot_half(q_float) * sin[:, None]).to(v.dtype)
    k = (k_float * cos[:, None] + _rot_half(k_float) * sin[:, None]).to(v.dtype)
    attention = flash_attn_varlen_func(q, k, v, cu, cu, maxlen, maxlen, dropout_p=0.0, causal=False)
    x = x + block.attn_out(attention.reshape(T, d))
    h = block.ff_norm(x)
    return x + block.ff_out(block.act(block.ff_proj(h)) * block.up_proj(h))


class Packed:
    """Compact prompts concatenated into one flat stream (lp_model.Packed): no padding, one segment each.

    ids, pos  (T,) token ids and per-prompt positions (0, 1, ... within every prompt)
    cu        (B + 1,) int32 prompt boundaries in the stream (cu_seqlens of flash-attn)
    maxlen    length of the longest prompt
    rows      (S,) stream index of every placeholder: the sites of prompt 0, then of prompt 1, ...
    """

    def __init__(self, examples, mask_id, device):
        ids = []
        positions = []
        cu = [0]
        rows = []
        for example in examples:
            offset = cu[-1]
            length = len(example["input_ids"])
            ids.extend(example["input_ids"])
            positions.extend(range(length))
            cu.append(offset + length)
            for position in example["mask_positions"]:
                rows.append(offset + position)
        self.ids = torch.tensor(ids, device=device)
        self.pos = torch.tensor(positions, device=device)
        self.cu = torch.tensor(cu, dtype=torch.int32, device=device)
        self.maxlen = max(len(example["input_ids"]) for example in examples)
        self.rows = torch.tensor(rows, device=device)
        assert bool((self.ids[self.rows] == mask_id).all())
        assert int((self.ids == mask_id).sum()) == len(rows)


class _LLaDATrunk:
    """Frozen LLaDA-1.5: the states entering its last block, for packed compact prompts (lp_model.Trunk)."""

    def __init__(self, adapter):
        model = adapter.model
        config = model.config
        transformer = model.model.transformer
        assert config.n_kv_heads == config.n_heads
        assert not config.alibi
        assert config.rope
        assert config.block_type == "llama"
        assert not config.attention_layer_norm
        self.mask_id = adapter.mask_id
        self.device = next(model.parameters()).device
        self.transformer = transformer
        self.blocks = list(transformer.blocks)
        self.last = self.blocks[-1]
        self.norm = transformer.ln_f
        self.n_heads = config.n_heads
        self.width = config.d_model
        head_dim = self.width // self.n_heads
        exponents = torch.arange(0, head_dim, 2, device=self.device, dtype=torch.float) / head_dim
        self.inv_freq = 1.0 / (config.rope_theta**exponents)

    def rope(self, positions):
        """sin and cos of the rotary angles at the given positions, (T, head_dim) fp32 (modeling_llada's RoPE)."""
        angles = positions.float()[:, None] * self.inv_freq[None]
        angles = torch.cat((angles, angles), -1)
        return angles.sin(), angles.cos()

    @torch.no_grad()
    def branch_input(self, packed):
        """The bf16 states entering the last block, (T, d), with the stream's sin and cos."""
        sin, cos = self.rope(packed.pos)
        x = self.transformer.wte(packed.ids)
        for block in self.blocks[:-1]:
            x = _llada_block(block, x, packed.cu, packed.maxlen, sin, cos, self.n_heads)
        return x, sin, cos


# ---------------------------------------------------------------------------------------------
# Nemotron and dQwen3.5: one prompt per forward pass, last-layer inputs captured (lpx_model.py)
# ---------------------------------------------------------------------------------------------


class _StopForward(Exception):
    """Raised by the capture hook to end the frozen forward pass at the last layer."""


def _prefix_tail_mask(length, block_start, device):
    """Nemotron's prefix/tail attention mask [1, 1, T, T] (True = may attend): causal before b, all keys from b."""
    index = torch.arange(length, device=device)
    query = index[:, None]
    key = index[None, :]
    allow = ((query < block_start) & (key <= query)) | (query >= block_start)
    return allow[None, None]


class _CaptureTrunk:
    """Frozen Nemotron (prefix/tail) or dQwen3.5: the inputs of the last layer for one compact prompt."""

    def __init__(self, adapter, backbone):
        model = adapter.model
        self.adapter = adapter
        self.backbone = backbone
        self.device = next(model.parameters()).device
        if backbone == "nemotron":
            assert adapter.mode == "dlm", "the Nemotron head was trained with every layer in diffusion mode"
            self.layers = model.encoder.layers
            self.norm = model.encoder.norm if hasattr(model.encoder, "norm") else model.norm
            self.width = model.config.hidden_size
        else:
            config = model.config
            self.layers = model.model.layers
            self.norm = model.model.norm
            self.width = config.text_config.hidden_size if hasattr(config, "text_config") else config.hidden_size
        self.last = self.layers[-1]

    def set_mask(self, length, block_start):
        """Nemotron: install the prefix/tail mask for a prompt of `length` tokens whose tail starts at block_start."""
        if self.backbone == "nemotron":
            self.adapter.explicit_mask["mask"] = _prefix_tail_mask(length, block_start, self.device)

    def clear_mask(self):
        """Nemotron: remove the mask again, so that the adapter's own forward passes are unaffected."""
        if self.backbone == "nemotron":
            self.adapter.explicit_mask["mask"] = None

    @torch.no_grad()
    def capture(self, ids):
        """Run the frozen model on one prompt up to its last layer; return that layer's (args, kwargs).

        The hook lives only for this call, so the adapter's own forward passes are unaffected. For Nemotron the
        prefix/tail mask must be set (set_mask) before the call.
        """
        captured = {}

        def hook(module, args, kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            raise _StopForward()

        handle = self.last.register_forward_pre_hook(hook, with_kwargs=True)
        x = torch.tensor([ids], device=self.device)
        try:
            if self.backbone == "nemotron":
                self.adapter.model(
                    input_ids=x,
                    labels=None,
                    use_cache=False,
                    past_key_values=None,
                    use_causal_mask=False,
                    output_last_hidden_states_only=True,
                )
            else:
                self.adapter.model(input_ids=x)
        except _StopForward:
            pass
        finally:
            handle.remove()
        return captured["args"], captured["kwargs"]


# ---------------------------------------------------------------------------------------------
# The trainable branch and the head
# ---------------------------------------------------------------------------------------------


class Branch(torch.nn.Module):
    """The trained part of the length head (lpx_model.Branch): copied last block, copied final norm, linear head.

    The copies are fp32 and separate from the frozen backbone. The head is Linear(d, lmax) over the lengths
    1..lmax, initialised from N(0, head_std^2) with a generator seeded by `seed` and a zero bias (the training
    start of Appendix B, Length Allocation; a loaded checkpoint overwrites every weight).
    """

    def __init__(self, last_block, final_norm, width, lmax=L_MAX, seed=0, head_std=1e-3):
        super().__init__()
        self.lmax = lmax
        self.block = copy.deepcopy(last_block).float()
        self.norm = copy.deepcopy(final_norm).float()
        for module in list(self.block.modules()) + list(self.norm.modules()):
            module._forward_pre_hooks.clear()
            module._forward_hooks.clear()
        generator = torch.Generator().manual_seed(seed)
        self.head = torch.nn.Linear(width, lmax)
        with torch.no_grad():
            self.head.weight.copy_(torch.randn(self.head.weight.shape, generator=generator) * head_std)
            self.head.bias.zero_()
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def param_groups(self, lr_block, lr_head, weight_decay):
        """AdamW groups of Appendix B, Length Allocation: block and norm at lr_block, head at lr_head; no decay on
        vectors."""
        block_parameters = list(self.block.parameters()) + list(self.norm.parameters())
        return [
            {"params": [p for p in block_parameters if p.ndim >= 2], "lr": lr_block, "weight_decay": weight_decay},
            {"params": [p for p in block_parameters if p.ndim < 2], "lr": lr_block, "weight_decay": 0.0},
            {"params": [self.head.weight], "lr": lr_head, "weight_decay": weight_decay},
            {"params": [self.head.bias], "lr": lr_head, "weight_decay": 0.0},
        ]


class LengthHead:
    """A frozen DLM with a trained length branch: per-site length distributions for a canvas (Appendix B, Length
    Allocation)."""

    def __init__(self, adapter, backbone, trunk, branch, config):
        self.adapter = adapter
        self.backbone = backbone  # "llada", "nemotron" or "dqwen": the name in the checkpoint and in section43_seed
        self.trunk = trunk
        self.branch = branch
        self.config = config  # the training configuration stored in the checkpoint
        self.lmax = branch.lmax
        self.device = trunk.device

    @classmethod
    def load(cls, adapter, ckpt_path, device):
        """The head of checkpoint `ckpt_path` on top of `adapter`, its branch on `device`.

        `adapter` is a loaded adapter (fewshot_dlms.adapters.load) or its name ("llada", "nemotron_pt",
        "dqwen"), which is then loaded onto `device`. The checkpoint (an authors' final.pt) is loaded unchanged
        and must belong to the adapter's backbone.
        """
        if isinstance(adapter, str):
            from fewshot_dlms.adapters import load as load_adapter

            adapter = load_adapter(adapter, device)
        if adapter.name not in BACKBONE_OF_ADAPTER:
            raise ValueError(f"no length head for adapter {adapter.name!r}; expected one of {sorted(BACKBONE_OF_ADAPTER)}")
        backbone = BACKBONE_OF_ADAPTER[adapter.name]
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        config = checkpoint["config"]
        if config["backbone"] != backbone:
            raise ValueError(f"{ckpt_path} is a {config['backbone']} head, but the adapter is {adapter.name}")
        if backbone == "llada":
            trunk = _LLaDATrunk(adapter)
        else:
            trunk = _CaptureTrunk(adapter, backbone)
        branch = Branch(trunk.last, trunk.norm, trunk.width, lmax=config["lmax"]).to(device)
        branch.load_state_dict(checkpoint["branch"])
        branch.eval()
        return cls(adapter, backbone, trunk, branch, config)

    def log_probs(self, examples):
        """Log-probabilities over the lengths 1..lmax for every site of a list of compact prompts (S, lmax), fp32.

        `examples` are dicts with "input_ids" and "mask_positions" (compact_prompt); the rows are their sites in
        order (lpx_model.Branch.forward_examples). Gradients reach the branch only.
        """
        autocast = torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
        if self.backbone == "llada":
            packed = Packed(examples, self.trunk.mask_id, self.device)
            x, sin, cos = self.trunk.branch_input(packed)
            with autocast:
                y = _llada_block(self.branch.block, x, packed.cu, packed.maxlen, sin, cos, self.trunk.n_heads)
                z = self.branch.norm(y[packed.rows]).float()
            return torch.log_softmax(self.branch.head(z), -1)

        rows = []
        for example in examples:
            ids = example["input_ids"]
            positions = example["mask_positions"]
            self.trunk.set_mask(len(ids), min(positions))
            try:
                args, kwargs = self.trunk.capture(ids)
                with autocast:
                    y = self.branch.block(*args, **kwargs)
                    y = y[0] if isinstance(y, (tuple, list)) else y
                    z = self.branch.norm(y[0, positions]).float()
            finally:
                self.trunk.clear_mask()
            rows.append(torch.log_softmax(self.branch.head(z), -1))
        return torch.cat(rows, 0)

    @torch.no_grad()
    def predict_examples(self, examples, max_tokens=MAX_TOKENS):
        """Probabilities over the lengths 1..lmax for a list of compact prompts: one float64 [J, lmax] array each.

        Prompts are grouped longest first into packs of at most max_tokens tokens (lp_model.predict); the
        probabilities are exp of the fp32 log-softmax, in float64 (lpx_alloc.py).

        LLaDA-1.5 numerics. The frozen evaluation packed 32 compact prompts (about 8,500 tokens) per forward
        pass. A bf16 matrix product over a short stream (below roughly 768 tokens on an H100) runs other GPU
        kernels, whose rounding moves a single prompt's distributions by up to 0.07 in total variation. A pack
        shorter than MIN_STREAM_TOKENS therefore gets one more segment of FILLER_TOKEN tokens without
        placeholders. Segments never attend to each other, so the filler changes nothing but the kernels, and
        every prediction then matches the frozen evaluation to about 1e-6 (tests/test_gpu_head.py).
        """
        out = [None] * len(examples)
        order = sorted(range(len(examples)), key=lambda i: -len(examples[i]["input_ids"]))
        start = 0
        while start < len(order):
            group = [order[start]]
            tokens = len(examples[order[start]]["input_ids"])
            start += 1
            while start < len(order) and tokens + len(examples[order[start]]["input_ids"]) <= max_tokens:
                group.append(order[start])
                tokens += len(examples[order[start]]["input_ids"])
                start += 1
            pack = [examples[i] for i in group]
            if self.backbone == "llada" and tokens < MIN_STREAM_TOKENS:
                pack.append({"input_ids": [FILLER_TOKEN] * (MIN_STREAM_TOKENS - tokens), "mask_positions": []})
            probs = self.log_probs(pack).exp().double().cpu().numpy()
            row = 0
            for i in group:
                n_sites = len(examples[i]["mask_positions"])
                out[i] = probs[row : row + n_sites]
                row += n_sites
        return out

    def predict(self, canvas):
        """The head's distribution over the lengths 1..lmax at each target site of a canvas: float64 [J, lmax].

        The head reads compact_prompt(canvas), so the result does not depend on the canvas's current lengths.
        """
        return self.predict_examples([compact_prompt(canvas, self.adapter.mask_id)])[0]

    def sample(self, canvas, K, seed):
        """K independent learned allocations for a canvas: K lists of J lengths in 1..lmax.

        The head is evaluated once; its probabilities are rounded to float32 and back to float64, the precision in
        which the paper's runs stored them before drawing (as experiments/length_allocation/run.py and
        experiments/java_refactoring/run.py do), and the draws are sample_allocations(those probabilities, K, seed). With
        seed = section43_seed("report", head.backbone, episode_id, J) they are the paper's Section 4.3 draws.
        """
        probs = self.predict(canvas).astype(np.float32).astype(np.float64)
        return sample_allocations(probs, K, seed)


def fresh_head(adapter, lmax=L_MAX, seed=0):
    """A LengthHead on a loaded adapter with an untrained branch: the training start of Appendix B, Length Allocation.

    The branch copies the adapter's last block and final norm and draws its head weights with standard deviation 1e-3
    from a generator seeded by `seed` (Branch). The adapter's own weights are frozen here; experiments/length_allocation/train_head.py trains
    the branch.
    """
    if adapter.name not in BACKBONE_OF_ADAPTER:
        raise ValueError(f"no length head for adapter {adapter.name!r}; expected one of {sorted(BACKBONE_OF_ADAPTER)}")
    backbone = BACKBONE_OF_ADAPTER[adapter.name]
    for parameter in adapter.model.parameters():
        parameter.requires_grad_(False)
    if backbone == "llada":
        trunk = _LLaDATrunk(adapter)
    else:
        trunk = _CaptureTrunk(adapter, backbone)
    branch = Branch(trunk.last, trunk.norm, trunk.width, lmax=lmax, seed=seed).to(trunk.device)
    return LengthHead(adapter, backbone, trunk, branch, {"backbone": backbone, "lmax": lmax})
