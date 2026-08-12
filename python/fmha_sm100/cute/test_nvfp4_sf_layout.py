# SPDX-License-Identifier: Apache-2.0
"""Golden-vector tests pinning MSA's NVFP4 KV scale-factor byte layouts.

PROVENANCE
----------
Ported -- **deliberately copied, not imported** -- from

    /home/fans/vllm/tests/v1/attention/test_nvfp4_kv_sf_layout.py

That file is pure NumPy (no torch, no CUDA, no vLLM import), so it ports by
copy.  The copy is the point: the two repos are not co-installed, and each side
must *independently re-derive* the layout.  A shared helper would let a single
edit satisfy both sides at once and defeat the entire purpose of the pin.

WHAT THIS PINS
--------------
An NVFP4 KV cache stores, per (page, kv-head), a block of FP8 (E4M3) scale
factors -- one scale per 16 contiguous elements of a token's head vector.  At
MiniMax-M3 geometry (``block_size=128``, ``head_dim=128``) that is

    128 tokens x (128 // 16) = 128 x 8 = 1024 scale bytes per (page, head).

vLLM's KV-cache store kernel writes those 1024 bytes in **two different byte
orders**, and MSA's NVFP4 prefill reader must match both:

(a) **K** scales -- plain linear ``t * S + s``.
    ``vllm/csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu:162`` (the store).

(b) **V** scales -- 4x4 token-quad swizzle required by the SM100 trtllm-gen
    MHA kernel.
    ``nvfp4_kv_cache_kernels.cu:25-39`` (``swizzle_scale_offset``) and
    ``:164-171`` (the store).

Page layout is ``[K_data | K_scale | V_data | V_scale]``
(``nvfp4_kv_cache_kernels.cu:8``).

THE FAILURE MODE THIS EXISTS FOR (plan risk R1)
-----------------------------------------------
Both layouts are bijections onto ``range(1024)``, both agree that the packed
per-(token, head) footprint is ``full_dim == 72`` bytes, and they disagree only
about *where each individual byte goes*.  If the trtllm-gen swizzle moves and
vLLM's writer follows, **every shape check, every dtype check and every
``full_dim == 72`` check still passes** -- the only symptom is degraded
accuracy.  ``TestShapeChecksCannotCatchThis`` demonstrates that directly;
``TestHostPackersMatchGoldenTable`` is the pin that makes MSA's CI fail the day
the writer moves, rather than the day someone notices the perplexity.

All three layouts agree at ``(0, 0)`` and at ``(127, 7)``, so **those are
useless as probes**.  ``(t=1, s=0)`` is the probe: linear -> 8, swizzle -> 1.

WHAT WAS DROPPED IN THE PORT, AND WHY
-------------------------------------
* **The 128x4 column of ``GOLDEN_OFFSETS``.**  The cuBLAS/cuDNN 128x4 tiling is
  deleted from this branch entirely (plan section 1): no flag, no dual path, no
  test-only remnant.  MSA uses no block-scaled ``tcgen05`` MMA, so 128x4 buys it
  nothing.  The golden table keeps the same 20 (t, s) entries with **2** columns
  instead of 3.

* **``TestAlignmentInvariant`` -- DELIBERATELY DROPPED, NOT LOST.**  It existed
  to document the ``block_size % 128 == 0`` constraint.  That constraint is a
  property of 128x4's tiling of the *global* scale matrix: a per-(page, head)
  region was independently addressable only when its first global row was
  128-aligned.  **Neither surviving layout has a global row map** -- linear and
  4x4 are both page-local by construction, indexing inside the page via
  ``block_offset``.  So the constraint no longer applies to anything in this
  branch and is retired (plan section 11).  The page-locality that replaces it
  is asserted positively in
  ``TestBijection::test_both_layouts_are_page_local_by_construction``.

* **``msa_scale_128x4_offset`` / ``msa_paged_kv_scale_row`` /
  ``msa_region_base_offset``** and the 128x4-specific bijection and padding
  tests, for the same reason.  ``TestSwizzlePrecondition``'s second test,
  ``test_msa_pads_instead_of_breaking_on_ragged_scale_cols``, went with them:
  it is entirely about 128x4's ``round_up(scale_cols, 4)`` padding.

WHAT WAS **KEPT**, AND WHY THE ASYMMETRY IS DELIBERATE
------------------------------------------------------
``TestSwizzlePrecondition::test_v_swizzle_needs_head_size_multiple_of_64`` is
**ported, not dropped** -- the opposite decision from ``TestAlignmentInvariant``
above, and the two are worth contrasting.  The plan's section 9b names only
``TestBijection``, ``TestGoldenTable`` and ``TestShapeChecksCannotCatchThis``
as the classes to port, so this class is unaccounted for there; dropping it on
that silence would have silently retired a **live** constraint.

The difference is ownership.  ``block_size % 128 == 0`` was a property of
128x4's tiling of the global scale matrix, and dies with that layout.
``head_size % 64 == 0`` is a property of the **V swizzle itself** --
``swizzle_scale_offset`` divides by ``s_group = scale_dim / 4``, so at
head_size 16/32/48 the scale_dim is 1/2/3, ``s_group == 0``, and the writer
divides by zero (``nvfp4_kv_cache_kernels.cu:208-215``).  The swizzle survives
this change, so the constraint survives with it, and it is one of the plan's
own section 11 boundary conditions.  The vLLM comment at ``:523-529`` states
the test exists so nobody relaxes the check from ``% 64`` back to the more
obvious ``% 16``; that reason is untouched by anything here.

BOUNDARY CONDITIONS ENCODED HERE (plan section 11)
--------------------------------------------------
* ``head_dim == 128`` -- 4x4 is free for the reader only while a warp spans
  whole token quads (``32 / S >= 4``, i.e. D <= 128).  Enforced at
  ``interface.py:323-326`` and kernel ``:78-81``.
* ``block_size % 4 == 0`` -- the V swizzle tiles tokens in quads; otherwise the
  last quad straddles the region.  ``nvfp4_kv_cache_kernels.cu:216-218``.
* ``head_size % 64 == 0`` -- ``s_group = S / 4`` must be non-zero or the writer
  divides by zero.  ``nvfp4_kv_cache_kernels.cu:208-215``.

RUNNING
-------
The pure-integer tests (golden table, bijection, simplification equivalence,
layout agreement, negative control) need **numpy only** -- no GPU, no torch.
The two host-packer tests import ``quantize`` lazily and skip when torch is
absent.  This is the fastest signal after any layout edit::

    python3 -m pytest -q test_nvfp4_sf_layout.py
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# M3 geometry
# ---------------------------------------------------------------------------

M3_BLOCK_SIZE = 128  # tokens per page
M3_HEAD_DIM = 128  # elements per (token, head)
M3_NUM_KV_HEADS = 4
M3_SCALE_COLS = M3_HEAD_DIM // 16  # 8 scale bytes per (token, head)
M3_REGION_BYTES = M3_BLOCK_SIZE * M3_SCALE_COLS  # 1024


# ---------------------------------------------------------------------------
# (a) vLLM K scale: LINEAR
#     nvfp4_kv_cache_kernels.cu:155-163
#
#       scale_dst = scale_block
#                 + head        * scale_head_stride
#                 + block_offset* scale_block_offset_stride
#                 + scale_idx;
#
#     with the HND strides from the dispatch (:243-248):
#       scale_head_stride         = block_size * scale_dim
#       scale_block_offset_stride = scale_dim
#     so, *within* one (page, head) region, offset = t * scale_dim + s.
# ---------------------------------------------------------------------------


def vllm_k_scale_offset(t, s, scale_cols):
    """Byte offset of the K scale for token ``t``, scale-column ``s``."""
    t = np.asarray(t)
    s = np.asarray(s)
    return t * scale_cols + s


# ---------------------------------------------------------------------------
# (b) vLLM V scale: 4x4 token-quad swizzle
#     nvfp4_kv_cache_kernels.cu:25-39
#
#       s_group    = scale_dim / 4;
#       swizzled_t = (t / 4) * 4 + (s / s_group);
#       swizzled_s = (s % s_group) * 4 + (t % 4);
#       return swizzled_t * scale_dim + swizzled_s;
#
#     and :164-171, which splits that back into (swizzled_t, swizzled_s) and
#     re-applies the HND strides -- which, since
#     scale_block_offset_stride == scale_dim, reproduces the same value.
# ---------------------------------------------------------------------------


def vllm_v_scale_offset(t, s, scale_cols):
    """Byte offset of the V scale for token ``t``, scale-column ``s``.

    Transcribed line-for-line from the C++ ``swizzle_scale_offset``.
    """
    t = np.asarray(t)
    s = np.asarray(s)
    s_group = scale_cols // 4
    swizzled_t = (t // 4) * 4 + (s // s_group)
    swizzled_s = (s % s_group) * 4 + (t % 4)
    return swizzled_t * scale_cols + swizzled_s


# ---------------------------------------------------------------------------
# MSA's own re-derivations -- what the kernel and quantize.py implement.
#
# These are written the way MSA computes them, NOT by calling the two functions
# above.  If either side drifts, the golden table breaks loudly.
#
# The V swizzle simplifies (plan section 2).  Substituting S = 4*s_group into
# the C form collapses both s terms:
#
#     (s // s_group)*S + (s % s_group)*4
#   = 4*(s_group*(s // s_group) + s % s_group)
#   = 4*s
#
# so for any S divisible by 4:
#
#     offset_V(t, s) = (t // 4) * (4*S) + 4*s + (t % 4)
#
# i.e. a plain row-major [T/4][S][4] array indexed by (t//4, s, t%4).  In the
# kernel that is two shifts and a mask: (t>>2)*(4*S) + (s<<2) + (t&3).
# TestSimplificationEquivalence proves the two forms identical.
# ---------------------------------------------------------------------------


def msa_linear_offset(t, s, scale_cols):
    """MSA K reader / ``quantize.nvfp4_scale_linear_offset``: ``t*S + s``."""
    t = np.asarray(t)
    s = np.asarray(s)
    return t * scale_cols + s


def msa_swizzle4x4_offset(t, s, scale_cols):
    """MSA V reader / ``quantize.nvfp4_scale_swizzle4x4_offset``.

    Closed form: ``(t // 4) * (4*S) + 4*s + (t % 4)``.
    """
    t = np.asarray(t)
    s = np.asarray(s)
    return (t // 4) * (4 * scale_cols) + 4 * s + (t % 4)


# ---------------------------------------------------------------------------
# full_dim: the shape both sides agree on
# ---------------------------------------------------------------------------


def vllm_full_dim(head_size):
    """vLLM ``nvfp4_kv_cache_full_dim``: fp4 data bytes + fp8 scale bytes.

    Mirrors ``vllm/utils/torch_utils.py:414-417`` and the C++ dispatch's
    ``data_dim + scale_dim`` (nvfp4_kv_cache_kernels.cu:194-196).
    """
    return head_size // 2 + head_size // 16


def msa_full_dim(head_dim):
    """Same quantity on the MSA side: 2 fp4/byte data + 1 fp8 scale per 16."""
    data_bytes = head_dim // 2  # cvt_fp4x8 packing, 2 values per byte
    scale_bytes = head_dim // 16  # scale_cols = head_dim // 16
    return data_bytes + scale_bytes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Exactly two layouts survive on this branch.  There is no third entry, and no
# flag that could select one: K is always linear, V is always the 4x4 swizzle.
_LAYOUTS = {
    "linear": lambda t, s: msa_linear_offset(t, s, M3_SCALE_COLS),
    "swizzle4x4": lambda t, s: msa_swizzle4x4_offset(t, s, M3_SCALE_COLS),
}

# The layout-name strings frozen by the API contract.  ``pack_nvfp4_scale``
# accepts exactly these; anything else is a typo that must not silently pick a
# default.
LAYOUT_NAMES = ("linear", "swizzle4x4")

_OFFSET_FNS = {
    "linear": msa_linear_offset,
    "swizzle4x4": msa_swizzle4x4_offset,
}


def _all_offsets(fn, block_size=M3_BLOCK_SIZE, scale_cols=M3_SCALE_COLS):
    t = np.repeat(np.arange(block_size), scale_cols)
    s = np.tile(np.arange(scale_cols), block_size)
    return np.asarray(fn(t, s)).reshape(-1)


def _ts_grid(block_size=M3_BLOCK_SIZE, scale_cols=M3_SCALE_COLS):
    t = np.repeat(np.arange(block_size), scale_cols)
    s = np.tile(np.arange(scale_cols), block_size)
    return t, s


# ---------------------------------------------------------------------------
# 1. Each layout is a BIJECTION over the 1024-byte per-(page, head) region
# ---------------------------------------------------------------------------


class TestBijection:
    @pytest.mark.parametrize("name", sorted(_LAYOUTS))
    def test_bijection_over_1024_bytes(self, name):
        """M3 geometry: 128 tokens x 8 scale cols tiles exactly 1024 bytes."""
        offsets = _all_offsets(_LAYOUTS[name])
        assert offsets.size == M3_REGION_BYTES == 1024
        assert sorted(offsets.tolist()) == list(range(M3_REGION_BYTES)), (
            f"{name} is not a bijection over the {M3_REGION_BYTES}-byte region"
        )

    @pytest.mark.parametrize("name", sorted(_LAYOUTS))
    @pytest.mark.parametrize("block_size", [4, 8, 16, 32, 64, 128, 256])
    def test_both_layouts_are_page_local_by_construction(self, name, block_size):
        """No global row map: a region is self-contained at any block_size % 4 == 0.

        This is what replaces the dropped ``TestAlignmentInvariant``.  128x4
        tiled the *global* scale matrix and therefore needed
        ``block_size % 128 == 0``; both surviving layouts address only inside
        the (page, head) region, so no page/head alignment constraint exists.
        Note the offsets do not depend on page or head at all -- that is the
        property being asserted.
        """
        offsets = _all_offsets(_LAYOUTS[name], block_size=block_size)
        assert sorted(offsets.tolist()) == list(range(block_size * M3_SCALE_COLS))

    def test_no_page_or_head_term_exists(self):
        """Both offset functions take only (t, s, S) -- there is nothing else
        to get wrong.  The page/head displacement is a host-supplied byte
        stride added outside these functions (plan section 7 requirement 1)."""
        import inspect

        for fn in (msa_linear_offset, msa_swizzle4x4_offset):
            params = list(inspect.signature(fn).parameters)
            assert params == ["t", "s", "scale_cols"], (
                f"{fn.__name__} grew a page/head parameter: {params}"
            )


# ---------------------------------------------------------------------------
# 2. Frozen golden table
#
# Ported from vLLM's tests/v1/attention/test_nvfp4_kv_sf_layout.py with the
# 128x4 column dropped: same 20 (t, s) entries, 2 columns instead of 3.
#
# Measured from the implementations above against
#   vllm  csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu
# DO NOT "fix" a failure here by editing the table.  A diff here means the
# writer changed its scale layout, which is an ABI break for MSA's reader.
# ---------------------------------------------------------------------------

# (t, s) -> (linear, swizzle4x4)
GOLDEN_OFFSETS: dict = {
    (0, 0): (0, 0),
    (0, 1): (1, 4),
    (0, 3): (3, 12),
    (0, 4): (4, 16),
    (0, 7): (7, 28),
    (1, 0): (8, 1),
    (1, 1): (9, 5),
    (2, 0): (16, 2),
    (3, 0): (24, 3),
    (3, 7): (31, 31),
    (4, 0): (32, 32),
    (4, 4): (36, 48),
    (5, 2): (42, 41),
    (7, 7): (63, 63),
    (31, 0): (248, 227),
    (32, 0): (256, 256),
    (32, 7): (263, 284),
    (63, 3): (507, 495),
    (64, 0): (512, 512),
    (127, 7): (1023, 1023),
}

# The plan's section 2 worked example, restated as an explicit sub-table so a
# reader can check the doc against the code without re-deriving anything.
WORKED_EXAMPLE = {
    (0, 0): (0, 0),
    (1, 0): (8, 1),
    (0, 4): (4, 16),
    (4, 0): (32, 32),
    (32, 0): (256, 256),
    (127, 7): (1023, 1023),
}


class TestGoldenTable:
    def test_table_covers_the_anchors(self):
        assert (0, 0) in GOLDEN_OFFSETS
        assert (1, 0) in GOLDEN_OFFSETS
        assert len(GOLDEN_OFFSETS) >= 20

    def test_table_has_exactly_two_columns(self):
        """128x4 is deleted from this branch; its column is gone from the
        table too.  Not commented out, not zero-filled -- gone."""
        assert all(len(v) == 2 for v in GOLDEN_OFFSETS.values())

    @pytest.mark.parametrize(("ts", "expected"), sorted(GOLDEN_OFFSETS.items()))
    def test_golden_offsets(self, ts, expected):
        t, s = ts
        actual = (
            int(msa_linear_offset(t, s, M3_SCALE_COLS)),
            int(msa_swizzle4x4_offset(t, s, M3_SCALE_COLS)),
        )
        assert actual == expected, (
            f"scale layout changed at (t={t}, s={s}): got {actual}, golden {expected}"
        )

    @pytest.mark.parametrize(("ts", "expected"), sorted(WORKED_EXAMPLE.items()))
    def test_matches_the_plan_worked_example(self, ts, expected):
        assert GOLDEN_OFFSETS[ts] == expected

    def test_origin_is_the_only_forced_agreement(self):
        """Both layouts must map (0, 0) -> 0; that is why (0,0) is useless as a
        correctness probe and (1, 0) is the real one."""
        for name, fn in _LAYOUTS.items():
            assert int(fn(0, 0)) == 0, name

    def test_table_is_not_dominated_by_agreeing_entries(self):
        """A table full of (0,0)-style agreements would pass under a broken
        layout.  Most entries must actually distinguish the two layouts."""
        disagree = [ts for ts, (k, v) in GOLDEN_OFFSETS.items() if k != v]
        agree = [ts for ts, (k, v) in GOLDEN_OFFSETS.items() if k == v]
        assert len(disagree) == 13
        assert len(agree) == 7
        assert len(disagree) > len(agree)
        # the two forced anchors are in the agreeing set, as documented
        assert (0, 0) in agree and (127, 7) in agree
        # and the probe is in the disagreeing set
        assert (1, 0) in disagree


# ---------------------------------------------------------------------------
# 3. Simplification equivalence (plan section 2, section 9b item 1)
#
# The kernel implements the closed form, not the 4-line C form.  If they ever
# diverge, the kernel silently reads the wrong byte for every token whose
# t % 4 != 0.
# ---------------------------------------------------------------------------


def _c_form_swizzle(t, s, scale_dim):
    """Literal 4-line transcription of ``swizzle_scale_offset``
    (nvfp4_kv_cache_kernels.cu:33-39), integer-for-integer."""
    s_group = scale_dim // 4
    swizzled_t = (t // 4) * 4 + (s // s_group)
    swizzled_s = (s % s_group) * 4 + (t % 4)
    return swizzled_t * scale_dim + swizzled_s


class TestSimplificationEquivalence:
    @pytest.mark.parametrize("scale_cols", [4, 8, 16, 32])
    def test_closed_form_equals_c_form(self, scale_cols):
        """For every S divisible by 4 and every t < 256, the closed form and
        the C form are bit-identical."""
        for t in range(256):
            for s in range(scale_cols):
                c = _c_form_swizzle(t, s, scale_cols)
                closed = int(msa_swizzle4x4_offset(t, s, scale_cols))
                assert c == closed, (
                    f"S={scale_cols} t={t} s={s}: C form {c} != closed form {closed}"
                )

    @pytest.mark.parametrize("scale_cols", [4, 8, 16, 32])
    def test_closed_form_equals_the_shift_and_mask_kernel_form(self, scale_cols):
        """``(t>>2)*(4*S) + (s<<2) + (t&3)`` -- the form the kernel emits,
        with no division at all."""
        for t in range(256):
            for s in range(scale_cols):
                shifted = (t >> 2) * (4 * scale_cols) + (s << 2) + (t & 3)
                assert shifted == int(msa_swizzle4x4_offset(t, s, scale_cols))

    @pytest.mark.parametrize("scale_cols", [4, 8, 16, 32])
    def test_swizzle_is_row_major_T4_S_4(self, scale_cols):
        """The closed form *is* a plain row-major ``[T/4][S][4]`` array indexed
        by ``(t//4, s, t%4)`` -- that is why it costs the kernel nothing."""
        tokens = 256
        arr = np.arange(tokens * scale_cols).reshape(tokens // 4, scale_cols, 4)
        for t in range(tokens):
            for s in range(scale_cols):
                flat = int(msa_swizzle4x4_offset(t, s, scale_cols))
                assert arr[t // 4, s, t % 4] == flat

    @pytest.mark.parametrize("scale_cols", [4, 8, 16, 32])
    def test_c_form_matches_vllm_transcription(self, scale_cols):
        """The vectorised ``vllm_v_scale_offset`` and the scalar C-form
        transcription must agree; both are transcriptions of the same C++."""
        for t in range(0, 256, 7):
            for s in range(scale_cols):
                assert int(vllm_v_scale_offset(t, s, scale_cols)) == _c_form_swizzle(
                    t, s, scale_cols
                )


# ---------------------------------------------------------------------------
# 4. THE FLIPPED ASSERTION (plan section 9b item 4, section 6)
#
# vLLM's tests/v1/attention/test_nvfp4_kv_sf_layout.py carries
# ``TestLayoutsDisagree``, which asserts MSA's reader and vLLM's writer use
# different byte orders and therefore need an adapter.  That file names itself
# as the acceptance criterion for this work.  Here is the flip: MSA's two
# layouts now AGREE with vLLM's two writers, byte for byte.
#
# (0, 0) and (127, 7) agree under every layout ever considered, so they prove
# nothing.  (1, 0) is the probe: 8 vs 1.
# ---------------------------------------------------------------------------


class TestMsaAgreesWithVllm:
    def test_agreement_at_the_probe_t1_s0(self):
        vk = int(vllm_k_scale_offset(1, 0, M3_SCALE_COLS))
        vv = int(vllm_v_scale_offset(1, 0, M3_SCALE_COLS))
        mk = int(msa_linear_offset(1, 0, M3_SCALE_COLS))
        mv = int(msa_swizzle4x4_offset(1, 0, M3_SCALE_COLS))

        assert mk == vk == 8, "MSA K reader no longer matches vLLM's linear K store"
        assert mv == vv == 1, "MSA V reader no longer matches vLLM's swizzled V store"
        # and the probe really is a probe: the two layouts differ there.
        assert mk != mv

    def test_agreement_is_total_not_incidental(self):
        """Agreement must hold for all 1024 cells, not just the probe."""
        t, s = _ts_grid()
        assert np.array_equal(
            np.asarray(msa_linear_offset(t, s, M3_SCALE_COLS)),
            np.asarray(vllm_k_scale_offset(t, s, M3_SCALE_COLS)),
        )
        assert np.array_equal(
            np.asarray(msa_swizzle4x4_offset(t, s, M3_SCALE_COLS)),
            np.asarray(vllm_v_scale_offset(t, s, M3_SCALE_COLS)),
        )

    @pytest.mark.parametrize("scale_cols", [4, 8, 16, 32])
    @pytest.mark.parametrize("block_size", [4, 8, 64, 128])
    def test_agreement_holds_off_the_m3_geometry(self, scale_cols, block_size):
        t, s = _ts_grid(block_size=block_size, scale_cols=scale_cols)
        assert np.array_equal(
            np.asarray(msa_linear_offset(t, s, scale_cols)),
            np.asarray(vllm_k_scale_offset(t, s, scale_cols)),
        )
        assert np.array_equal(
            np.asarray(msa_swizzle4x4_offset(t, s, scale_cols)),
            np.asarray(vllm_v_scale_offset(t, s, scale_cols)),
        )


# ---------------------------------------------------------------------------
# 5. full_dim == 72 on both sides WHILE the two layouts disagree internally
# ---------------------------------------------------------------------------


class TestShapeChecksCannotCatchThis:
    def test_full_dim_agrees_at_72(self):
        assert vllm_full_dim(M3_HEAD_DIM) == 72
        assert msa_full_dim(M3_HEAD_DIM) == 72
        assert vllm_full_dim(M3_HEAD_DIM) == msa_full_dim(M3_HEAD_DIM)

    def test_equal_shapes_coexist_with_unequal_offsets(self):
        """The whole point, and the reason risk R1 needs a golden-vector test:
        every shape/size invariant matches while byte order does not.  No dtype
        check, no ``size(3) == full_dim`` check and no page-size arithmetic can
        detect a scale-layout mismatch."""
        # Same last dim.
        assert vllm_full_dim(M3_HEAD_DIM) == msa_full_dim(M3_HEAD_DIM) == 72
        # Same scale-region size.
        assert M3_BLOCK_SIZE * M3_SCALE_COLS == M3_REGION_BYTES
        # Same total bytes per (page, head) for data+scale.
        assert M3_BLOCK_SIZE * 72 == 9216
        # Both layouts are bijections over the identical byte range ...
        for fn in _LAYOUTS.values():
            assert sorted(_all_offsets(fn).tolist()) == list(range(M3_REGION_BYTES))
        # ... and yet they place token 1's first scale 7 bytes apart:
        assert int(msa_linear_offset(1, 0, M3_SCALE_COLS)) != int(
            msa_swizzle4x4_offset(1, 0, M3_SCALE_COLS)
        )

    @pytest.mark.parametrize("head_size", [64, 128, 256])
    def test_full_dim_formula_matches_across_head_sizes(self, head_size):
        assert vllm_full_dim(head_size) == msa_full_dim(head_size)


# ---------------------------------------------------------------------------
# 6. NEGATIVE CONTROL (plan section 9d item 1) -- the gate must be able to fail
#
# Cross-feed the two surviving layouts: pack with one, gather with the other.
# The recovered-value count must be EXACTLY 64 of 1024.
#
# 64 is not a guess.  It is measured in vLLM's
# tests/v1/attention/test_nvfp4_kv_sf_layout.py::TestLayoutsDisagree::
# test_disagreement_is_pervasive_not_incidental, which asserts
# ``int((k == v).sum()) == 64`` for exactly these two layouts.
#
# HARD EQUALITY, not an inequality.  An inequality ("> 0 wrong") drifts
# silently: a partially-correct new layout could keep passing it forever.
#
# Note the logical payload is int32 with 1024 DISTINCT values, not uint8.  With
# uint8 payloads the 1024 cells cannot all be distinct, so value aliasing would
# inflate the recovered count above the true positional-agreement count and the
# hard equality would be meaningless.
# ---------------------------------------------------------------------------


class TestNegativeControl:
    def _pack(self, layout):
        t, s = _ts_grid()
        off = np.asarray(_OFFSET_FNS[layout](t, s, M3_SCALE_COLS))
        logical = np.arange(M3_REGION_BYTES, dtype=np.int32)  # distinct payload
        buf = np.full(M3_REGION_BYTES, -1, dtype=np.int32)
        buf[off] = logical
        assert not (buf == -1).any(), f"{layout} left holes -- not a bijection"
        return buf, off, logical

    def test_nvfp4_negative_control_linear_packed_read_as_swizzle(self):
        buf, _, logical = self._pack("linear")
        t, s = _ts_grid()
        wrong_off = np.asarray(msa_swizzle4x4_offset(t, s, M3_SCALE_COLS))
        recovered = int((buf[wrong_off] == logical).sum())
        assert recovered == 64, (
            "cross-gather recovered %d of 1024 (expected exactly 64); the two "
            "layouts have changed relative to each other" % recovered
        )

    def test_nvfp4_negative_control_swizzle_packed_read_as_linear(self):
        buf, _, logical = self._pack("swizzle4x4")
        t, s = _ts_grid()
        wrong_off = np.asarray(msa_linear_offset(t, s, M3_SCALE_COLS))
        recovered = int((buf[wrong_off] == logical).sum())
        assert recovered == 64, (
            "cross-gather recovered %d of 1024 (expected exactly 64)" % recovered
        )

    def test_nvfp4_negative_control_offset_agreement_count_is_64(self):
        """The same 64, stated positionally: this is the exact quantity vLLM's
        test_disagreement_is_pervasive_not_incidental asserts."""
        t, s = _ts_grid()
        k = np.asarray(msa_linear_offset(t, s, M3_SCALE_COLS))
        v = np.asarray(msa_swizzle4x4_offset(t, s, M3_SCALE_COLS))
        assert int((k == v).sum()) == 64
        # 64 / 1024 = 6.25%: a wrong reader is right on token 0 and on every
        # t % 4 == 0 whose column happens to line up -- which is exactly why
        # the bug survives smoke tests and destroys accuracy.
        assert int((k == v).sum()) < M3_REGION_BYTES // 8

    def test_nvfp4_negative_control_self_gather_is_perfect(self):
        """Sanity: the control is measuring the layout, not a broken harness.
        Gathering with the SAME layout must recover all 1024."""
        for layout in LAYOUT_NAMES:
            buf, off, logical = self._pack(layout)
            assert int((buf[off] == logical).sum()) == M3_REGION_BYTES


# ---------------------------------------------------------------------------
# 7. HOST PACKERS (plan section 9b items 2 and 3) -- THIS IS THE R1 PIN
#
# quantize.py's two offset helpers must reproduce the frozen golden table, and
# pack_nvfp4_scale must round-trip through them.  These need torch (quantize.py
# imports it), so they import lazily and skip when torch is absent -- the rest
# of this file is pure integer arithmetic and runs anywhere.
# ---------------------------------------------------------------------------


def _quantize_module():
    pytest.importorskip("torch", reason="quantize.py imports torch")
    import quantize

    for name in ("nvfp4_scale_linear_offset", "nvfp4_scale_swizzle4x4_offset",
                 "pack_nvfp4_scale"):
        assert hasattr(quantize, name), (
            f"quantize.py is missing {name}; the frozen API contract is "
            "nvfp4_scale_linear_offset / nvfp4_scale_swizzle4x4_offset / "
            "pack_nvfp4_scale(scale, *, page_size, scale_cols, layout)"
        )
    return quantize


class TestHostPackersMatchGoldenTable:
    """R1 pin: MSA's own host helpers, checked against the frozen table.

    If the trtllm-gen swizzle moves and vLLM's writer follows, the *device*
    symptom is only degraded accuracy.  This is the test that turns it into a
    CI failure, from the MSA side, on the day it happens.
    """

    @pytest.mark.parametrize(("ts", "expected"), sorted(GOLDEN_OFFSETS.items()))
    def test_quantize_offsets_reproduce_golden(self, ts, expected):
        quantize = _quantize_module()
        t, s = ts
        actual = (
            int(quantize.nvfp4_scale_linear_offset(t, s, M3_SCALE_COLS)),
            int(quantize.nvfp4_scale_swizzle4x4_offset(t, s, M3_SCALE_COLS)),
        )
        assert actual == expected, (
            f"quantize.py's offsets moved at (t={t}, s={s}): got {actual}, "
            f"golden {expected}"
        )

    def test_quantize_offsets_match_the_local_rederivations_everywhere(self):
        quantize = _quantize_module()
        for t in range(M3_BLOCK_SIZE):
            for s in range(M3_SCALE_COLS):
                assert int(
                    quantize.nvfp4_scale_linear_offset(t, s, M3_SCALE_COLS)
                ) == int(msa_linear_offset(t, s, M3_SCALE_COLS))
                assert int(
                    quantize.nvfp4_scale_swizzle4x4_offset(t, s, M3_SCALE_COLS)
                ) == int(msa_swizzle4x4_offset(t, s, M3_SCALE_COLS))

    @pytest.mark.parametrize("scale_cols", [4, 8, 16, 32])
    def test_quantize_offsets_match_vllm_off_the_m3_geometry(self, scale_cols):
        quantize = _quantize_module()
        for t in range(0, 256, 5):
            for s in range(scale_cols):
                assert int(quantize.nvfp4_scale_linear_offset(t, s, scale_cols)) == int(
                    vllm_k_scale_offset(t, s, scale_cols)
                )
                assert int(
                    quantize.nvfp4_scale_swizzle4x4_offset(t, s, scale_cols)
                ) == int(vllm_v_scale_offset(t, s, scale_cols))


class TestPackRoundTrip:
    """``pack_nvfp4_scale(logical, layout=L)`` gathered with ``offset_L``
    returns ``logical`` -- for both L, over several regions, with no padding.
    """

    @pytest.mark.parametrize("layout", LAYOUT_NAMES)
    def test_round_trip(self, layout):
        quantize = _quantize_module()
        import torch

        n_regions = 3
        page_size = M3_BLOCK_SIZE
        scale_cols = M3_SCALE_COLS

        # Varying bytes: a constant payload makes any permutation a no-op and
        # the round-trip proves nothing (plan section 9c).
        g = torch.Generator().manual_seed(1234)
        logical = torch.randint(
            0x30, 0x3F, (n_regions, page_size, scale_cols),
            dtype=torch.uint8, generator=g,
        )

        packed = quantize.pack_nvfp4_scale(
            logical, page_size=page_size, scale_cols=scale_cols, layout=layout
        )

        # No padding: exactly page_size * scale_cols bytes per region.
        assert packed.numel() == n_regions * page_size * scale_cols, (
            f"pack_nvfp4_scale padded: {packed.numel()} bytes for "
            f"{n_regions} x {page_size} x {scale_cols}"
        )
        flat = packed.reshape(n_regions, page_size * scale_cols)

        t, s = _ts_grid(block_size=page_size, scale_cols=scale_cols)
        off = torch.as_tensor(
            np.asarray(_OFFSET_FNS[layout](t, s, scale_cols)), dtype=torch.long
        )
        for r in range(n_regions):
            gathered = flat[r][off].reshape(page_size, scale_cols)
            assert torch.equal(gathered, logical[r]), (
                f"round-trip failed for layout={layout}, region {r}"
            )

    def test_the_two_layouts_produce_different_bytes(self):
        """Guards against ``layout=`` being ignored: a packer that silently
        dropped the argument would round-trip perfectly under both names."""
        quantize = _quantize_module()
        import torch

        g = torch.Generator().manual_seed(99)
        logical = torch.randint(
            0x30, 0x3F, (1, M3_BLOCK_SIZE, M3_SCALE_COLS),
            dtype=torch.uint8, generator=g,
        )
        a = quantize.pack_nvfp4_scale(
            logical, page_size=M3_BLOCK_SIZE, scale_cols=M3_SCALE_COLS,
            layout="linear",
        ).reshape(-1)
        b = quantize.pack_nvfp4_scale(
            logical, page_size=M3_BLOCK_SIZE, scale_cols=M3_SCALE_COLS,
            layout="swizzle4x4",
        ).reshape(-1)
        assert not torch.equal(a, b), "layout= appears to be ignored"


# ---------------------------------------------------------------------------
# 8. Boundary conditions (plan section 11)
#
# These are properties of the layouts, encoded so that relaxing a constraint
# elsewhere trips a test here rather than producing silent corruption.
# ---------------------------------------------------------------------------


class TestSwizzlePrecondition:
    """PORTED from vLLM's class of the same name (:533-564), minus its
    128x4-only second test.  Kept rather than dropped: unlike
    ``block_size % 128 == 0``, this constraint belongs to the V swizzle itself
    and survives the removal of 128x4.  See the header for the contrast.
    """

    @pytest.mark.parametrize(
        ("head_size", "well_formed"),
        [
            (16, False),  # scale_cols=1 -> s_group=0 -> integer div by zero
            (32, False),  # scale_cols=2 -> s_group=0
            (48, False),  # scale_cols=3 -> s_group=0
            (64, True),  # scale_cols=4
            (128, True),  # scale_cols=8  <- M3
            (192, True),  # scale_cols=12
            (256, True),  # scale_cols=16
        ],
    )
    def test_v_swizzle_needs_head_size_multiple_of_64(self, head_size, well_formed):
        """``swizzle_scale_offset`` divides by ``s_group = scale_dim / 4``, so
        head_size must be % 64, not merely % 16.  vLLM enforces exactly that at
        nvfp4_kv_cache_kernels.cu:208-215; this pins the *reason*, so nobody
        relaxes it back to % 16."""
        scale_cols = head_size // 16
        assert (scale_cols % 4 == 0) == well_formed
        # This is exactly the dispatch's STD_TORCH_CHECK condition ...
        assert (head_size % 64 == 0) == well_formed
        # ... and the weaker % 16 rule would have let 32 and 48 through.
        assert head_size % 16 == 0

        if not well_formed:
            # s_group == 0: the device kernel would divide by zero here.
            assert scale_cols // 4 == 0
            # The C form is what actually divides; show it raises rather than
            # quietly returning something.
            with pytest.raises(ZeroDivisionError):
                _c_form_swizzle(1, 0, scale_cols)
            return

        offs = sorted(
            int(msa_swizzle4x4_offset(t, s, scale_cols))
            for t in range(M3_BLOCK_SIZE)
            for s in range(scale_cols)
        )
        assert offs == list(range(M3_BLOCK_SIZE * scale_cols))

    @pytest.mark.parametrize("head_size", [16, 32, 48])
    def test_bijectivity_cannot_detect_a_ragged_head_size(self, head_size):
        """MSA's closed form stays a *bijection* where vLLM's writer faults.

        This is the asymmetry that makes the ``% 64`` check load-bearing on the
        MSA side too.  ``(t//4)*(4S) + 4s + (t%4)`` is a row-major
        ``[T/4][S][4]`` reshape, and a reshape is bijective for **any** S >= 1
        -- including S = 1, 2, 3.  So bijectivity is NOT a safety property here:
        MSA would happily read a self-consistent permutation of a cache that
        vLLM could never have written, because ``swizzle_scale_offset`` divides
        by ``s_group == 0`` and faults first.

        Same shape as the file's main theme (``TestShapeChecksCannotCatchThis``):
        the structural check passes and the bytes are still wrong.
        ``head_dim == 128`` is what keeps this unreachable in practice
        (interface.py:323-326, kernel :78-81).
        """
        scale_cols = head_size // 16
        assert scale_cols < 4  # ragged: s_group would be 0

        offs = sorted(
            int(msa_swizzle4x4_offset(t, s, scale_cols))
            for t in range(M3_BLOCK_SIZE)
            for s in range(scale_cols)
        )
        # Bijective -- the check that would "catch" this does not.
        assert offs == list(range(M3_BLOCK_SIZE * scale_cols))
        # ... while the writer cannot even compute an offset.
        with pytest.raises(ZeroDivisionError):
            _c_form_swizzle(1, 0, scale_cols)

    @pytest.mark.parametrize("scale_cols", [5, 6, 7, 9, 10])
    def test_simplification_requires_four_divides_S(self, scale_cols):
        """The closed form is only equal to the C form when ``4 | S``.

        For S not divisible by 4 the collapse
        ``(s//s_group)*S + (s%s_group)*4 -> 4*s`` does not hold, and the two
        forms disagree.  This is the precondition of the whole simplification
        in plan section 2, stated as a test rather than as prose -- so that
        extending the kernel to a ragged S trips here instead of silently
        reading the wrong byte.
        """
        assert scale_cols % 4 != 0
        differs = any(
            _c_form_swizzle(t, s, scale_cols)
            != int(msa_swizzle4x4_offset(t, s, scale_cols))
            for t in range(64)
            for s in range(scale_cols)
        )
        assert differs, (
            f"closed form unexpectedly matches the C form at S={scale_cols}; "
            "the 4 | S precondition may be weaker than documented"
        )


class TestBoundaryConditions:
    @pytest.mark.parametrize("block_size", [1, 2, 3, 4, 5, 8, 127, 128])
    def test_v_swizzle_needs_block_size_multiple_of_4(self, block_size):
        """The V swizzle tiles tokens in quads.  When ``block_size % 4 != 0``
        the last quad straddles the region: the offsets leave the region and
        stop being a bijection over ``block_size * S`` bytes.  vLLM rejects it
        at nvfp4_kv_cache_kernels.cu:216-218."""
        offs = sorted(
            int(msa_swizzle4x4_offset(t, s, M3_SCALE_COLS))
            for t in range(block_size)
            for s in range(M3_SCALE_COLS)
        )
        region = block_size * M3_SCALE_COLS
        is_bijection = offs == list(range(region))
        assert is_bijection == (block_size % 4 == 0), (
            f"block_size={block_size}: swizzle bijectivity should track "
            "block_size % 4 == 0"
        )
        if block_size % 4 != 0:
            assert max(offs) >= region  # the tail quad reads outside the region

    @pytest.mark.parametrize("block_size", [1, 2, 3, 4, 5, 127, 128])
    def test_linear_layout_has_no_quad_constraint(self, block_size):
        """K is linear, so it is a bijection at any block_size.  The quad rule
        is a V-only constraint -- worth stating, because the two are easily
        conflated into one global 'block_size % 4' myth."""
        offs = sorted(
            int(msa_linear_offset(t, s, M3_SCALE_COLS))
            for t in range(block_size)
            for s in range(M3_SCALE_COLS)
        )
        assert offs == list(range(block_size * M3_SCALE_COLS))

    def test_head_dim_128_keeps_4x4_free_for_the_reader(self):
        """4x4 costs the reader nothing only while a warp spans whole token
        quads: 32 lanes / S columns >= 4 tokens, i.e. head_dim <= 128 (plan
        section 3).  head_dim != 128 is rejected outright at
        interface.py:323-326 and kernel :78-81, so the degradation is
        unreachable -- but this is the boundary to re-check if D moves."""
        assert M3_HEAD_DIM == 128
        assert M3_SCALE_COLS == 8
        for head_dim, tokens_per_warp in ((64, 8), (128, 4), (256, 2), (512, 1)):
            s = head_dim // 16
            assert 32 // s == tokens_per_warp
            assert (tokens_per_warp >= 4) == (head_dim <= 128)

    def test_region_is_exactly_1024_bytes_with_no_padding(self):
        """Both surviving layouts pack a region as exactly ``page_size * S``
        bytes.  128x4 needed ``round_up(rows,128) x round_up(cols,4)``; nothing
        here does."""
        assert M3_REGION_BYTES == M3_BLOCK_SIZE * M3_SCALE_COLS == 1024
        for fn in _LAYOUTS.values():
            offs = _all_offsets(fn)
            assert offs.min() == 0 and offs.max() == M3_REGION_BYTES - 1
            assert len(set(offs.tolist())) == M3_REGION_BYTES
