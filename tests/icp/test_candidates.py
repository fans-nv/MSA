"""The local-candidate selector: the band split, the ladder, and the arms.

Three things are gated here, in three tiers.

**Tier 1, no torch and no GPU.** The Python side of the selector is a mirror of
things that live in CUDA -- the ladder, the partition and thread constants, the
``cap`` codes -- and a mirror that drifts is a silent wrong answer, because
every arm is bit-exact against every other and no output comparison can see a
dispatch that quietly took a different one. So these tests read
``csrc/*.cu``/``*.cuh`` and assert the Python agrees. They also gate the band
split and the duck-typed metadata binding, which are properties of the *types*
and need no device.

**Tier 2, ``@pytest.mark.gpu``.** Bit-identity between the arms, the output
contract, head order, global id formation and the dispatch identity.

**Tier 3, ``@pytest.mark.gpu``, the negative controls.** A separate extension
built with ``ICP_RS_NEGATIVE_CONTROL`` defined, each control perturbing exactly
one site. A gate nobody has seen fail is not a gate.

WHAT refined-icp-v1 CHANGED HERE
--------------------------------

* **The geometry.** ``CandidateGeometry`` is now ``local_valid_blocks``,
  ``forced_column``, ``active_rows`` and a host ``scan_block_begin``.
  ``diagonal_owned`` and ``diagonal_local_ordinal`` are gone: fragment placement
  makes every rank an owner, so there is nothing left to own.
* **The global id is ``scan_block_begin + column``**, with no ``* world + rank``
  term. Every id test below therefore runs at a **non-zero
  ``scan_block_begin``**, for exactly the reason the rank tests used to run at
  ``rank != 0``: at 0 a dropped scan offset is the identity and the test could
  not fail.
* **Forcing is exclusion, not ``+inf``.** The forced column is removed from
  ordinary ranking on every rank and emits C4's invalid record; the *receiver*
  reserves a slot and injects the block once. So the selector's own output must
  NOT contain the forced block -- the inverse of what this file used to assert.
* **``live_local_blocks`` is gone**, replaced by the rank-independent
  ``live_blocks(max_kv_len, page_size=...)``.

WHAT THE NONBLOCKING POLICY CHANGED HERE (N2, 2026-09-13)
---------------------------------------------------------

The prefill entry point used to prove its ``live_blocks`` argument bounded the
device counts, by reading ``local_valid_blocks.max()`` into a Python integer on
every launch. Checks may no longer read a CUDA value into Python on the
submission path, and being eager is not an exemption. The extent is now
``PrefillPlan.scan_extent`` -- ``max_local_blocks``, fixed at startup -- so the
bound cannot be under-reported and there is nothing to check.

Three tests carry what the removed refusal used to carry:

* ``test_an_under_reported_live_extent_can_no_longer_truncate`` -- the inverted
  replacement for ``test_prefill_refuses_an_under_reported_live_extent``. An
  under-report must now be INERT (same 16 candidates), and the launch identity
  must show the geometry did not shrink with it.
* ``test_the_prefill_submission_path_reads_no_device_value`` -- every route from
  a CUDA tensor to a Python value raises, and the selector still runs.
* ``test_a_launch_that_does_not_cover_the_row_traps_on_the_device`` -- the
  backstop, driven through the extension in its own process, because a fired
  device assertion poisons the context.

And two habits, both of which have produced a test that could not fail:

* ``torch.equal`` on the fp32 view of a candidate tensor is broken.
  ``int32(-1)`` is ``0xFFFFFFFF``, which bitcasts to a negative quiet NaN, so it
  reports a mismatch on every invalid slot of two byte-identical buffers --
  false alarms on short rows, and it never reports a *match*. Compare
  ``.view(torch.int32)``.
* **Head order is rank-major, which is invisible at ``H_local == 1``.** A
  mutation that permutes heads is provably the identity there, so every head
  test below runs at ``H_local == 2``.

And one more: poison every output buffer before every launch and assert zero
survivors, or a comparison of two untouched buffers passes vacuously.

MUTATION CHECK, tier 1, all runnable with no torch::

    LADDER_RUNGS drop 32          -> test_ladder_mirrors_every_copy_in_csrc
    ladder_items `<=` -> `<`      -> test_ladder_mirrors_every_copy_in_csrc
    LOCAL_THREADS = 64            -> test_launch_constants_mirror_the_header
    PARTITION_BLOCKS = 2048       -> test_launch_constants_mirror_the_header
    SelectArm.CAPPED16 = 32       -> test_arm_values_are_the_kernel_cap_codes
    selector_arm falls back       -> test_an_unknown_arm_is_refused_by_name
    drop the prefill isinstance   -> test_a_plan_of_the_wrong_band_is_refused
    give decode `out` a default   -> test_decode_has_no_allocating_form
    drop the decode T check       -> test_decode_pins_t_to_the_captured_shape
    DecodePlan.band = "prefill"   -> test_a_plan_of_the_wrong_band_is_refused
    drop the unbanded refusal     -> test_an_unbanded_plan_cannot_be_built
    ladder_extent -> a constant   -> test_decode_takes_its_rung_from_capacity
    live_blocks keeps a rank term -> test_live_blocks_is_rank_independent
    _V1_REQUIRED loses a name     -> test_from_metadata_names_every_missing_field
"""

from __future__ import annotations

import pathlib
import re

import pytest

import fmha_sm100.icp
from fmha_sm100.icp.candidates import (
    CAND_K,
    LADDER_RUNGS,
    LOCAL_THREADS,
    MAX_PREFILL_PARTITIONS,
    MEASUREMENT_ARMS,
    PARTITION_BLOCKS,
    PARTITIONED_ITEMS,
    SHIPPED_ARM,
    CandidateGeometry,
    DecodePlan,
    PrefillPlan,
    SelectArm,
    SelectorPlan,
    allocate_workspace,
    ladder_items,
    live_blocks,
    live_local_blocks,
    prefill_launch,
    prefill_scan_coverage,
    select_decode_candidates,
    select_prefill_candidates,
    selector_arm,
)

CSRC = pathlib.Path(fmha_sm100.icp.__file__).resolve().parent / "csrc"
INF = float("inf")

#: Every GPU case below runs at a NON-ZERO scan origin. The global id is
#: ``scan_block_begin + column``; at ``scan_block_begin == 0`` a kernel that
#: dropped the origin (negative control 3, and the obvious porting mistake from
#: the block-cyclic scheme) produces exactly the right answer, so a suite that
#: only ever scans from 0 cannot fail.
SCAN_BEGIN = 4096


def _source(name: str) -> str:
    return (CSRC / name).read_text()


def _strip_comments(text: str) -> str:
    """``//`` and ``/* */`` removed, so a *mention* is not read as a *use*.

    These sources document what refined-icp-v1 replaced, in terms, which is
    exactly the string a "the old formation is gone" check greps for.
    """
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


# --------------------------------------------------------------------------
# tier 1: the mirrors
# --------------------------------------------------------------------------


#: ``(source, regex)`` for each of the three copies of the ladder that ship: the
#: launcher's dispatch, the attribute query the dispatch gate compares against,
#: and the negative control's. Three copies is deliberate -- drift between them
#: is exactly what the gate exists to catch -- so the parser reads all three.
LADDER_COPIES = {
    "launcher": ("local_candidates.cu", r"SELECT_PARTITION\((\d+)\)"),
    "attribute query": (
        "local_candidates.cu",
        r"Q\(icp::local_candidates_radix_kernel<(\d+)>",
    ),
    # The control build drives three arms off ONE chain, so its rung action is
    # the macro parameter. `SELECT\(` is still accepted: that is what the chain
    # spelled before it was factored, and a parser that only knew the new
    # spelling would report a rolled-back file as "no ladder found".
    "negative control": (
        "local_candidates_rs_control.cu",
        r"(?:MACRO|SELECT)\((\d+)\)",
    ),
}


def _ladder(name: str) -> list[tuple[int, int]]:
    """``[(threshold, Items), ...]`` ascending, with the ``else`` arm last.

    The ``else`` arm carries no threshold, so it is keyed by ``None`` -- which
    is what lets the three chains be compared as plain lists.
    """
    source, action = LADDER_COPIES[name]
    text = _source(source)
    # `[\s\\]*` and not `\s*`: the control build's chain lives inside a
    # function-like macro, so every line ends in a continuation backslash, which
    # is not whitespace and would make this parser silently find nothing.
    rungs = [
        (int(bound), int(items))
        for bound, items in re.findall(
            rf"blocks <= (\d+)\)[\s\\]*\{{[\s\\]*{action}", text
        )
    ]
    fallback = re.search(rf"\}} else \{{[\s\\]*{action}", text)
    assert rungs, f"no {name} ladder found; this parser is stale, not the code"
    assert fallback, f"no {name} else-arm found"
    return [*rungs, (None, int(fallback.group(1)))]


def test_ladder_mirrors_every_copy_in_csrc():
    ladders = {name: _ladder(name) for name in LADDER_COPIES}
    launcher = ladders["launcher"]
    for name, rungs in ladders.items():
        assert rungs == launcher, f"the {name} ladder has drifted"

    thresholds = [bound for bound, _ in launcher if bound is not None]
    assert thresholds == sorted(thresholds), "the chain must be ascending"
    assert tuple(items for _, items in launcher) == LADDER_RUNGS

    # The Python mirror reproduces it exactly, boundary included -- which is
    # where an off-by-one `<` would hide.
    previous = 0
    for bound, items in launcher[:-1]:
        assert ladder_items(bound) == items
        assert ladder_items(previous + 1) == items
        previous = bound
    assert ladder_items(previous + 1) == launcher[-1][1]


def test_a_rung_always_covers_one_ctas_worth():
    # The rung's contract: 128 * Items must reach the extent it was chosen for,
    # or a CTA does not cover its own partition.
    for extent in [0, 1, 17, 127, 128, 129, 4095, 4096]:
        assert LOCAL_THREADS * ladder_items(extent) >= extent
    # Above the top rung the ladder saturates and the partitioning takes over.
    assert LOCAL_THREADS * LADDER_RUNGS[-1] == PARTITION_BLOCKS


def test_launch_constants_mirror_the_header():
    header = _source("local_candidates.cuh")
    assert re.search(rf"kLocalThreads = {LOCAL_THREADS};", header)
    assert re.search(rf"kLocalPartition = {PARTITION_BLOCKS};", header)
    assert re.search(rf"kTopK = {CAND_K};", _source("merge_topk.cuh"))


def test_arm_values_are_the_kernel_cap_codes():
    # `SelectArm`'s value IS the extension's `cap`, so the enum and the
    # TORCH_CHECK that admits a cap are one landing unit.
    check = re.search(
        r"TORCH_CHECK\((cap == .*?),\s*\n\s*\"cap must be",
        _source("local_candidates.cu"),
        re.S,
    )
    assert check, "the cap TORCH_CHECK moved; this parser is stale"
    accepted = {int(v) for v in re.findall(r"cap == (-?\d+)", check.group(1))}
    assert accepted == {int(a) for a in SelectArm}


def test_the_selector_forms_the_explicit_affine_global_id():
    """The physical profile is explicit; kernels do not infer it from rank.

    The Python side no longer *has* a rank to check here -- that is the point --
    so the mirror is against the source: the id must be formed from
    ``scan_block_begin`` and the admitted static stride, in every arm.
    """
    for name in ("local_candidates.cuh", "local_candidates_rs.cuh"):
        # Comment-stripped: both files DISCUSS the block-cyclic formation they
        # replaced, and a parser that could not tell code from prose would
        # report the explanation as the defect.
        text = _strip_comments(_source(name))
        assert "scan_block_begin + global_block_stride * local" in text, name
        # The old formation, in any spacing. Its absence is what makes
        # `scan_block_begin + column` the only id this kernel can emit.
        assert not re.search(r"local\s*\*\s*world\s*\+\s*rank", text), name
        assert not re.search(r"\bowns\s*\[", text), name


def test_the_shipped_arm_is_the_bounded_radix_select():
    assert SHIPPED_ARM is SelectArm.RADIX_BOUNDED
    assert SHIPPED_ARM not in MEASUREMENT_ARMS
    assert set(MEASUREMENT_ARMS) | {SHIPPED_ARM} == set(SelectArm)
    # The default with no environment override is the shipped arm.
    assert selector_arm() is SHIPPED_ARM


def test_an_unknown_arm_is_refused_by_name():
    # Never a silent fallback: every arm is bit-exact against every other, so a
    # fallback runs and times one arm under another's name and no output
    # comparison can catch it.
    with pytest.raises(ValueError, match="unknown selector arm"):
        selector_arm("radix_unbounded")
    with pytest.raises(ValueError, match="cap code"):
        selector_arm(7)
    assert selector_arm("radixsr") is SelectArm.RADIX_BOUNDED


def test_the_env_override_names_an_arm(monkeypatch):
    monkeypatch.setenv("ICP_SELECTOR", "capped8")
    assert selector_arm() is SelectArm.CAPPED8
    assert selector_arm(SelectArm.SORT) is SelectArm.SORT
    monkeypatch.setenv("ICP_SELECTOR", "nonsense")
    with pytest.raises(ValueError):
        selector_arm()


def test_full_row_controls_are_rejected_before_validation_or_build(monkeypatch):
    from types import SimpleNamespace

    from fmha_sm100.icp import candidates as c

    def must_not_run(*_args, **_kwargs):
        pytest.fail("unsupported controls reached tensor validation or JIT build")

    monkeypatch.setattr(c, "_validate", must_not_run)
    monkeypatch.setattr(c._build, "_select_control_ext", must_not_run)
    with pytest.raises(ValueError, match="RADIX_FULL_ROW.*mutation-control"):
        c.select_with_control(
            None,
            None,
            plan=_plan(DecodePlan),
            workspace=SimpleNamespace(partials=None),
            out=None,
            control=0,
            arm=SelectArm.RADIX_FULL_ROW,
        )


# --------------------------------------------------------------------------
# tier 1: the band split
# --------------------------------------------------------------------------


def _plan(cls, **kwargs):
    base = dict(
        icp_degree=4,
        icp_rank=1,
        num_heads_local=2,
        max_local_blocks=2048,
        token_capacity=8,
    )
    return cls(**{**base, **kwargs})


class _Shape:
    """Just enough of a tensor to reach the decode entry point's shape rule.

    The rule is checked before anything touches torch, and a stand-in keeps that
    gate in the tier that runs without torch installed.
    """

    def __init__(self, tokens: int) -> None:
        self.shape = (tokens, 8, 2048)


def test_an_unbanded_plan_cannot_be_built():
    with pytest.raises(TypeError, match="carries no band"):
        _plan(SelectorPlan)


def test_a_plan_of_the_wrong_band_is_refused():
    prefill, decode = _plan(PrefillPlan), _plan(DecodePlan)
    assert (prefill.band, decode.band) == ("prefill", "decode")
    with pytest.raises(TypeError, match="needs a PrefillPlan"):
        select_prefill_candidates(_Shape(8), None, plan=decode, live_blocks=0)
    with pytest.raises(TypeError, match="needs a DecodePlan"):
        select_decode_candidates(
            _Shape(8), None, plan=prefill, workspace=None, out=None
        )


def test_a_workspace_cannot_cross_the_band():
    # Dataclass equality is class-sensitive, which is what makes the plan check
    # inside the decode entry point reject a workspace allocated for the other
    # band -- the fields are identical here and the objects still differ.
    assert _plan(PrefillPlan) != _plan(DecodePlan)
    assert _plan(PrefillPlan) == _plan(PrefillPlan)


def test_the_workspace_is_decodes_alone():
    # It exists to keep allocation out of a capture, and prefill is never
    # captured, so a prefill plan must not be able to allocate one. This guard
    # is correct and is NOT weakened anywhere below: every GPU test in this file
    # either takes a DecodePlan and a workspace, or takes a PrefillPlan and the
    # prefill entry point, which has no workspace at all.
    with pytest.raises(TypeError, match="decode's"):
        allocate_workspace(_plan(PrefillPlan), "cuda")


def test_decode_has_no_allocating_form():
    # A tensor allocated inside a capture is freed when the capture's pool is
    # reset, so the graph would replay against memory handed to something else.
    # The decode entry point therefore has no default for `out`.
    import inspect

    decode = inspect.signature(select_decode_candidates).parameters["out"]
    prefill = inspect.signature(select_prefill_candidates).parameters["out"]
    assert decode.default is inspect.Parameter.empty
    assert prefill.default is None


def test_decode_pins_t_to_the_captured_shape():
    with pytest.raises(ValueError, match="token_capacity"):
        select_decode_candidates(
            _Shape(7), None, plan=_plan(DecodePlan), workspace=None, out=None
        )


def test_both_entry_points_are_keyword_only_past_geometry():
    # The live-extent argument has to be addable without breaking a caller.
    import inspect

    for fn in (select_prefill_candidates, select_decode_candidates):  # noqa: E501
        kinds = [p.kind for p in inspect.signature(fn).parameters.values()]
        assert kinds[:2] == [inspect.Parameter.POSITIONAL_OR_KEYWORD] * 2
        assert all(k is inspect.Parameter.KEYWORD_ONLY for k in kinds[2:])


# --------------------------------------------------------------------------
# tier 1: the plan
# --------------------------------------------------------------------------


def test_decode_takes_its_rung_from_capacity():
    plan = _plan(DecodePlan, max_local_blocks=2048)
    assert plan.ladder_extent == plan.max_local_blocks
    assert plan.planned_items == ladder_items(2048)


def test_prefill_takes_its_whole_geometry_from_the_startup_extent():
    """USED TO ASSERT that the rung and the partition count came from the LIVE
    extent, and that "sizing either from capacity is the defect the eager path
    exists to remove".

    N2 (2026-09-13) reverses that half. Sizing the launch from per-step data is
    what forced the entry point to prove the per-step value bounded the device
    counts -- ``int(local_valid_blocks.max())``, a submission-path
    synchronisation the nonblocking policy forbids on either band. The extent is
    now ``scan_extent``, i.e. ``max_local_blocks``, which a serving runtime fixes
    at startup from ``max_model_len``.

    ``launch()`` itself is unchanged and still parameterised: it is the mirror of
    ``icp::prefill_launch`` that ``planned_prefill_kernel`` and the ladder gates
    compare against, and the rungs it names are the rungs the extension builds.
    What changed is the argument the entry point feeds it.
    """
    plan = _plan(PrefillPlan, max_local_blocks=8 * PARTITION_BLOCKS)
    assert plan.scan_extent == plan.max_local_blocks
    assert plan.launch(plan.scan_extent) == prefill_launch(plan.max_local_blocks)
    assert plan.partial_shape(4, plan.scan_extent)[2] == 8
    # The mirror, at extents the entry point no longer passes.
    assert plan.launch(1) == prefill_launch(1)
    assert plan.launch(1).items == 1 and plan.launch(1).partitions == 0
    assert plan.launch(PARTITION_BLOCKS).partitions == 0
    assert plan.launch(PARTITION_BLOCKS + 1).partitions == 2
    assert plan.launch(PARTITION_BLOCKS + 1).items == PARTITIONED_ITEMS
    assert not hasattr(plan, "ladder_extent")
    assert not hasattr(plan, "partial_count")


def test_the_startup_extent_covers_every_row_the_plan_admits():
    """Why a startup extent needs no runtime check, in one inequality.

    The kernel clamps every row to ``max_local_blocks``, and a launch sized from
    ``max_local_blocks`` scans at least that many columns, so no metadata can
    name a row the launch does not cover. The second half of the test is the
    discriminating one: a SHORTER extent does not cover a longer row, by both
    mechanisms -- too short a rung, and too few partitions -- which is the
    silent truncation the old synchronising check existed to catch and the
    reason the extent is not per-step.
    """
    for capacity in (
        1,
        17,
        127,
        128,
        300,
        2048,
        4096,
        4097,
        8192,
        3 * PARTITION_BLOCKS + 5,
    ):
        assert prefill_scan_coverage(capacity) >= capacity
        assert prefill_scan_coverage(capacity) >= prefill_launch(capacity).items
    # Rung too short: 100 live columns buy a 128-column CTA, and a row of 300 is
    # cut at 128.
    assert prefill_scan_coverage(100) == LOCAL_THREADS < 300
    # Partition count too short, which the rung check alone cannot see: every
    # partition launched is fully spanned and the row still runs past the last.
    assert prefill_scan_coverage(PARTITION_BLOCKS) == PARTITION_BLOCKS < 9000


def test_both_arms_trap_a_launch_that_does_not_cover_the_row():
    """The device-side half of the N2 bound, mirrored from the source.

    With the extent a startup constant neither assertion can fire from the
    shipping entry points -- that is the point of the fix. They are gated here
    so that the invariant stays stated on the device: a future launcher that
    re-derives the extent from per-step data traps instead of quietly dropping
    columns. Comment-stripped, so the paragraphs explaining them do not pass for
    the assertions themselves.
    """
    for name in ("local_candidates.cuh", "local_candidates_rs.cuh"):
        text = _strip_comments(_source(name))
        assert re.search(
            r"assert\(\s*min\([^;]*kLocalPartition\)\s*<=\s*kLocalThreads\s*\*\s*Items\)",
            text,
        ), f"{name}: the rung-spans-its-partition trap is gone"
        assert re.search(
            r"assert\(static_cast<int64_t>\(valid\)\s*<=\s*"
            r"static_cast<int64_t>\(partitions\)\s*\*\s*kLocalPartition\)",
            text,
        ), f"{name}: the partitions-span-the-row trap is gone"


def test_a_partitioned_prefill_rung_is_the_one_that_spans_a_partition():
    # Cutting it drops the tail of every partition with no other symptom.
    assert PARTITIONED_ITEMS * LOCAL_THREADS == PARTITION_BLOCKS
    for live in (PARTITION_BLOCKS + 1, 3 * PARTITION_BLOCKS, 255 * PARTITION_BLOCKS):
        assert prefill_launch(live).items == PARTITIONED_ITEMS


def test_prefill_refuses_a_partition_count_the_combine_cannot_stream():
    plan = _plan(
        PrefillPlan, max_local_blocks=(MAX_PREFILL_PARTITIONS + 1) * PARTITION_BLOCKS
    )
    plan.launch(MAX_PREFILL_PARTITIONS * PARTITION_BLOCKS)
    with pytest.raises(ValueError, match="at most"):
        plan.launch(MAX_PREFILL_PARTITIONS * PARTITION_BLOCKS + 1)


def test_prefill_refuses_a_live_extent_outside_capacity():
    plan = _plan(PrefillPlan, max_local_blocks=2048)
    for bad in (-1, 2049):
        with pytest.raises(ValueError, match="live_blocks"):
            plan.launch(bad)


def test_prefill_has_an_explicit_arm_and_no_decode_workspace():
    # Prefill keeps its typed eager API and retained partials. Its new explicit
    # whole-row opt-in must not silently replace the bounded default.
    import inspect

    parameters = inspect.signature(select_prefill_candidates).parameters
    assert parameters["arm"].default is SHIPPED_ARM
    assert "workspace" not in parameters and "live_blocks" in parameters


def test_live_blocks_is_rank_independent():
    """USED TO ASSERT that a per-rank block-cyclic count was dominated by a
    per-rank bound, via ``live_local_blocks(len, icp_degree=, icp_rank=)``.

    That whole shape is gone. Under fragment placement every rank holds
    ``R = 128/W`` rows of every logical block and scores the **full** global
    block domain, so the reachable column count is ``ceil(len / page_size)`` on
    every rank -- identical across ranks rather than differing by one, and ``W``
    times what the block-cyclic count gave. The bound therefore takes neither
    ``icp_degree`` nor ``icp_rank``, and the new assertion is that it dominates
    every row at the batch's longest sequence.
    """
    for page in (64, 128):
        for max_len in (1, 127, 128, 129, 4096, 131072):
            bound = live_blocks(max_len, page_size=page)
            assert bound == -(-max_len // page)
            # Monotone in the sequence length, so the value at the maximum
            # bounds every shorter row in the batch -- which is what
            # `live_blocks` has to be for the launch it sizes.
            for length in (1, max_len // 2, max_len):
                assert -(-length // page) <= bound
    assert live_blocks(0, page_size=128) == 0
    with pytest.raises(ValueError):
        live_blocks(-1, page_size=128)
    with pytest.raises(ValueError):
        live_blocks(128, page_size=0)


def test_live_local_blocks_is_a_raising_stub_and_names_its_replacement():
    # The replacement takes FEWER arguments, so a caller that dropped
    # icp_degree/icp_rank but kept the name would bind to nothing and fail
    # somewhere less obvious, and a caller that kept them would silently compute
    # a count that is now W times too small.
    with pytest.raises(RuntimeError, match="live_blocks"):
        live_local_blocks(131072, icp_degree=4, icp_rank=1, page_size=128)
    with pytest.raises(RuntimeError, match="block-cyclic"):
        live_local_blocks(131072)


def test_head_geometry_is_rank_major():
    plan = _plan(PrefillPlan, icp_degree=4, icp_rank=3, num_heads_local=2)
    assert plan.num_heads_group == 8
    # The rank no longer enters the global block id, but it still pins the HEAD
    # window, where a rotation is a silent wrong answer.
    assert plan.head_offset == 6


def test_partitioning_starts_above_one_ctas_worth():
    assert _plan(DecodePlan, max_local_blocks=PARTITION_BLOCKS).partial_count == 0
    assert not _plan(DecodePlan, max_local_blocks=PARTITION_BLOCKS).partitioned
    plan = _plan(DecodePlan, max_local_blocks=PARTITION_BLOCKS + 1)
    assert plan.partial_count == 2 and plan.partitioned
    assert plan.partial_shape[2] == 2


@pytest.mark.parametrize(
    "bad",
    [
        dict(icp_degree=1),
        dict(icp_degree=9),
        dict(icp_rank=4),
        dict(icp_rank=-1),
        dict(num_heads_local=0),
        dict(max_local_blocks=0),
        dict(token_capacity=0),
    ],
)
def test_the_plan_refuses_geometry_the_kernel_would_not_run(bad):
    with pytest.raises(ValueError):
        _plan(PrefillPlan, **bad)


def test_the_plan_refuses_ids_that_would_not_fit_int32():
    # USED TO PASS `max_local_blocks=(1<<31)//4, icp_degree=8`, because the id
    # was `local * icp_degree + rank` and the degree was part of the bound.
    # refined C1 makes the id `scan_block_begin + column`, so the plan can only
    # bound the WINDOW's own width; the full bound needs the launch's scan
    # origin and is re-checked host-side in `local_candidates.cu`.
    with pytest.raises(ValueError, match="int32"):
        _plan(PrefillPlan, max_local_blocks=(1 << 31) + 1)


def test_the_plan_is_frozen():
    plan = _plan(PrefillPlan)
    with pytest.raises(Exception):
        plan.max_local_blocks = 1


# --------------------------------------------------------------------------
# tier 1: the duck-typed metadata binding
# --------------------------------------------------------------------------


class _Metadata:
    """A stand-in for the serving stack's attention metadata object.

    Duck-typed on purpose -- this package has no serving-stack dependency -- so
    the integration point is four attribute names and the failure mode of a
    stale builder is an ``AttributeError``, by design and not by accident.
    """

    def __init__(self, **attrs):
        for name, value in attrs.items():
            setattr(self, name, value)


def _v1_metadata(**overrides):
    attrs = dict(
        topk_num_valid_pages="counts",
        icp_forced_column_v1="columns",
        icp_scan_block_begin_v1=4096,
        icp_active_rows="active",
    )
    attrs.update(overrides)
    return _Metadata(**attrs)


def test_from_metadata_binds_the_v1_names():
    geometry = CandidateGeometry.from_metadata(_v1_metadata())
    assert geometry.local_valid_blocks == "counts"
    assert geometry.forced_column == "columns"
    assert geometry.active_rows == "active"
    # A host int, not a device value: the scan origin decides the ids and must
    # be readable without a synchronisation.
    assert geometry.scan_block_begin == 4096
    assert isinstance(geometry.scan_block_begin, int)


@pytest.mark.parametrize(
    "missing",
    [
        "topk_num_valid_pages",
        "icp_forced_column_v1",
        "icp_scan_block_begin_v1",
        "icp_active_rows",
    ],
)
def test_from_metadata_names_every_missing_field(missing):
    """A pre-refinement builder must fail HERE, by name.

    The names are versioned (``_v1``) because the meaning of the diagonal
    changed without its dtype or its shape changing. Keeping the old names would
    let an unmigrated builder bind cleanly and feed block-cyclic local ordinals
    into a fragment-placement kernel: wrong selections, no error anywhere.
    """
    attrs = {k: v for k, v in vars(_v1_metadata()).items() if k != missing}
    with pytest.raises(AttributeError) as excinfo:
        CandidateGeometry.from_metadata(_Metadata(**attrs))
    assert missing in str(excinfo.value)
    assert "refined-icp-v1" in str(excinfo.value)


def test_from_metadata_refuses_a_builder_that_still_owns_a_diagonal():
    # Mixing the two generations is the exact silent mis-selection the rename
    # exists to prevent: this object has every v1 name AND the retired one, so
    # it would otherwise bind and run.
    stale = _v1_metadata(icp_owns_diagonal="owned")
    with pytest.raises(AttributeError, match="icp_owns_diagonal"):
        CandidateGeometry.from_metadata(stale)


def test_the_geometry_has_no_diagonal_fields_left():
    # The rename is the whole migration story, so assert the old names are gone
    # from the type rather than only from the binder.
    fields = set(CandidateGeometry.__dataclass_fields__)
    assert fields == {
        "local_valid_blocks",
        "forced_column",
        "active_rows",
        "scan_block_begin",
        "global_block_stride",
    }
    assert "diagonal_owned" not in fields
    assert "diagonal_local_ordinal" not in fields


# --------------------------------------------------------------------------
# tier 2: the kernel. GPU.
# --------------------------------------------------------------------------


def _make_case(
    plan,
    *,
    tokens,
    live,
    device,
    seed=0,
    all_active=True,
    scan_block_begin=SCAN_BEGIN,
    forced_column=None,
):
    """Scores plus geometry for one launch, with a forced column on half the rows.

    ``forced_column`` defaults to ``live // 2`` rather than to ``live - 1``: the
    kernel's own negative control 2 replaces ``forced_col[token]`` with
    ``valid - 1``, and a case whose forced column IS ``valid - 1`` makes that
    control provably vacuous. Odd rows carry ``-1``, which C3 says is legal and
    means "the forced block is not in this scan window", so both branches of the
    exclusion are exercised in one launch.
    """
    import torch

    g = torch.Generator(device=device).manual_seed(seed)
    scores = torch.randn(
        (tokens, plan.num_heads_group, plan.max_local_blocks),
        device=device,
        generator=g,
        dtype=torch.float32,
    )
    nvalid = torch.full((tokens,), int(live), dtype=torch.int32, device=device)
    column = int(live) // 2 if forced_column is None else int(forced_column)
    forced = torch.full((tokens,), -1, dtype=torch.int32, device=device)
    forced[::2] = column
    active = torch.ones((tokens,), dtype=torch.bool, device=device)
    if not all_active:
        active[1::3] = False
    return scores, CandidateGeometry(
        local_valid_blocks=nvalid,
        forced_column=forced,
        active_rows=active,
        scan_block_begin=scan_block_begin,
    )


def _expected_valid_count(geometry, *, token: int, live: int) -> int:
    """How many ids row ``token`` must emit.

    ``min(16, live)`` less the forced column, which C3 excludes from ordinary
    ranking on EVERY rank and which the receiver injects instead.
    """
    if not bool(geometry.active_rows[token]):
        return 0
    column = int(geometry.forced_column[token])
    excluded = 1 if 0 <= column < live else 0
    return min(CAND_K, live - excluded)


#: A bit pattern no candidate can produce: score bits 0x5A5A5A5A with an id
#: whose complement is not a legal key. Surviving it means the kernel did not
#: write that slot, which is what makes "the two buffers agree" mean anything.
POISON = 0x5A5A5A5A


def _poisoned(shape, device):
    import torch

    return torch.full(shape, POISON, dtype=torch.int32, device=device).view(
        torch.float32
    )


def _assert_no_survivors(out):
    import torch

    assert not (out.view(torch.int32) == POISON).any(), (
        "a slot of `out` was never written; any comparison against it is vacuous"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("arm", list(SelectArm), ids=lambda a: a.name.lower())
@pytest.mark.parametrize("live", [1, 15, 16, 130, 520, 2048])
def test_every_arm_agrees_with_the_shipped_one(cuda_device, arm, live):
    """All measurement arms remain reachable on the decode path.

    USED TO BUILD a ``PrefillPlan``, allocate a workspace from it and pass
    ``arm=`` to ``select_prefill_candidates``. All three are wrong now and two
    of them always were: the workspace is decode's (it exists to keep allocation
    out of a capture, and ``allocate_workspace`` refuses a ``PrefillPlan`` for
    exactly that reason). Prefill now admits explicit bounded/full-row choices,
    while the other measurement arms remain decode-only. This gate still
    exercises the fixed-shape decode API.
    """
    import torch

    tokens = 8
    plan = _plan(DecodePlan, token_capacity=tokens, max_local_blocks=2048)
    workspace = allocate_workspace(plan, cuda_device)
    scores, geometry = _make_case(plan, tokens=tokens, live=live, device=cuda_device)

    shape = (tokens, plan.num_heads_group, CAND_K, 2)
    ref = _poisoned(shape, cuda_device)
    got = _poisoned(shape, cuda_device)
    select_decode_candidates(
        scores, geometry, plan=plan, workspace=workspace, out=ref, arm=SHIPPED_ARM
    )
    select_decode_candidates(
        scores, geometry, plan=plan, workspace=workspace, out=got, arm=arm
    )
    _assert_no_survivors(ref)
    _assert_no_survivors(got)

    # The id half is exact for every arm. The score half is exact too, except
    # that SORT preserves -0.0 where the radix arms emit +0.0 (C5 mandates the
    # flush), so it is compared with a flush applied to both sides.
    assert torch.equal(ref.view(torch.int32)[..., 1], got.view(torch.int32)[..., 1])
    assert torch.equal(ref[..., 0] + 0.0, got[..., 0] + 0.0)


@pytest.mark.gpu
@pytest.mark.parametrize("live", [1, 15, 16, 130])
def test_the_output_contract_holds(cuda_device, live):
    """C4/C6 on the eager prefill path.

    USED TO ASSERT ``x % icp_degree == icp_rank`` for every emitted id -- C1's
    block-cyclic ownership. There is no such residue any more: the id is
    ``scan_block_begin + column`` and is rank-independent, so what replaces it is
    a window check plus the count the forced-column exclusion implies.
    """
    import torch

    tokens = 8
    plan = _plan(PrefillPlan, token_capacity=tokens, max_local_blocks=2048)
    scores, geometry = _make_case(
        plan, tokens=tokens, live=live, device=cuda_device, all_active=False
    )
    out = _poisoned((tokens, plan.num_heads_group, CAND_K, 2), cuda_device)
    select_prefill_candidates(scores, geometry, plan=plan, live_blocks=live, out=out)
    _assert_no_survivors(out)

    ids = out.view(torch.int32)[..., 1].cpu()
    scores_out = out[..., 0].cpu()
    for t in range(tokens):
        want_count = _expected_valid_count(geometry, token=t, live=live)
        forced_id = SCAN_BEGIN + int(geometry.forced_column[t])
        for h in range(plan.num_heads_group):
            row = ids[t, h].tolist()
            valid = [x for x in row if x >= 0]
            assert row[: len(valid)] == valid, "the -1s must be a tail"
            assert len(set(valid)) == len(valid), "no duplicate global ids"
            # refined C1: every emitted id names a column of THIS scan window.
            assert all(SCAN_BEGIN <= x < SCAN_BEGIN + live for x in valid)
            assert len(valid) == want_count
            # refined C3: the forced column is EXCLUDED here and injected by the
            # receiver, so the producer must not emit it.
            if int(geometry.forced_column[t]) >= 0:
                assert forced_id not in valid
            # C4: an invalid slot is exactly (-inf, -1).
            for k in range(len(valid), CAND_K):
                assert scores_out[t, h, k] == -INF


@pytest.mark.gpu
def test_the_forced_column_is_excluded_on_every_rank(cuda_device):
    """C3, and it INVERTS what this file used to assert.

    USED TO ASSERT that the diagonal was forced INTO the output by its owner and
    by nobody else (``present == diagonal_owned[t]``), via a ``+inf`` score.
    refined-icp-v1 removes both halves: there is no owner, because every rank
    holds a fragment of every block; and forcing by ``+inf`` is gone, because a
    legitimate ordinary ``+inf`` with a smaller id outranks the forced block
    under C5 and displaces it. The forced column is now removed from ordinary
    ranking on EVERY rank and the receiver injects the block exactly once --
    which is gated in ``test_merge_key.py``.

    So: the forced column carries the BEST score in its row and must still be
    absent, and the two-sided control is the same launch with
    ``forced_column = -1``, where it must be present. Without that second half
    the test would pass against a kernel that emitted nothing at all.
    """
    import torch

    tokens, live = 8, 520
    plan = _plan(PrefillPlan, token_capacity=tokens, max_local_blocks=2048)
    column = live // 2
    scores, geometry = _make_case(
        plan, tokens=tokens, live=live, device=cuda_device, forced_column=column
    )
    # The forced column is the best block in the row by a mile: only the
    # exclusion can keep it out.
    scores[:, :, column] = 1e30
    forced_id = SCAN_BEGIN + column

    out = _poisoned((tokens, plan.num_heads_group, CAND_K, 2), cuda_device)
    select_prefill_candidates(scores, geometry, plan=plan, live_blocks=live, out=out)
    _assert_no_survivors(out)
    ids = out.view(torch.int32)[..., 1].cpu()

    # The two-sided control: the same scores with no forced column at all.
    unforced = CandidateGeometry(
        local_valid_blocks=geometry.local_valid_blocks,
        forced_column=torch.full_like(geometry.forced_column, -1),
        active_rows=geometry.active_rows,
        scan_block_begin=geometry.scan_block_begin,
    )
    control = _poisoned((tokens, plan.num_heads_group, CAND_K, 2), cuda_device)
    select_prefill_candidates(
        scores, unforced, plan=plan, live_blocks=live, out=control
    )
    _assert_no_survivors(control)
    control_ids = control.view(torch.int32)[..., 1].cpu()

    for t in range(tokens):
        excluded = int(geometry.forced_column[t]) >= 0
        for h in range(plan.num_heads_group):
            # PRECONDITION: with no exclusion this block MUST be selected in
            # EVERY head, or "absent" below would be true of any kernel at all.
            assert forced_id in control_ids[t, h].tolist(), (
                f"row {t} head {h}: the best-scoring block was not selected "
                "even without the exclusion, so this case cannot observe C3"
            )
            present = forced_id in ids[t, h].tolist()
            assert present is not excluded, (
                f"row {t} head {h}: C3 excludes the forced column from ordinary "
                "ranking, on every rank, and the receiver injects it instead"
            )


@pytest.mark.gpu
def test_head_order_is_rank_major(cuda_device):
    # H_local == 2, because at H_local == 1 a head permutation is the identity
    # and this test would be unable to fail.
    import torch

    tokens = 4
    plan = _plan(
        PrefillPlan,
        icp_degree=2,
        icp_rank=1,
        num_heads_local=2,
        token_capacity=tokens,
        max_local_blocks=512,
    )
    assert plan.num_heads_group == 4 and plan.head_offset == 2
    scores, geometry = _make_case(
        plan, tokens=tokens, live=200, device=cuda_device, forced_column=-1
    )
    # Give every head a different winner, so swapping two heads is observable.
    for h in range(plan.num_heads_group):
        scores[:, h, :] = -1.0
        scores[:, h, h * 7] = 10.0
    out = _poisoned((tokens, plan.num_heads_group, CAND_K, 2), cuda_device)
    select_prefill_candidates(scores, geometry, plan=plan, live_blocks=200, out=out)
    _assert_no_survivors(out)
    ids = out.view(torch.int32)[..., 1].cpu()
    for h in range(plan.num_heads_group):
        best = SCAN_BEGIN + h * 7  # refined C1: no rank term in the id
        for t in range(tokens):
            # Per TOKEN row. `ids[:, h]` is [tokens, 16], so `best in
            # ids[:, h].tolist()` compares an int against a list OF LISTS and is
            # False for every input -- a test that could only ever fail, which
            # is as useless as one that could only ever pass.
            row = ids[t, h].tolist()
            assert best in row, f"head {h}, token {t} lost its own winner {best}: {row}"
    # The heads must not all be answering with the same row either, or the
    # permutation this test exists to detect would be invisible. Heads 0 and 1
    # put their winner inside the first 16 columns, where the -1.0 tie fills the
    # same slots anyway; heads 2 and 3 put it at column 14 and 21, and 21 is
    # outside that window, so head 3's row MUST differ from head 0's.
    assert ids[0, 3].tolist() != ids[0, 0].tolist(), (
        "every head returned the same ids, so a head permutation would be "
        "invisible and this test would be vacuous"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("scan_begin", [128, 4096, 1 << 20])
def test_the_global_id_is_the_scan_origin_plus_the_column(cuda_device, scan_begin):
    """refined C1, at a NON-ZERO origin.

    USED TO BE ``test_the_global_id_carries_this_rank``, asserting
    ``id % icp_degree == icp_rank`` and parametrised over ``rank in (1, 2, 3)``
    because a rank rotation is invisible at rank 0. The id no longer carries the
    rank at all; the thing that is invisible at 0 now is the SCAN ORIGIN, so
    that is what is parametrised, and 0 is deliberately not in the list.
    """
    import torch

    tokens, live = 4, 300
    plan = _plan(
        PrefillPlan,
        icp_degree=4,
        icp_rank=2,
        token_capacity=tokens,
        max_local_blocks=512,
    )
    scores, geometry = _make_case(
        plan, tokens=tokens, live=live, device=cuda_device, scan_block_begin=scan_begin
    )
    out = _poisoned((tokens, plan.num_heads_group, CAND_K, 2), cuda_device)
    select_prefill_candidates(scores, geometry, plan=plan, live_blocks=live, out=out)
    _assert_no_survivors(out)
    ids = out.view(torch.int32)[..., 1].cpu()
    live_ids = ids[ids >= 0]
    assert live_ids.numel() > 0
    assert (live_ids >= scan_begin).all(), (
        "an id below the scan origin means the origin was dropped -- the exact "
        "block-cyclic porting mistake negative control 3 reproduces"
    )
    assert (live_ids < scan_begin + live).all()
    # And the ids are the columns of THIS window, not a rank-strided subset of
    # them: with 300 live columns and 16 winners per row, at least two winners
    # must be adjacent-ish rather than congruent to one residue class.
    residues = {int(x) % plan.icp_degree for x in live_ids.tolist()}
    assert len(residues) > 1, (
        "every selected id shares one residue mod icp_degree, which is what "
        "block-cyclic ownership produced and refined C1 does not"
    )


@pytest.mark.gpu
def test_the_dispatch_launches_the_kernel_it_says_it_does(cuda_device):
    # All arms are bit-exact, so no output comparison can see a dispatch that
    # quietly took a different one. Compare the launched kernel's identity
    # against the second, independent copy of the ladder in the extension.
    # Decode, because `planned_kernel` mirrors the DECODE launcher's ladder.
    from fmha_sm100.icp.candidates import last_launch, planned_kernel, record_launches

    tokens = 4
    plan = _plan(DecodePlan, token_capacity=tokens, max_local_blocks=512)
    workspace = allocate_workspace(plan, cuda_device)
    scores, geometry = _make_case(plan, tokens=tokens, live=300, device=cuda_device)
    out = _poisoned((tokens, plan.num_heads_group, CAND_K, 2), cuda_device)
    # The native recorder is process-wide and earlier tests may have used it.
    # Assert this call's increment, while keeping the kernel-identity check.
    before = last_launch()["count"]
    record_launches(True)
    try:
        select_decode_candidates(
            scores, geometry, plan=plan, workspace=workspace, out=out
        )
        launched = last_launch()
    finally:
        record_launches(False)
    assert launched["count"] == before + 1, "expected exactly one recorded launch"
    expected = planned_kernel(SHIPPED_ARM, plan.max_local_blocks)
    assert launched["symbol"] == expected["symbol"]
    assert launched["regs"] == expected["regs"]


@pytest.mark.gpu
def test_the_decode_entry_point_is_capturable(cuda_device):
    import torch

    plan = _plan(DecodePlan, token_capacity=4, max_local_blocks=512)
    workspace = allocate_workspace(plan, cuda_device)
    scores, geometry = _make_case(plan, tokens=4, live=300, device=cuda_device)
    out = workspace.candidates

    select_decode_candidates(scores, geometry, plan=plan, workspace=workspace, out=out)
    torch.cuda.synchronize()
    eager = out.view(torch.int32).clone()

    graph = torch.cuda.CUDAGraph()
    out.view(torch.int32).fill_(POISON)
    with torch.cuda.graph(graph):
        select_decode_candidates(
            scores, geometry, plan=plan, workspace=workspace, out=out
        )
    graph.replay()
    torch.cuda.synchronize()
    _assert_no_survivors(out)
    assert torch.equal(out.view(torch.int32), eager)


@pytest.mark.gpu
def test_the_workspace_refuses_to_allocate_during_capture(cuda_device):
    import torch

    plan = _plan(DecodePlan, token_capacity=4, max_local_blocks=512)
    allocate_workspace(plan, cuda_device)  # warm the JIT outside the capture
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="before cudagraph capture"):
        with torch.cuda.graph(graph):
            allocate_workspace(plan, cuda_device)


# --------------------------------------------------------------------------
# tier 2: the eager prefill path. GPU.
# --------------------------------------------------------------------------


@pytest.mark.gpu
def test_the_extent_does_not_change_the_answer(cuda_device):
    """Every rung emits the same candidates. Driven at the EXTENSION.

    USED TO drive ``select_prefill_candidates`` twice with different
    ``live_blocks``. That comparison is vacuous now: N2 makes the entry point
    size both launches from ``scan_extent``, so the two calls would take one
    geometry and the test would assert that a launch equals itself.

    The claim itself is still real and still load-bearing -- it is why moving
    the extent from the live count to the startup capacity cannot change a
    single emitted tuple -- so it is made one level down, where the extent is
    still a parameter. Both extents here cover the row (520 live columns), which
    is the regime the rungs are bit-exact in; an extent that does NOT cover the
    row is the device trap's business, not this test's.
    """
    import torch

    from fmha_sm100.icp import _build

    plan = _plan(PrefillPlan, token_capacity=8, max_local_blocks=4096)
    scores, geometry = _make_case(plan, tokens=8, live=520, device=cuda_device)
    shape = (8, plan.num_heads_group, CAND_K, 2)

    def at_extent(extent):
        out = _poisoned(shape, cuda_device)
        partials = torch.empty(
            plan.partial_shape(8, extent), dtype=torch.float32, device=cuda_device
        )
        _build._select_ext().select_prefill_candidates(
            scores,
            geometry.local_valid_blocks,
            geometry.forced_column,
            geometry.active_rows,
            out,
            partials,
            geometry.scan_block_begin,
            plan.use_pdl,
            extent,
        )
        _assert_no_survivors(out)
        return out

    narrow, wide = at_extent(520), at_extent(plan.max_local_blocks)
    assert torch.equal(wide.view(torch.int32), narrow.view(torch.int32))
    # PRECONDITION: the two launches must have taken DIFFERENT geometries.
    assert plan.launch(520) != plan.launch(plan.max_local_blocks)
    # And the entry point takes the wide one, whatever it is told.
    entry = _poisoned(shape, cuda_device)
    select_prefill_candidates(scores, geometry, plan=plan, live_blocks=520, out=entry)
    assert torch.equal(entry.view(torch.int32), wide.view(torch.int32))


@pytest.mark.gpu
def test_a_retained_partials_buffer_changes_nothing_and_allocates_nothing(cuda_device):
    """The prefill scratch, supplied instead of allocated per call.

    Prefill runs once per layer per query chunk, and every one of those calls
    allocated a partitioned-scratch tensor whose shape is
    ``(T, H_group, partitions, 16, 2)`` -- and ``partitions`` comes from
    ``scan_extent``, a startup constant. So the allocation was per call and the
    size was not.

    Nothing about the launch changes: the extent is still ``scan_extent`` and no
    host value is read off a device tensor here. The claims are therefore
    exactly two -- the answer is unchanged, and the buffer is reused rather than
    replaced -- plus the one that makes the first mean something: the scratch is
    **poisoned** between calls, so a kernel that read a partial before writing
    it would carry the poison into the output instead of agreeing.
    """
    import torch

    tokens = 8
    # Above PARTITION_BLOCKS, so the launch really is partitioned and the
    # scratch really exists; at or below it one CTA covers the row and there is
    # nothing to retain.
    plan = _plan(
        PrefillPlan, token_capacity=tokens, max_local_blocks=2 * PARTITION_BLOCKS
    )
    scores, geometry = _make_case(plan, tokens=tokens, live=520, device=cuda_device)
    shape = (tokens, plan.num_heads_group, CAND_K, 2)
    partial_shape = plan.partial_shape(tokens, plan.scan_extent)
    # PRECONDITION: this capacity must actually partition. At a capacity that
    # fits one CTA per row the scratch is zero-width and retaining it is a
    # claim about nothing.
    assert partial_shape[2] > 0, (
        f"max_local_blocks={plan.max_local_blocks} plans "
        f"{partial_shape[2]} partitions, so there is no scratch to retain"
    )

    want = _poisoned(shape, cuda_device)
    select_prefill_candidates(scores, geometry, plan=plan, live_blocks=520, out=want)
    _assert_no_survivors(want)

    partials = torch.empty(partial_shape, dtype=torch.float32, device=cuda_device)
    ptr = partials.data_ptr()
    got = _poisoned(shape, cuda_device)
    select_prefill_candidates(
        scores, geometry, plan=plan, live_blocks=520, out=got, partials=partials
    )
    _assert_no_survivors(got)
    assert torch.equal(got.view(torch.int32), want.view(torch.int32))

    def allocations() -> int:
        # The allocator's CUMULATIVE allocation count. `memory_allocated()`
        # would not see a per-call scratch at all: it is freed before the call
        # returns, so the live byte count is back where it started either way.
        return torch.cuda.memory_stats(cuda_device)["allocation.all.allocated"]

    def one_call(**kwargs) -> int:
        """One selector call and the allocations it made, and nothing else.

        `out` is reused and re-poisoned in place, and the comparisons stay
        OUTSIDE the measured window: `==`, `.any()` and `torch.equal` all
        allocate, and a window wide enough to include them measures the test
        instead of the selector.
        """
        got.view(torch.int32).fill_(POISON)
        torch.cuda.synchronize()
        start = allocations()
        select_prefill_candidates(
            scores, geometry, plan=plan, live_blocks=520, out=got, **kwargs
        )
        torch.cuda.synchronize()
        return allocations() - start

    for _ in range(3):
        partials.fill_(float("nan"))
        retained = one_call(partials=partials)
        _assert_no_survivors(got)
        assert torch.equal(got.view(torch.int32), want.view(torch.int32)), (
            "a repeated call through the retained scratch disagreed with the "
            "allocating call -- the scratch is not write-before-read"
        )
        assert partials.data_ptr() == ptr
        assert retained == 0, (
            f"a call through the retained scratch still made {retained} allocations"
        )

    # The discriminating half: the same call WITHOUT the scratch allocates it,
    # so "0" above is a property of the workspace and not of the measurement.
    allocating = one_call()
    _assert_no_survivors(got)
    assert allocating == 1, (
        f"the allocating path made {allocating} allocations for one call; if "
        "it is 0 this comparison cannot tell the two paths apart"
    )

    # And the buffer is checked exactly as the allocated one would have been.
    with pytest.raises(ValueError, match="partials shape"):
        select_prefill_candidates(
            scores,
            geometry,
            plan=plan,
            live_blocks=520,
            out=_poisoned(shape, cuda_device),
            partials=partials[:1],
        )


@pytest.mark.gpu
def test_an_under_reported_live_extent_can_no_longer_truncate(cuda_device):
    """REPLACES ``test_prefill_refuses_an_under_reported_live_extent``.

    That test gated the refusal raised by ``_check_live_blocks``, which reached
    it by reading ``local_valid_blocks.max()`` into a Python integer on every
    prefill layer and query chunk -- the N2 submission-path synchronisation. The
    check is gone, and the *guarantee* it protected is not weakened but
    strengthened: an under-reported extent is no longer refused, it is INERT,
    because nothing per-step sizes the launch. So the assertion inverts -- the
    same call that used to raise must now return the same 16 candidates as the
    truthful one, down to an extent of 0.

    A "same answer" assertion alone would also pass against a selector that
    truncated both calls identically, so the launch identity is checked too: the
    rung is the capacity rung whatever ``live_blocks`` says, and the two rungs
    are distinguishable symbols.
    """
    import torch

    from fmha_sm100.icp.candidates import (
        last_launch,
        planned_prefill_kernel,
        record_launches,
    )

    plan = _plan(PrefillPlan, token_capacity=4, max_local_blocks=2048)
    scores, geometry = _make_case(plan, tokens=4, live=300, device=cuda_device)
    shape = (4, plan.num_heads_group, CAND_K, 2)
    truth = _poisoned(shape, cuda_device)
    select_prefill_candidates(scores, geometry, plan=plan, live_blocks=300, out=truth)
    _assert_no_survivors(truth)
    # PRECONDITION: the rows must be long enough that a truncated scan would
    # show. At 300 live columns every row emits a full 16, and the columns an
    # extent of 128 would drop are where most of them come from.
    assert int((truth.view(torch.int32)[..., 1] >= 0).sum(-1).min()) == CAND_K

    for understated in (299, 128, 1, 0):
        got = _poisoned(shape, cuda_device)
        select_prefill_candidates(
            scores, geometry, plan=plan, live_blocks=understated, out=got
        )
        _assert_no_survivors(got)
        assert torch.equal(got.view(torch.int32), truth.view(torch.int32)), (
            f"live_blocks={understated} changed the answer, so it still sizes "
            "the scan; an under-report is supposed to be inert"
        )

    record_launches(True)
    try:
        select_prefill_candidates(
            scores,
            geometry,
            plan=plan,
            live_blocks=1,
            out=_poisoned(shape, cuda_device),
        )
        launched = last_launch()
    finally:
        record_launches(False)
    assert launched["count"] > 0
    assert launched["symbol"] == planned_prefill_kernel(plan.scan_extent)["symbol"]
    # The discriminating half: the rung an under-report WOULD have named is a
    # different kernel, so "it took the capacity rung" is not vacuously true.
    assert planned_prefill_kernel(1)["symbol"] != launched["symbol"]

    # The host-side range check survives: it costs nothing and a value outside
    # capacity still means the caller's metadata is wrong.
    with pytest.raises(ValueError, match="live_blocks"):
        select_prefill_candidates(scores, geometry, plan=plan, live_blocks=-1)
    with pytest.raises(TypeError, match="host int"):
        select_prefill_candidates(
            scores, geometry, plan=plan, live_blocks=geometry.local_valid_blocks.max()
        )


@pytest.mark.gpu
def test_the_prefill_submission_path_reads_no_device_value(cuda_device, monkeypatch):
    """The N2 regression gate: no blocking device->host read on this path.

    Every way a CUDA tensor can become a Python value is made to raise, and the
    selector must still run. This is the check that fails if the readback is
    reintroduced in any spelling -- ``.item()``, ``int(t)``, ``.cpu()``,
    ``.tolist()`` -- rather than only in the one the old code used.

    The JIT build and the first extension load happen before the patches: they
    are startup, not submission, and ``cpp_extension.load`` is entitled to
    whatever it does.
    """
    import torch

    plan = _plan(PrefillPlan, token_capacity=4, max_local_blocks=2048)
    scores, geometry = _make_case(plan, tokens=4, live=300, device=cuda_device)
    shape = (4, plan.num_heads_group, CAND_K, 2)
    reference = _poisoned(shape, cuda_device)
    select_prefill_candidates(
        scores, geometry, plan=plan, live_blocks=300, out=reference
    )

    def refuse(what):
        def raiser(*_args, **_kwargs):
            raise AssertionError(
                f"the prefill submission path called {what}, which blocks on "
                "the device. Checks may not read a CUDA value into Python here "
                "-- see docs/CONTRACT.md and the N2 fix in candidates.py."
            )

        return raiser

    for attribute in (
        "item",
        "tolist",
        "cpu",
        "numpy",
        "__int__",
        "__float__",
        "__bool__",
        "__index__",
    ):
        monkeypatch.setattr(
            torch.Tensor, attribute, refuse(f"Tensor.{attribute}()"), raising=False
        )
    monkeypatch.setattr(torch.cuda, "synchronize", refuse("torch.cuda.synchronize()"))
    monkeypatch.setattr(
        torch.cuda.Stream, "synchronize", refuse("Stream.synchronize()"), raising=False
    )
    monkeypatch.setattr(
        torch.cuda.Event, "synchronize", refuse("Event.synchronize()"), raising=False
    )

    got = _poisoned(shape, cuda_device)
    select_prefill_candidates(scores, geometry, plan=plan, live_blocks=300, out=got)

    # PRECONDITION: the patches must be able to fire, or this passes vacuously.
    with pytest.raises(AssertionError, match="blocks on the device"):
        int(geometry.local_valid_blocks.max())

    monkeypatch.undo()
    _assert_no_survivors(got)
    assert torch.equal(got.view(torch.int32), reference.view(torch.int32))


#: Drives the extension directly with an extent that does NOT cover the row --
#: 100 buys a 128-column CTA and the row holds 300 live blocks. The Python entry
#: point cannot produce this any more (it passes the plan's capacity), so the
#: trap is proved through the extension. A build without the assertions reaches
#: the print and exits 0; with them the device aborts and CUDA reports it at the
#: synchronise, which is why this runs in its own process: a fired device
#: assertion poisons the context for everything after it.
_TRUNCATION_TRAP = """
import sys
sys.path.insert(0, {root!r})
import torch
from fmha_sm100.icp import _build

T, H, N, LIVE, EXTENT = 1, 1, 2048, 300, 100
scores = torch.randn((T, H, N), device="cuda")
nvalid = torch.full((T,), LIVE, dtype=torch.int32, device="cuda")
forced = torch.full((T,), -1, dtype=torch.int32, device="cuda")
active = torch.ones((T,), dtype=torch.bool, device="cuda")
out = torch.empty((T, H, 16, 2), dtype=torch.float32, device="cuda")
partials = torch.empty((T, H, 0, 16, 2), dtype=torch.float32, device="cuda")
_build._select_ext().select_prefill_candidates(
    scores, nvalid, forced, active, out, partials, 0, False, EXTENT)
torch.cuda.synchronize()
print("NO TRAP: the launch truncated the row and said nothing")
"""


@pytest.mark.gpu
def test_a_launch_that_does_not_cover_the_row_traps_on_the_device():
    """The backstop fires, and it returns nothing to Python while doing it.

    "Do not replace a wrong bound with silent truncation" is the review's
    explicit warning. The bound is now structural, so this can only be reached
    by driving the extension by hand -- but if it is reached, it aborts on the
    device instead of dropping columns quietly. A device assertion is the only
    form of this check that the nonblocking policy allows: it copies no status
    to the host and nothing on the submission path waits for it.
    """
    import pathlib
    import subprocess
    import sys

    import fmha_sm100.icp

    root = str(pathlib.Path(fmha_sm100.icp.__file__).resolve().parents[2])
    finished = subprocess.run(
        [sys.executable, "-c", _TRUNCATION_TRAP.format(root=root)],
        capture_output=True,
        text=True,
        timeout=900,
    )
    output = finished.stdout + finished.stderr
    assert "NO TRAP" not in output, (
        "a launch whose extent does not cover the row ran to completion: the "
        "row was truncated silently, which is the exact failure the removed "
        f"host check existed to catch.\n{output[-2000:]}"
    )
    assert finished.returncode != 0, output[-2000:]
    assert "assert" in output.lower(), (
        "the process failed, but not at the device assertion -- so this test "
        f"is gating something else.\n{output[-2000:]}"
    )


@pytest.mark.gpu
def test_prefill_refuses_to_run_inside_a_capture(cuda_device):
    """Everything the eager path does is illegal under capture.

    USED TO BE TWO TESTS, one of them expecting a "graph-private pool" message
    from the allocation. There is one refusal now and it fires before anything
    allocates, so the entry point refuses the *call* rather than one of its
    steps. (The list of what is illegal lost one item with N2: the synchronising
    ``live_blocks`` check is gone from this band, and the geometry is the same
    startup constant a capture would bake in. The allocation and the per-call
    ``T`` are still reason enough.)
    """
    import torch

    plan = _plan(PrefillPlan, token_capacity=4, max_local_blocks=512)
    scores, geometry = _make_case(plan, tokens=4, live=100, device=cuda_device)
    select_prefill_candidates(scores, geometry, plan=plan, live_blocks=100)
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="not capturable"):
        with torch.cuda.graph(graph):
            select_prefill_candidates(scores, geometry, plan=plan, live_blocks=100)


@pytest.mark.gpu
def test_the_prefill_dispatch_takes_the_rung_the_startup_extent_names(cuda_device):
    """USED TO ASSERT the rung named by the per-call ``live_blocks`` (520).

    Same gate, new extent: N2 makes the launch a function of
    ``plan.scan_extent``, so the symbol to expect is the capacity rung's. The
    negative half is unchanged in spirit and is what keeps it honest -- the rung
    the live extent would have named is a *different* kernel, so this cannot
    pass by the two being the same symbol.
    """
    from fmha_sm100.icp.candidates import (
        last_launch,
        planned_prefill_kernel,
        record_launches,
    )

    # One arm on this path, so only a wrong rung can go unseen -- and no output
    # comparison can catch it on a case shorter than the rung it dropped.
    plan = _plan(PrefillPlan, token_capacity=4, max_local_blocks=4096)
    scores, geometry = _make_case(plan, tokens=4, live=520, device=cuda_device)
    record_launches(True)
    try:
        select_prefill_candidates(scores, geometry, plan=plan, live_blocks=520)
        launched = last_launch()
    finally:
        record_launches(False)
    assert launched["count"] > 0
    assert launched["symbol"] == planned_prefill_kernel(plan.scan_extent)["symbol"]
    assert launched["symbol"] != planned_prefill_kernel(520)["symbol"]


# --------------------------------------------------------------------------
# tier 3: the negative controls. GPU.
# --------------------------------------------------------------------------

#: Which controls each geometry can actually observe, and WHY. A control driven
#: outside its regime is vacuous, and listing the regime is what stops one being
#: reported as a pass. Every entry's precondition is asserted in the test body
#: before the control is allowed to fire.
#:
#: refined-icp-v1 re-pointed two of these without renumbering them: control 2
#: used to be "forcing ignores owns[token]" and is now "exclusion ignores
#: forced_col[] and drops valid-1", and control 3 used to be "the id uses
#: (rank+1)%world" and is now "the id drops scan_block_begin". Controls 7, 8 and
#: 9 are new.
#:
#: ``arm`` is part of the regime, not a detail: the control build dispatches on
#: the SAME ``cap`` code the shipping launcher does -- 0 sort, -1/-2 radix, >0
#: capped -- and the controls do not all live in all three arms.
#: ``ICP_RS_CTL(2)``, ``(3)``, ``(5)``, ``(6)`` and ``(9)`` are in
#: ``local_candidates_rs.cuh`` only, so driving them on the sort arm would gate
#: nothing. ``(1)``, ``(4)``, ``(7)`` and ``(8)`` exist in both, and refined
#: -icp-v1 hoisted them into ``local_candidates.cuh`` precisely so the sort arm
#: is gated at all -- so those are driven on BOTH arms rather than on whichever
#: one happened to be the default.
_BOTH_ARMS = (SelectArm.SORT, SelectArm.RADIX_BOUNDED)

CONTROL_CASES = [
    *[
        (
            1,
            dict(
                live=5000,
                max_local_blocks=8192,
                arm=arm,
                why="skips the last partition; needs more than one",
            ),
        )
        for arm in _BOTH_ARMS
    ],
    (
        2,
        dict(
            live=520,
            max_local_blocks=2048,
            arm=SelectArm.RADIX_BOUNDED,
            why="excludes valid-1 instead of forced_col",
        ),
    ),
    (
        3,
        dict(
            live=520,
            max_local_blocks=2048,
            arm=SelectArm.RADIX_BOUNDED,
            why="drops scan_block_begin from the id",
        ),
    ),
    *[
        (
            4,
            dict(
                live=5000,
                max_local_blocks=8192,
                arm=arm,
                why="the combine reads count-16; needs more than one partition",
            ),
        )
        for arm in _BOTH_ARMS
    ],
    (
        5,
        dict(
            live=520,
            max_local_blocks=2048,
            arm=SelectArm.RADIX_BOUNDED,
            why="keeps 15 local candidates, not 16",
        ),
    ),
    (
        6,
        dict(
            live=5,
            max_local_blocks=2048,
            arm=SelectArm.RADIX_BOUNDED,
            why="the bounded arm's live count is short by one; needs a row "
            "shorter than 16 AND the bounded radix arm, which is the only "
            "one that passes a non-negative live_count",
        ),
    ),
    *[
        (
            7,
            dict(
                live=520,
                max_local_blocks=2048,
                poison_tail=True,
                arm=arm,
                why="reads one column past valid; needs a poisoned invalid tail",
            ),
        )
        for arm in _BOTH_ARMS
    ],
    *[
        (
            8,
            dict(
                live=520,
                max_local_blocks=2048,
                arm=arm,
                why="skips the last winner write; needs a poisoned output",
            ),
        )
        for arm in _BOTH_ARMS
    ],
    (
        9,
        dict(
            live=520,
            max_local_blocks=2048,
            arm=SelectArm.RADIX_BOUNDED,
            why="keeps 14 local candidates; control 5's non-vacuous sibling",
        ),
    ),
]


@pytest.mark.gpu
@pytest.mark.parametrize(
    "control,case",
    CONTROL_CASES,
    ids=[f"control{c}-{case['arm'].name.lower()}" for c, case in CONTROL_CASES],
)
def test_each_negative_control_changes_the_output(cuda_device, control, case):
    """Each control perturbs exactly one site and must be observable HERE.

    This drives the capacity-fed launcher, so it takes a ``DecodePlan`` -- the
    control entry point is the decode one, and ``allocate_workspace`` refuses a
    ``PrefillPlan`` by design. (USED TO build a ``PrefillPlan`` and allocate a
    workspace from it, which cannot run at all.)
    """
    import torch

    from fmha_sm100.icp.candidates import select_with_control

    tokens, live = 4, case["live"]
    plan = _plan(
        DecodePlan,
        icp_degree=4,
        icp_rank=1,
        token_capacity=tokens,
        max_local_blocks=case["max_local_blocks"],
    )
    workspace = allocate_workspace(plan, cuda_device)
    scores, geometry = _make_case(plan, tokens=tokens, live=live, device=cuda_device)
    shape = (tokens, plan.num_heads_group, CAND_K, 2)

    # --- the preconditions. A control outside its regime is not a pass. ------
    if control in (1, 4):
        assert plan.partial_count > 1, (
            f"control {control} ({case['why']}) needs more than one partition; "
            f"this capacity gives {plan.partial_count}"
        )
    if control == 2:
        columns = geometry.forced_column.tolist()
        assert any(c >= 0 for c in columns), "no row forces anything"
        assert all(c != live - 1 for c in columns if c >= 0), (
            "the forced column IS valid-1, which is exactly what control 2 "
            "substitutes, so this case cannot observe it"
        )
    if control == 3:
        assert geometry.scan_block_begin != 0, (
            "control 3 drops scan_block_begin; at a zero origin it is the "
            "identity and cannot be observed"
        )
    if control == 6:
        assert live < CAND_K, (
            "control 6 shortens the live-count bound, which only exists on a "
            f"row shorter than {CAND_K}"
        )
    if case.get("poison_tail"):
        # Control 7 reads ONE column past `valid`. Against an -inf-filled or
        # zero-filled tail that is silently harmless, which is why the blanket
        # score-buffer fill made this whole class of bug invisible: the column
        # has to hold something that would WIN if it were read.
        scores[:, :, live] = 1e30

    ref = _poisoned(shape, cuda_device)
    select_decode_candidates(scores, geometry, plan=plan, workspace=workspace, out=ref)
    _assert_no_survivors(ref)

    if case.get("poison_tail"):
        past_the_end = geometry.scan_block_begin + live
        assert not (ref.view(torch.int32)[..., 1] == past_the_end).any(), (
            "the shipped selector already emitted the column past `valid`, so "
            "control 7 cannot be distinguished from correct behaviour"
        )
    if control in (5, 8, 9):
        full = (ref.view(torch.int32)[..., 1] >= 0).sum(-1)
        assert int(full.min()) == CAND_K, (
            f"control {control} drops one emitted candidate, which is only "
            "observable on rows that emit all 16"
        )

    arm = case["arm"]

    # Control 0 perturbs nothing, so it must reproduce the shipping answer, on
    # whichever arm this control needs -- every arm is bit-exact against every
    # other. A gate that cannot tell 0 from a real control is not gating.
    unperturbed = _poisoned(shape, cuda_device)
    workspace.partials.view(torch.int32).fill_(POISON)
    select_with_control(
        scores,
        geometry,
        plan=plan,
        workspace=workspace,
        out=unperturbed,
        control=0,
        arm=arm,
    )
    assert torch.equal(unperturbed.view(torch.int32), ref.view(torch.int32))

    # Controls 1 and 8 leave the previous launch's correct value in the scratch
    # (or in `out`), so both have to be poisoned before they can fire at all.
    perturbed = _poisoned(shape, cuda_device)
    workspace.partials.view(torch.int32).fill_(POISON)
    select_with_control(
        scores,
        geometry,
        plan=plan,
        workspace=workspace,
        out=perturbed,
        control=control,
        arm=arm,
    )
    assert not torch.equal(perturbed.view(torch.int32), ref.view(torch.int32)), (
        f"control {control} ({case['why']}) did not change the output on arm "
        f"{arm.name}; either it is compiled out, this geometry cannot observe "
        "it, or the launch did not take the arm this case asked for. The third "
        "is not hypothetical: the extension dispatches on `cap` exactly as the "
        "shipping launcher does (0 sort, -1/-2 radix, >0 capped), so a wrapper "
        "that clamped the arm to a non-negative cap would run the sort arm -- "
        "which contains none of ICP_RS_CTL(2), (3), (5), (6) or (9) -- under "
        "the radix arm's name, and every one of those would report here as "
        "'did not fire'."
    )
