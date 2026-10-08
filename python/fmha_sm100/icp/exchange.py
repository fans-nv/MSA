"""K5: the fused symmetric-memory candidate exchange + merge.

One launch per indexer layer replaces ``all_gather_into_tensor`` +
:func:`fmha_sm100.icp.merge_candidates`. The kernel pushes this rank's candidate
slice straight into each peer's receive buffer, hands off with a ``.sys``-scope
release/acquire (or with a generation tag riding inside the payload, under
:attr:`MaskPreset.LAMPORT`), and writes this rank's ``[T, H_local, 16]`` int32
output. The *selection* is not reimplemented here: ``k5_exchange.cu`` and
``k2_merge.cu`` compile the same ``merge_topk.cuh``, which is the only reason
``tests/test_exchange.py`` can gate the two with ``torch.equal``.

Integrator's checklist -- each item bought by a real failure
-----------------------------------------------------------

**1. ``torch.cuda.set_device(local_rank)`` before ``init_process_group()``.**
Symmetric memory rendezvous keys on the CUDA ordinal each rank reports. Without
the ``set_device`` every rank reports ordinal 0 and
``CUDASymmetricMemoryAllocator::rendezvous`` raises "detected allocations from
overlapping devices from different ranks" -- the check doing its job on
ordinals it cannot disambiguate, not a bug.

**2. Never mask ``CUDA_VISIBLE_DEVICES`` per rank.** One device per process is
the standard external-LB layout everywhere else, and it is precisely what
breaks here: every rank then sees a single device, calls ``set_device(0)``, and
item 1's failure returns. Measured upstream: 4 ranks with per-task masking fail
4/4 at rendezvous; the identical run unmasked passes. Give every rank the whole
device list and select with ``set_device(local_rank)``.

**3. Never set ``TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES=1``.** It is the
documented escape hatch from item 1 and it is not free:
``CUDASymmetricMemory.cu:853`` skips ``init_multicast_for_block`` when it is
set, **silently disabling NVLS multicast** -- a loud failure traded for a quiet
mismeasurement of exactly the thing this kernel exists to exploit. Leave
``TORCH_SYMMMEM`` and ``NVSHMEM_MAX_TEAMS`` unset too.

**4. One stream per group.** Every symmetric-memory operation for one process
group must be issued on a single stream; two streams on one group deadlock
permanently with no error (pytorch#189228), which PyTorch's own ops warn about
at ``CUDASymmetricMemoryOps.cu:73-100``.

**5. Nothing is allocated during cudagraph capture.** Symmetric memory cannot
be, and ``rendezvous()`` is a host-blocking collective that cannot be captured
at all. So :class:`IcpExchange` allocates and rendezvouses **eagerly in
``__init__``** while :meth:`IcpExchange.exchange_and_merge` only validates
shapes. Construct it outside the capture at the largest ``tokens`` you will
replay, and capture only the ``exchange_and_merge`` calls.

``tokens`` is the allocation **capacity**; a captured call may execute over a
shorter prefix by passing ``num_tokens``, so one window serves every graph
instead of one window per graph. Both numbers are *captured* quantities, never
``num_actual_tokens`` (CONTRACT C4): the extent is the graph's padded row count,
fixed across ranks and across replays, and rows past the real batch inside it
carry ``(-inf, -1)`` and nobody reads them.

**6. The generation counter must live in device memory.** A host-side scalar
would be baked into the graph at capture, so every replay would re-publish the
same number and every peer's acquire would pass instantly on the *previous*
replay's flag -- silent wrong answers, not a hang. This class defaults to the
kernel's on-device derivation (``seq = own_generation + 1``): monotonic by
construction, needing no host involvement, and unfreezable by a capture.

Presets
-------

:class:`MaskPreset` values *are* the kernel's ``opts`` mask, so a preset and a
raw mask are the same argument. The kernel instantiates three masks -- ``0``
(the control, not a preset), ``12`` (:attr:`MaskPreset.HANDSHAKE`) and ``512``
(:attr:`MaskPreset.LAMPORT`) -- and :data:`INSTANTIATED_MASKS` is that dispatch
table's Python mirror. The ``OPT_*`` bits themselves, the measurements that
disproved five of them, and the two that are implemented but dispatched by
nothing are documented in ``csrc/k5_exchange.cu``.

Failure behaviour
-----------------

A failed launch is not silent and does not leave ``out`` untouched: it sets an
error flag, fills the failing blocks' output rows with **block id 0**, and makes
every later launch on the same workspace short-circuit the same way. Read
:meth:`IcpExchange.exchange_and_merge` before integrating -- the choice of ``0``
carries a caller obligation on prefill paths that this API cannot check.
"""

from __future__ import annotations

import enum

import torch
import torch.distributed as dist

from . import _build

CAND_K = 16

#: ~1 s at 2 GHz. A timeout that turns a cross-rank mismatch into a reported
#: error instead of an undebuggable hang inside a cudagraph replay -- not a
#: latency budget.
DEFAULT_SPIN_CYCLES = 2_000_000_000

#: The FLOOR on ``max_blocks``, and the value ``kDefaultMaxBlocks`` in
#: ``csrc/k5_exchange.cu`` holds.
#:
#: It is not a number this class has to agree with. ``k5_exchange`` takes
#: ``max_blocks`` as an ARGUMENT and plans from that argument
#: (``k5_exchange.cu:904``), and :class:`IcpExchange` passes the same
#: ``self.max_blocks`` to ``k5_plan`` and to ``k5_exchange``, so host and device
#: agree by construction rather than by matching this constant. The kernel's own
#: ``kDefaultMaxBlocks`` survives in exactly two places: as the pybind default of
#: ``k5_plan``, and as the OPT_HIER residency bound (``k5_exchange.cu:932``,
#: which compares the LIVE grid against the constant, never against the
#: argument). :func:`plan_max_blocks` is what this class actually uses, and the
#: OPT_HIER bound is why that function is not applied under that bit.
DEFAULT_MAX_BLOCKS = 64

#: Fewest slots at which the ack may be dropped. The obligation is a reuse
#: distance of 2 over a sequence whose wrap also counts, so 2 is never enough:
#: a back-to-back sweep of ``n`` layers repeats a slot across the wrap whenever
#: ``n % slots == 1``, and ``slots=2`` hits that at every odd ``n`` (57 layers
#: included). Callers still owe the ``n % slots != 1`` check; this bound only
#: removes the case no ``n`` can survive.
ACK_FREE_SLOTS = 3

MODE_PULL = 0
MODE_PUSH = 1

# Kernel option bits, mirrored from `csrc/k5_exchange.cu`'s OPT_* enum and
# asserted equal to it at load time by `opt_bits()`. Read that enum for the
# rationale and the numbers; do not restate them here.
#
# The bit *positions* are frozen even though five of the ten original bits have
# been dropped: the survivors keep the values they were measured under, so every
# results CSV that quotes a mask (0, 12, 512, 768) still names the same
# protocol. Renumbering to close the gaps would silently re-point that history.
OPT_SELF_ELIDE = 1 << 0
OPT_NOFENCE = 1 << 2
OPT_HIER = 1 << 3
OPT_NODIV = 1 << 8
OPT_LAMPORT = 1 << 9

#: The NAME -> VALUE table. It is asserted EXACTLY equal to the kernel's
#: ``k5_opts()`` at workspace construction, so this dict and ``csrc/
#: k5_exchange.cu`` are one atomic landing unit -- a bit added or removed on one
#: side without the other is a loud failure in :func:`opt_bits`, never a silent
#: divergence. It is a *naming* table and not a support table: it contains bits
#: the kernel implements and refuses, so that a mask number or an option string
#: resolves to its bit and is refused **by name with the reason**, rather than
#: dying as an unknown option.
_OPT_NAMES = {
    "self_elide": OPT_SELF_ELIDE,
    "nofence": OPT_NOFENCE,
    "hier": OPT_HIER,
    "nodiv": OPT_NODIV,
    "lamport": OPT_LAMPORT,
}

#: Bits any dispatched mask may set. Mirrors ``kOptSupported``.
SUPPORTED_OPTS = OPT_NOFENCE | OPT_HIER | OPT_LAMPORT

#: Implemented in the kernel's device code, dispatched by nothing, and refused
#: by name. Mirrors ``kRetiredOpts`` / ``kDormantOpts``, and the split is not
#: cosmetic: ``OPT_SELF_ELIDE`` was dispatched, measured and taken out again
#: (mask 513, within +/-0.4 us of plain 512), while ``OPT_NODIV`` has never been
#: dispatched *here* at all -- the 24-of-24 matrix that chose 512 over 12
#: contains no mask-768 cell. Landing the first back needs a `case`; landing the
#: second needs a `case`, a gate and a price.
RETIRED_OPTS = OPT_SELF_ELIDE
DORMANT_OPTS = OPT_NODIV

#: The masks ``k5_exchange``'s dispatch switch actually instantiates -- the
#: Python mirror of that switch, and the reason an unbuilt mask is a *message*
#: here rather than an anonymous ``TORCH_CHECK`` at first launch, which under a
#: cudagraph may be the first *captured* launch.
#:
#: ``0`` is in the set and is not a preset: it is the control arm every
#: optimisation is measured against, and what makes the kernel's ``opts = 0``
#: pybind default legal instead of an exception on the first launch. It is
#: **not** byte-identical to the pre-options kernel any more -- the failure
#: containment (`fill_failed_rows`) is unconditional on the mask, deliberately,
#: because gating safety code on ``OPTS != 0`` manufactures a mask that is
#: silently less safe than the others. Registers, shared memory and occupancy
#: are unchanged, so the control arm's *cost* is unchanged; only byte-identity
#: is gone. See ``docs/CONTRACT.md`` §9.
INSTANTIATED_MASKS = frozenset({0, OPT_NOFENCE | OPT_HIER, OPT_LAMPORT})

#: Bits that change the *shape* of the receive window, i.e. that the allocation
#: must agree with. ``OPT_LAMPORT`` doubles every candidate so it can carry its
#: generation tag. Running a mask against a window sized for a different one
#: writes past the end of a *peer's* symmetric allocation -- which no allocator
#: sees and which still returns plausible block ids. That is why these bits are
#: fixed at construction and are not a per-call argument.
LAYOUT_OPTS = OPT_LAMPORT


class MaskPreset(enum.IntEnum):
    """The two shipped ``opts`` masks. The value *is* the mask.

    Both integer values are frozen: the docs and every results CSV quote them.
    A third mask -- ``0``, the control -- is instantiated by the kernel and is
    deliberately **not** a preset: it is a bisection and measurement arm, not a
    configuration anyone should choose. Pass the raw integer ``0`` for it, and
    see :data:`INSTANTIATED_MASKS`.

    ``HANDSHAKE`` (12)
        ``OPT_NOFENCE | OPT_HIER``: a release/acquire handshake, tuned. It must
        land **whole** -- the two bits are non-additive and either one alone
        regresses part of the range by 8-16%, which is why the kernel does not
        build masks 4 and 8 at all.

        ``OPT_HIER`` costs the no-residency property: every block must reach
        the arrival counter before any block spins, so the grid must be
        resident. The kernel refuses ``nblocks > 64`` under this bit rather
        than deadlocking.

        It also costs the **short extent**: one flag for the whole grid means
        every block must be on the same generation, and a block skipped by a
        short launch is not. ``exchange_and_merge(num_tokens=...)`` refuses this
        mask below capacity -- see its docstring.

        It is **no longer "the recommendation"**. ``LAMPORT`` beat it in 24 of
        24 measured cells (sm_107a, C in {2, 4}, six token counts, both ack
        settings), by 1.34 to 4.71 us/layer. ``HANDSHAKE`` remains the mask to
        use when the 2x window is not affordable, and the mask whose ack-off
        configuration has the most measurement behind it.

    ``LAMPORT`` (512)
        ``OPT_LAMPORT``: data-as-flag. No release, no acquire, no arrival
        counter and no fence -- the generation tag rides inside every 64-bit
        payload word, so the arrival of a word *is* the proof it belongs to
        this generation. Costs a **2x** symmetric window, which
        ``k5_slot_floats`` and therefore this class's allocation already
        account for.

        Was ``768`` (``OPT_LAMPORT | OPT_NODIV``) in this package's previous
        revision. It is 512 now because the landed kernel does not instantiate
        768: ``OPT_NODIV`` is implemented in its device code and dispatched by
        nothing. That is not a deletion -- see :data:`DORMANT_OPTS` and the
        ``OPT_NODIV`` refusal in ``csrc/k5_exchange.cu`` for the measurement
        that is missing and what landing it would take.

        The ack is **not** switched off by this mask. The kernel deletes only
        the publish flag under Lamport and keeps the consumer ack (control-block
        kind 1) exactly as it was, which is what lets this mask ship without
        the ack-free slot-reuse argument.

        Gated bit-identical against the all-gather reference at both C=2 and
        C=4 on sm_107a (207/207 plus a 10 000 x 2 cudagraph replay soak), and
        at C=2 and C=4 on GB300 sm_103a (11 gates x 3 masks x 2 world sizes,
        all pass, with the stale-producer poison firing at every mask). **C=8
        is unmeasured at every mask.**
    """

    HANDSHAKE = OPT_NOFENCE | OPT_HIER      # 12
    LAMPORT = OPT_LAMPORT                   # 512


def opt_bits(mod: object | None = None) -> dict[str, int]:
    """The kernel's own option table, asserted equal to the Python mirror.

    A silently renumbered bit would run a *different* optimisation under a
    preset's name, which is the one failure mode neither a benchmark nor a
    correctness gate can see from its own output.
    """
    if mod is None:
        mod = _build._k5()
    kernel_bits = dict(mod.k5_opts())
    if kernel_bits != _OPT_NAMES:
        raise RuntimeError(
            f"k5 option bits drifted: kernel={kernel_bits} python={_OPT_NAMES}"
        )
    return kernel_bits


def natural_tpb(k5_plan, heads_local: int) -> int:
    """``kWarpsPerBlock / H_local``, asked of the kernel instead of mirrored.

    ``plan_tpb`` (``csrc/k5_exchange.cu:652``) starts from
    ``kWarpsPerBlock / H_local`` -- one warp per ``(token, local head)`` -- and
    then *widens* the block until ``ceil(T / max_blocks)`` fits in it. At
    ``T == 1`` the widening term is ``ceil(1 / max_blocks) == 1``, which is
    never greater than a ``tpb`` the same function has already clamped to
    ``>= 1``. So ``k5_plan(1, H_local, 1)[0]`` returns the unwidened value and
    nothing else can.

    This is a PROBE rather than a mirrored ``WARPS_PER_BLOCK = 8`` on purpose:
    the one thing this module must not do is carry a second copy of a kernel
    constant that decides a grid. :data:`DEFAULT_MAX_BLOCKS`'s own docstring is
    the record of how that goes wrong.
    """
    return int(k5_plan(1, int(heads_local), 1)[0])


def plan_max_blocks(k5_plan, *, tokens: int, heads_local: int, opts: int,
                    floor: int = DEFAULT_MAX_BLOCKS) -> int:
    """The ``max_blocks`` at which ``plan_tpb`` stops widening the block.

    ``plan_tpb`` widens ``tpb`` -- it never adds blocks -- so a ``max_blocks``
    below ``ceil(T_cap / tpb_natural)`` is a request to fold the whole capacity
    into that many CTAs, and ``tpb`` grows until it fits. That was harmless
    while a window was allocated per extent, because the capacity WAS the
    extent. It stopped being harmless when one window at the global capacity
    started serving every extent: ``tpb`` is planned from the capacity (which is
    right -- see the kernel's note at ``k5_exchange.cu:238``, the row -> block
    map must not move) but the GRID is ``ceil(T / tpb)``, so a ``tpb`` widened
    on the capacity's behalf collapses every short launch's grid. At
    ``T_cap = 8192``, ``H_local = 2`` and ``max_blocks = 64`` the widened
    ``tpb`` is 128, and every extent up to 128 rows -- 4, 16 and 64 -- runs in
    ONE CTA where the per-extent windows gave 1, 4 and 16.

    Returning ``ceil(T_cap / tpb_natural)`` removes the widening: ``need`` then
    equals ``tpb_natural`` exactly, ``need > tpb`` is false, and ``tpb`` lands
    on its natural value. That restores ``ceil(T / tpb_natural)`` blocks at
    every extent -- which is exactly what a window allocated at that extent used
    to plan -- while leaving ``tpb`` a function of the CAPACITY alone, so the
    row -> block map is still fixed for the workspace's whole life and the
    Lamport tag is still written by one block and one counter.

    ``floor`` keeps the historical 64 as a lower bound: a capacity smaller than
    ``64 * tpb_natural`` needs no widening in the first place, and clamping up
    to the old default there keeps the planned ``tpb`` bit-identical to what
    every measurement in this repository was taken under.

    What this does NOT do is add blocks past the capacity's own row count; the
    grid is still ``ceil(T / tpb)`` and no block covers a row past ``T``.

    OPT_HIER IS EXCLUDED, and by a kernel refusal rather than a preference.
    That bit publishes one release for the whole grid, so the grid must be
    RESIDENT, and ``k5_exchange.cu:932`` bounds the live grid by the kernel's
    own ``kDefaultMaxBlocks`` constant -- *not* by the ``max_blocks`` it was
    handed. A derived ``max_blocks`` would plan 2048 blocks at
    ``T_cap = 8192, H_local = 2`` and the mask would then refuse its own
    full-capacity launch. It is also the one mask with nothing to gain: the
    refusal immediately below that one forbids OPT_HIER any extent short of the
    capacity, so its extent is always ``T_cap``, its grid is always
    ``nblocks``, and a widened ``tpb`` collapses no short launch because it has
    none. So under that bit this returns ``floor`` unchanged.
    """
    if int(opts) & OPT_HIER:
        return int(floor)
    tpb = natural_tpb(k5_plan, heads_local)
    need = -(-int(tokens) // tpb)  # ceil, in ints
    return max(int(floor), need)


def _rendezvous(symm_mem: object, tensor: torch.Tensor,
                group: dist.ProcessGroup | None) -> object:
    """``symm_mem.rendezvous`` across the two signatures torch has shipped."""
    try:
        return symm_mem.rendezvous(tensor, group=group)
    except TypeError:  # older signature takes the group *name*
        group_name = (group.group_name if group is not None
                      else dist.group.WORLD.group_name)
        return symm_mem.rendezvous(tensor, group_name)


class IcpExchange:
    """Fused ICP candidate exchange + merge over one ICP group.

    Parameters
    ----------
    group
        The ICP process group (``C`` consecutive TP ranks on one node). Its
        ``world_size`` is the ICP degree ``C``; its rank is ``icp_rank``.
    tokens
        ``T``, the **allocation capacity** in token rows (C4). Fixed for the
        lifetime of the object because the symmetric window is sized from it,
        and it is the largest extent any call may execute over.

        It is no longer the same number as the captured token count. A call may
        execute over a shorter **prefix** -- see ``num_tokens`` on
        :meth:`exchange_and_merge` -- so one window at the largest captured
        extent serves every smaller graph, instead of one window per graph. That
        matters because each instance owns a symmetric-memory window and a
        multicast team, and the team budget is finite: at NVSHMEM's default of
        128 teams, an NVLS-capable fabric can need ~48 internal teams per user
        team, so instances are the scarce resource here, not bytes.
    heads_group
        ``H_group = C * H_local``.
    heads_local
        ``H_local``, this rank's own GQA groups.
    slots
        Independent generations in flight. This class computes
        ``slot = layer_idx % slots``; the window is ``slots`` slots wide, so
        slots trade memory for the ack.

        **The obligation is "no two consecutive launches on a workspace may use
        the same slot" -- NOT ``slots >= 2``.** The sequence must have no two
        consecutive equal entries AND, because a capture bakes the vector in
        and replay repeats it, no equal wrap: for a back-to-back sweep of
        ``n`` layers that means ``slots >= ACK_FREE_SLOTS`` with
        ``n % slots != 1``. ``slots = n_call_sites`` satisfies it but is
        sufficient, not necessary, and costs 29x the 2-slot window; the in-tree
        caller uses 3 rotating slots for 57 layers at 1.5x.

        ``use_ack`` is derived as ``slots < ACK_FREE_SLOTS``, so the bound is
        enforced. The ``n % slots != 1`` half is **not**: it depends on the
        caller's layer count, which this class never sees. See
        ``docs/INTEGRATION.md`` section 6 for the in-tree policy, which adds an
        explicit ``slot_policy`` and a post-capture assertion this package has
        no way to make.
    preset
        A :class:`MaskPreset` or a raw ``opts`` mask, which must be one of
        :data:`INSTANTIATED_MASKS`. Fixed for the lifetime of the object: it
        decides the receive-window layout, so it is not expressible per call.
    dtype
        int32 only. refined-icp-v1 C4 pins the candidate carrier at int32 --
        the fp32 score's raw bits and the int32 block id, both bitcast. The
        parameter exists so a caller passing float32 (the pre-C4 producer
        buffer) gets a message instead of a silent reinterpretation; view it
        with ``t.view(torch.int32)``, never ``.int()`` or ``.to()``.
    device
        Defaults to the current CUDA device. It must be the device this rank
        called ``torch.cuda.set_device`` with -- see the module docstring.

    Allocation happens **here**, eagerly, because symmetric memory cannot be
    allocated during cudagraph capture and ``rendezvous()`` is host-blocking.
    Every rank in ``group`` must construct this object, with identical
    arguments, in the same order.
    """

    def __init__(self, group: dist.ProcessGroup, *, tokens: int,
                 heads_group: int, heads_local: int, slots: int = 57,
                 preset: int = MaskPreset.LAMPORT,
                 dtype: torch.dtype = torch.int32,
                 device: torch.device | int | None = None) -> None:
        import torch.distributed._symmetric_memory as symm_mem  # noqa: PLC0415

        if dtype != torch.int32:
            raise TypeError(
                "refined-icp-v1 C4 pins the candidate carrier at int32 (score "
                "bits and block id, both bitcast, never converted); got "
                f"{dtype}. The pre-refinement float32 carrier held the same two "
                "words, so the byte count is unchanged and no allocation moves "
                "-- but accepting float32 here would let a caller convert "
                "instead of bitcast, which silently corrupts every block id "
                "above 2**24."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("IcpExchange needs CUDA")

        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.T = int(tokens)
        self.Hg = int(heads_group)
        self.Hl = int(heads_local)
        self.slots = int(slots)
        self.dtype = dtype
        #: The allocation capacity, under the name the *extent* argument uses.
        #: ``self.T`` keeps its name because every existing reader of it means
        #: this number; the alias exists so a call site that is talking about
        #: capacity can say so.
        self.capacity = self.T
        if self.Hg != self.world * self.Hl:
            raise ValueError(
                f"C7: H_group ({self.Hg}) must be world ({self.world}) * "
                f"H_local ({self.Hl})"
            )
        if self.slots < 1:
            raise ValueError("slots must be >= 1")
        if self.T <= 0:
            raise ValueError("tokens must be >= 1")
        # C7 pins this; the kernel re-checks it.
        self.head_offset = self.rank * self.Hl

        if device is None:
            index = torch.cuda.current_device()
        elif isinstance(device, int):
            index = device
        else:
            device = torch.device(device)
            if device.type != "cuda":
                raise ValueError(f"device must be a CUDA device, got {device}")
            index = (torch.cuda.current_device() if device.index is None
                     else device.index)
        self.device = torch.device("cuda", index)

        self.mod = _build._k5()
        opt_bits(self.mod)  # fail loudly if the bit numbering drifted
        self.opts = int(preset)
        self.preset = preset
        _validate_mask(self.opts)

        # The ack is derived from `slots` alone:
        #
        #   slots >= 2   a rank running one replay ahead lands in a *different*
        #                slot, so it cannot overwrite a slot a peer has not
        #                read yet (the stream-order argument in
        #                csrc/k5_exchange.cu). The ack becomes redundant, and
        #                dropping it is worth 1.8-4.8 us/layer.
        #   slots == 1   the ack is the sole thing preventing that overwrite.
        #
        # THE `slots >= 2` HALF IS THE SUPERSEDED RULE. Measured 2026-09-09 on
        # hecate0356 (VR200/Rubin sm_107a, job 567080): the real obligation is
        # NO TWO CONSECUTIVE LAUNCHES ON A WORKSPACE MAY USE THE SAME SLOT, and
        # `slots >= 2` does not imply it. At 57 layers with slots=2 the slot
        # sequence 0,1,0,...,0 has last == first, so the replay boundary repeats
        # slot 0. With the merges made different lengths (rather than the
        # launches skewed -- the acquire absorbs that, which is why the older
        # control got 0/4000) that produces 32/128 SILENTLY wrong results at
        # mask 0 with the fault flag clear, and 127/128 detected timeouts at
        # mask 512. The flanking arm with the ack still off but the slot
        # discipline honoured is 0/128: the DISCIPLINE, not the ack, is what
        # safety rests on. What this line still gets right is only the
        # `slots == 1` half. See docs/INTEGRATION.md section 6 and
        # docs/STATUS.md section 1c; the in-tree caller carries the enforcing
        # version of this policy and this package does not.
        #
        # It used to read `self.slots < 2 and not (self.opts & OPT_LAMPORT)`,
        # on the reasoning that Lamport has no handshake to acknowledge. That
        # is wrong about this kernel and the kernel says so at its section 1:
        # OPT_LAMPORT deletes the *publish flag* (control-block kind 0) and
        # leaves the ack (kind 1) exactly as it was, "which is what makes
        # OPT_LAMPORT shippable with use_ack = 1, i.e. without the slot-reuse
        # argument". The old form turned the ack off at `slots == 1` under
        # Lamport -- the one configuration where it is the only protection --
        # and its `slots >= 2` half already covered every other case.
        #
        # Above the bound the ack is off by derivation, and what discharges it
        # is the CALLER's slot discipline -- no two consecutive launches on one
        # slot, wrap included -- which this class asserts and does not check.
        # A violation is a dead workspace under LAMPORT (the tag compare is
        # exact, so the launch times out into `err`) but is SILENT at mask 0.
        self.use_ack = self.slots < ACK_FREE_SLOTS
        # Per-INSTANCE, from this workspace's capacity, head count and mask,
        # rather than a new global default: the number that stops `plan_tpb`
        # widening the block depends on `T_cap` and `H_local`, so a global
        # constant would be right for one geometry and wrong for the next -- and
        # OPT_HIER needs the old one. The whole decision is in
        # `plan_max_blocks`, which takes no GPU and is gated on CPU.
        self.max_blocks = plan_max_blocks(self.mod.k5_plan, tokens=self.T,
                                          heads_local=self.Hl, opts=self.opts)
        self.spin_cycles = DEFAULT_SPIN_CYCLES

        # Grid shape and control-block size both come from the kernel, so host
        # and device agree by construction rather than by convention.
        #
        # Both are taken from the CAPACITY and neither moves with a call's
        # extent. `nblocks` is the control block's stride, so a block index
        # would otherwise address a different generation counter at a different
        # extent; and `tpb` fixes which block owns which row, which is what
        # makes a data word's Lamport tag strictly increasing (one block, one
        # counter, one word). A short call launches fewer of these blocks -- the
        # kernel derives that grid itself -- and changes nothing else.
        self.tpb, self.nblocks = self.mod.k5_plan(self.T, self.Hl,
                                                  self.max_blocks)
        control_words = self.mod.k5_ctl_words(self.slots, self.nblocks,
                                              self.world)

        # Receive-window arithmetic, in fp32 words. One slot holds a full
        # `[T, H_group, 16]` candidate list; two factors scale it:
        #
        #     slot_floats = planes * T * H_group * 16 * words_per_candidate
        #
        #     words_per_candidate  2 normally -- (score, bitcast id)
        #                          4 under OPT_LAMPORT, because the generation
        #                            tag rides in the upper half of every
        #                            64-bit payload word instead of in a
        #                            separate flag. Hence the "2x window" the
        #                            preset is documented as costing.
        #     planes               1 normally: each peer gets only the head
        #                            slice it owns.
        #                          `world` under OPT_BCAST (dropped from this
        #                            build): a multicast store lands at the
        #                            same offset in every peer's window, so
        #                            each peer needs a full plane per source.
        #
        # The whole allocation is `slot_floats * slots` 4-byte words.
        # `k5_slot_floats` is the single source of that number rather than a
        # formula repeated here, because a mask whose receive layout disagrees
        # with the allocation writes past the end of a *peer's* symmetric window
        # -- an overrun no allocator sees and that still returns plausible block
        # ids.
        #
        # The exported name still says "floats". It is deliberately not renamed:
        # the unit is now an int32 word, but the COUNT and the byte size are
        # identical (C4 changed the container, not the payload), and renaming an
        # exported pybind symbol for a quantity that did not change size is an
        # ABI break for nothing.
        self.slot_floats = int(
            self.mod.k5_slot_floats(self.T, self.Hg, self.world, self.opts)
        )
        device_index = self.device.index
        self._data = symm_mem.empty(self.slot_floats * self.slots,
                                    dtype=torch.int32, device=device_index)
        self._ctl = symm_mem.empty(int(control_words), dtype=torch.int32,
                                   device=device_index)
        self._h_data = _rendezvous(symm_mem, self._data, group)
        self._h_ctl = _rendezvous(symm_mem, self._ctl, group)

        # Flags must start at a known value and every rank must observe every
        # other rank's zeroes before the first release. `symm_mem.empty` does
        # not zero.
        self._ctl.zero_()
        self._data.zero_()
        torch.cuda.synchronize()
        self._h_ctl.barrier()
        torch.cuda.synchronize()

        self._err = torch.zeros(1, dtype=torch.int32, device=device_index)
        self._buf_ptrs = list(self._h_data.buffer_ptrs)
        self._ctl_ptrs = list(self._h_ctl.buffer_ptrs)
        assert len(self._buf_ptrs) == self.world == len(self._ctl_ptrs)
        assert self._buf_ptrs[self.rank] == self._data.data_ptr()
        assert self._ctl_ptrs[self.rank] == self._ctl.data_ptr()

        # NVLS multicast mappings. No surviving preset needs one -- the two
        # bits that did (OPT_BCAST, OPT_MCFLAG) were dropped from this build --
        # so these are recorded in `provenance()` and are NOT passed to the
        # kernel. The kernel used to accept them (`buf_mc`, `ctl_mc`), pack them
        # into `PeerPtrs` and dereference them nowhere; reading the handle here
        # is where the fact was actually coming from all along.
        #
        # They are still worth capturing, because a null `multicast_ptr` is a
        # diagnostic in its own right: it is at least as often the symptom of
        # TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES=1 -- which makes
        # CUDASymmetricMemory.cu:853 skip `init_multicast_for_block` and
        # silently disable NVLS multicast -- as of a fabric that genuinely
        # cannot do multicast. Never set that variable to make an
        # overlapping-devices error go away; see item 3 of the module
        # docstring.
        self._buf_mc = int(getattr(self._h_data, "multicast_ptr", 0) or 0)
        self._ctl_mc = int(getattr(self._h_ctl, "multicast_ptr", 0) or 0)
        self._closed = False

    # -- the exchange -----------------------------------------------------

    def exchange_and_merge(self, cand: torch.Tensor, *,
                           out: torch.Tensor | None = None,
                           layer_idx: int = 0,
                           forced: torch.Tensor | None = None,
                           n_ordinary: torch.Tensor | None = None,
                           num_tokens: int | None = None,
                           ) -> torch.Tensor:
        """Publish ``cand``, consume the peers' and write this rank's Top-16.

        Parameters
        ----------
        cand
            ``[N, H_group, 16, 2]`` **int32**, contiguous, on this object's
            device: this rank's candidates for one layer (CONTRACT C4), where
            ``N`` is this call's extent -- ``num_tokens`` if it was given and
            the allocation capacity :attr:`T` otherwise.
            ``[..., 0]`` is the fp32 score's raw bits and ``[..., 1]`` the
            int32 global block id, both **bitcast**. The selector emits this
            buffer as fp32, so pass ``cand.view(torch.int32)`` -- a
            reinterpretation of the same storage, never ``.int()`` or
            ``.to()``, which convert and destroy every id above ``2**24``.
        out
            Optional ``[N, H_local, 16]`` int32 destination, ``N`` as above.
            Allocating here is fine outside a capture; inside one, pass a buffer
            allocated before the capture.
        num_tokens
            This call's **execution extent**: how many token rows are published,
            polled and merged. ``None`` (the default) means the whole allocation
            capacity, which is what this method did before the argument existed.

            The window is allocated once at capacity and a shorter call executes
            over its **prefix**, at the same addresses -- so one instance serves
            every captured extent and the result on the rows they share is
            bit-identical to the full-capacity call. That is the point: a
            symmetric window and its multicast team are the scarce resource, so
            binding one instance per graph size to avoid exchanging padding
            trades the cheap thing for the expensive one.

            **The caller owns two guarantees this class cannot check.**

            1. *Every rank passes the same value.* The extent sets the addresses
               a rank publishes into its peers' windows and the addresses it
               polls in its own; two ranks that disagree do not exchange
               mismatched data, they exchange the wrong rows, and under
               ``MaskPreset.LAMPORT`` the poll simply never matches and the
               launch times out into the error flag. Nothing here can detect it,
               because detecting it needs a collective and this call is on the
               capture path.
            2. *It is a per-graph constant, never a live count.* refined-icp-v1
               C4 requires the exchange shape to be fixed across ranks **and
               across capture replays**. A capture bakes this integer in, so it
               must come from the graph's padded extent -- something the engine
               decided at build time -- and never from ``num_actual_tokens`` or
               any other per-step quantity. Deriving it from a device tensor
               would also be a host read on the submission path, which this
               package refuses everywhere else.

            **Not every mask admits a short extent.** ``MaskPreset.HANDSHAKE``
            sets ``OPT_HIER``, whose release is a single flag for the whole
            grid; the blocks a short launch skips stay a generation behind and
            the grid-wide flag can then satisfy some blocks of a later full
            launch and not others. That is refused here and in the kernel, by
            name. The shipping mask ``MaskPreset.LAMPORT`` and the control mask
            ``0`` both handshake per block -- or per word -- and are gated at
            short extents.

            An extent larger than the capacity **raises**. It is not clamped:
            a clamp would merge fewer rows than the caller asked for and leave
            the rest of ``out`` holding a previous batch's block ids, which a
            block-table load dereferences as real pages.
        layer_idx
            Which slot to use, modulo :attr:`slots`. With one slot per layer
            (the default) two layers of the same step never alias, which is
            what makes the consumer ack redundant. Pass the indexer layer's
            own index; the modulo means any monotonically advancing counter
            also works, it just stops being alias-free once it wraps within a
            step.
        forced, n_ordinary
            CONTRACT C3's row metadata: int32 ``[T]`` planes, **per token row**,
            supplied together or not at all. ``forced[t]`` is row ``t``'s forced
            block ``f = p // 128`` (``-1`` marks an inactive row, which then
            selects nothing) and ``n_ordinary[t]`` is how many ordinary winners
            that row wants, ``min(15, f)``. :func:`fmha_sm100.icp.forced_rows`
            derives both from each row's own query position.

            They are indexed by ``t`` and never by head, because forcing is a
            function of the query position and a per-head plane could express a
            contradiction. Omitting them selects the plain ordinary merge -- the
            behaviour this method had before the planes existed, which is why
            they default to ``None`` rather than being required.

            **Omitting them is not free.** The selector excludes the forced
            column from ordinary ranking unconditionally (CONTRACT C3, "forcing
            by exclusion"), so with no plane to reinject it the forced block is
            absent from *every* row: a query in logical block 31 gets
            ``[0..15]`` where the contract says ``[0..14, 31]``. The row is
            full, ascending and well formed, and the block the query sits in is
            simply not in it.

        Returns ``out``: global block ids, **ascending**, ``-1``-padded.

        Every rank in the group must call this the same number of times, in the
        same order, on the same stream. Shapes are only *validated* here --
        nothing is allocated -- so this call is cudagraph-capturable.

        On failure, and the caller obligation that comes with it
        --------------------------------------------------------

        A launch that fails -- a peer's release that never arrived inside the
        spin timeout, a Lamport tag that never matched, a row whose
        ``n_ordinary`` disagrees with C3's own ``min(15, f)``, or *any* launch
        issued after a previous one failed -- sets the error flag and fills
        **its own block's rows of** ``out`` **with global block id 0**. Not
        ``-1``, and not "leaves them untouched", which is what a previous
        revision of this kernel did.

        The metadata disagreement is in that list rather than clamped because
        ``n_ordinary > 15`` on a forced row makes the merge write one int32 past
        the row. ``k2_merge`` reports the same disagreement through its C5
        status word; this kernel has no status word and reports it through the
        error flag, so the two backends refuse the same inputs and render the
        refusal differently.

        Why the fill exists: ``out`` is typically a buffer shared by every
        indexer layer and consumed by attention *in the same step*, before the
        host has any opportunity to poll the flag. An untouched row reaching a
        block-table load holds either uninitialised memory -- an arbitrary
        block id, so an arbitrary page, so a wild address -- or a previous
        batch's ids, which read a neighbouring request's pages and return
        plausible tokens drawn from another request's KV.

        Why ``0`` and not ``-1``: the consumer this was designed against loads
        ``page = block_table_row[blk]`` with no mask on ``blk``, so ``-1``
        dereferences one element *before* the row. ``0`` is in range for every
        row that is read at all, because a row's valid length is
        ``min(max_topk, cdiv(kv_len, block_size))`` and is zero for padded and
        zero-length rows.

        **The precondition, which this API cannot express and therefore states.**
        That argument is a **decode** argument. It holds for a consumer that
        indexes a block table bounded by ``kv_len``. A *prefill* consumer that
        feeds these indices to a CSR builder (upstream: ``build_k2q_csr``, a
        third-party binary) sees a row containing sixteen duplicate copies of
        block 0, which is exactly the shape that can overflow a per-row CSR
        segment. :class:`IcpExchange` has no notion of decode versus prefill,
        deliberately, so:

            **If** ``out`` **may be consumed on a prefill path, poll**
            :meth:`check_error` **before handing it to that consumer.** Do not
            rely on the fill to make a failed launch safe there. On a decode
            path the fill is what makes the window between the fault and the
            poll memory-safe; on a prefill path the poll is the only guarantee.

        Failure is also **sticky within the workspace**: the kernel reads the
        error flag on entry and every subsequent launch returns immediately
        (filling its rows with 0) instead of spinning to its own timeout. That
        turns a step of 57 layers x 1 s of spin into one timeout. See
        :meth:`check_error` for what clearing the flag does and does not mean.
        """
        if self._closed:
            raise RuntimeError("IcpExchange is closed")
        # The extent. Refused above capacity and never clamped; see the
        # docstring. `int()` of a tensor would be a device->host read, so a
        # tensor is refused by type rather than silently synchronised.
        if num_tokens is None:
            extent = self.T
        else:
            if isinstance(num_tokens, torch.Tensor):
                raise TypeError(
                    "num_tokens must be a host int -- a per-GRAPH constant. "
                    "Reading it off a device tensor would synchronise on the "
                    "submission path and would make the exchange shape depend "
                    "on a live count, which refined-icp-v1 C4 forbids."
                )
            extent = int(num_tokens)
            if extent < 1:
                raise ValueError(f"num_tokens must be >= 1; got {extent}")
            if extent > self.T:
                raise ValueError(
                    f"num_tokens={extent} exceeds this exchange's allocated "
                    f"capacity of {self.T} token rows. Symmetric memory cannot "
                    "be resized (and cannot be allocated at all during "
                    "cudagraph capture), so construct the IcpExchange at the "
                    "largest extent you will ever replay. This is refused "
                    "rather than clamped: a clamp would merge the first "
                    f"{self.T} rows and leave the rest of `out` holding a "
                    "previous batch's block ids."
                )
            if extent != self.T and self.opts & OPT_HIER:
                # Mirrors the kernel's own refusal, with the reason, because
                # under a cudagraph the kernel's first chance to say it may be
                # the first CAPTURED launch. OPT_HIER publishes one release for
                # the whole grid and every block acquires on it, so all blocks
                # must be on the same generation -- and a block a short launch
                # skips is left a generation behind, after which the grid-wide
                # flag can satisfy some of the blocks of a later full launch and
                # not others. The acquire compares `>=`, so the ones below it
                # pass on another block's generation and the ones above it spin
                # to their timeout, decided by arrival order. Refused, because
                # there is no safe reading of it.
                raise ValueError(
                    f"num_tokens={extent} is shorter than this exchange's "
                    f"{self.T}-row allocation, and its mask sets OPT_HIER "
                    f"(opts={self.opts}). OPT_HIER's release is ONE flag for "
                    "the whole grid, so every block must be on the same "
                    "generation and a block that a short launch skips falls "
                    "behind. Build the workspace with MaskPreset.LAMPORT (the "
                    "shipping mask) or the control mask 0 if it must serve "
                    "several extents."
                )
        extent_source = ("num_tokens" if num_tokens is not None
                         else "this exchange's allocation capacity")
        expected_shape = (extent, self.Hg, CAND_K, 2)
        if tuple(cand.shape) != expected_shape:
            raise ValueError(
                f"cand must be {list(expected_shape)} -- the leading dimension "
                f"is this call's extent, from {extent_source}; got "
                f"{list(cand.shape)}. Pass the capacity buffer's prefix view "
                "`cand[:num_tokens]`, which is contiguous and allocates "
                "nothing."
            )
        if cand.dtype != torch.int32:
            raise TypeError(
                "cand must be int32 (refined-icp-v1 C4: score bits and block "
                f"id, both bitcast); got {cand.dtype}. Reinterpret a float32 "
                "producer buffer with `t.view(torch.int32)` -- never `.int()` "
                "or `.to()`, which convert and destroy every id above 2**24."
            )
        if not cand.is_contiguous():
            raise ValueError("cand must be contiguous")
        if cand.device != self.device:
            raise ValueError(
                f"cand is on {cand.device} but this exchange rendezvoused on "
                f"{self.device}"
            )
        if out is None:
            out = torch.empty((extent, self.Hl, CAND_K), dtype=torch.int32,
                              device=self.device)
        elif tuple(out.shape) != (extent, self.Hl, CAND_K):
            raise ValueError(
                f"out must be {[extent, self.Hl, CAND_K]} -- this call's "
                f"extent, from {extent_source}; got {list(out.shape)}"
            )
        elif out.dtype != torch.int32:
            raise TypeError(f"out must be int32, got {out.dtype}")

        # C3's planes. Both or neither: one alone is a caller that thinks it is
        # forcing and is not. Everything checked here is a shape or a dtype --
        # no value is read, so this stays cudagraph-capturable, and the values
        # are checked against C3's formula on the device.
        if (forced is None) != (n_ordinary is None):
            raise ValueError(
                "forced and n_ordinary must be supplied together (C3): got "
                f"forced={'a tensor' if forced is not None else None}, "
                f"n_ordinary="
                f"{'a tensor' if n_ordinary is not None else None}"
            )
        if forced is not None:
            for name, plane in (("forced", forced),
                                ("n_ordinary", n_ordinary)):
                if plane.dtype != torch.int32:
                    raise TypeError(
                        f"{name} must be int32 -- these are block ids and "
                        f"counts, never floats; got {plane.dtype}"
                    )
                if not plane.is_contiguous():
                    raise ValueError(f"{name} must be contiguous")
                if plane.device != self.device:
                    raise ValueError(
                        f"{name} is on {plane.device} but this exchange "
                        f"rendezvoused on {self.device}"
                    )
                if plane.numel() != extent:
                    raise ValueError(
                        f"{name} is a per-TOKEN-ROW plane of length "
                        f"{extent} -- this call's extent, from "
                        f"{extent_source}; got {plane.numel()}. It is indexed "
                        "by t only -- forcing is a function of the row's query "
                        "position, not of the head."
                    )

        # Slot rotation. `layer_idx` picks one of `slots` independent
        # generations, so with one slot per indexer layer (the default, 57 for
        # MiniMax-M3) no two layers of a step share a window and layer N never
        # waits on layer N-1's window being drained. The modulo makes
        # `slots == 1` degenerate to a single shared window -- which is exactly
        # the case `use_ack` above turns the acknowledgement back on for.
        slot = int(layer_idx) % self.slots
        self.mod.k5_exchange(
            cand,
            out,
            self._buf_ptrs,
            self._ctl_ptrs,
            self._err,
            None,          # seq: derived on device as own_generation + 1.
            self.rank,
            self.world,
            self.head_offset,
            slot,
            self.slots,
            MODE_PUSH,
            1,             # do_publish
            1,             # acquire_on
            1 if self.use_ack else 0,
            self.spin_cycles,
            self.max_blocks,
            self.opts,
            # TWENTY-TWO positional arguments; `slot_capacity_floats` is the
            # last of the nineteen that predate C3's planes, the planes follow
            # it, and `tokens_capacity` follows them. An earlier revision passed
            # twenty-two of a different shape: `buf_mc`,
            # `ctl_mc` and a `dbg` counter tensor went between `opts` and this
            # one.
            #
            # `buf_mc`/`ctl_mc` were packed into the kernel's `PeerPtrs` and
            # dereferenced by nothing -- the NVLS store path went with
            # OPT_BCAST/OPT_MCFLAG -- so the only consumer of multicast-ness is
            # `provenance()` below, which reads `handle.multicast_ptr` in Python
            # and never asked the kernel anyway. `dbg` was real but unreachable
            # through this API (it was hardcoded `None` right here) and cannot
            # be re-added for free: it is a kernel *parameter*, so it moves every
            # specialisation's constant bank, and it puts a null test inside the
            # spin loop -- both of which break the `opts = 0` SASS-identity the
            # control arm rests on. Consequence, stated rather than hidden: this
            # package currently has NO mechanism for proving that a poll
            # actually spun, so a slots gate here cannot rule out passing
            # vacuously.
            #
            # Under OPT_LAMPORT the kernel REFUSES a non-positive capacity: the
            # Lamport window is 2x the ordinary layout, and without the capacity
            # the launch cannot tell whether the allocation it was handed was
            # made for this mask. `self.slot_floats` came from
            # `k5_slot_floats(..., self.opts)`, so the mask that sized the
            # window is the mask that exchanges through it, by construction.
            self.slot_floats,
            # C3's planes are the last two arguments of the kernel entry point,
            # not neighbours of `out` where `k2_merge` keeps them, because this
            # entry point is called positionally and appending is the only way
            # to add to it without rebinding the nineteen above. `None` selects
            # the plain ordinary merge.
            forced,
            n_ordinary,
            # The ALLOCATION capacity, and the reason the extent above is free
            # to be smaller. Every stride in the receive window is taken from
            # this number, so `cand`'s own leading dimension says only how many
            # rows to do, never where they live. Always passed -- when the
            # extent IS the capacity the kernel's arithmetic is unchanged.
            self.T,
        )
        return out

    # -- diagnostics ------------------------------------------------------

    def check_error(self) -> None:
        """Raise if any exchange since the last check reported a failure.

        Host-synchronising, so it belongs at the end of a step or in a test --
        not inside a captured graph. The kernel sets this flag instead of
        spinning forever, because a hang inside a cudagraph replay is far worse
        to debug than a reported failure: a mismatched call sequence between
        ranks would otherwise wedge the whole job with no diagnostic.

        **Known gap: this clears the flag, and the kernel treats a set flag as
        "this workspace is dead".** The kernel reads the flag on entry, so while
        it is set every launch short-circuits and fills its rows with 0. Clearing
        it here therefore re-arms a workspace whose ranks are, by construction,
        a generation apart: a block that failed returned before its ack and
        before recording its generation while its peers may have done both.
        Nothing here proves that re-arming is safe, and nothing here refuses it
        either. Treat a raised error as fatal to the workspace -- close it and
        build another -- until this class grows the poisoning that would enforce
        that.
        """
        torch.cuda.synchronize()
        err_code = int(self._err.item())
        if err_code == 0:
            return
        # Cleared on read, so one reported failure is not re-raised forever.
        # See the docstring: this is a diagnostic convenience, not a recovery.
        self._err.zero_()
        raise RuntimeError(
            "k5 exchange failed (err flag set). Either a peer's release never "
            "arrived within the spin timeout -- which means the ranks in this "
            "group did not issue the same call sequence on the same stream -- "
            "or, under MaskPreset.LAMPORT, a tag never matched this "
            "generation, or an explicit generation of 0 was used, which a "
            "zero-filled window cannot be distinguished from. The failing "
            "blocks' rows of `out` hold block id 0; every launch issued after "
            "the first failure short-circuited and did the same. This "
            "workspace should be considered dead: close it and build another."
        )

    @property
    def control_block(self) -> torch.Tensor:
        """The raw uint32 control block, for tests that inspect the protocol."""
        return self._ctl

    def provenance(self) -> dict[str, object]:
        """What is actually running, for a CSV row or a bug report."""
        return {
            "preset": getattr(self.preset, "name", None),
            "opts": self.opts,
            "world": self.world,
            "T": self.T,
            "H_group": self.Hg,
            "H_local": self.Hl,
            "slots": self.slots,
            "use_ack": int(self.use_ack),
            "tpb": self.tpb,
            "nblocks": self.nblocks,
            # Per-instance since `plan_max_blocks`, so it belongs in the row:
            # `tpb`, `nblocks` and the control block all follow from it, and it
            # is the only one of the four a reader can change.
            "max_blocks": self.max_blocks,
            "slot_floats": self.slot_floats,
            "symm_mib": f"{self._data.numel() * 4 / 2 ** 20:.2f}",
            # The control block is symmetric memory too, and it is the part
            # that scales with `nblocks` rather than with the capacity. Reported
            # separately because `symm_mib` above has always been the DATA
            # window alone, and a reader comparing two `max_blocks` needs the
            # number that moved.
            "ctl_mib": f"{self._ctl.numel() * 4 / 2 ** 20:.2f}",
            "nvls_multicast": int(bool(self._buf_mc)),
            "arch": _build.arch(),
        }

    # -- lifetime ---------------------------------------------------------

    def close(self) -> None:
        """Release the symmetric memory, collectively and deterministically.

        Every rank must call this, and no rank may free a window a peer is
        still reading -- that is a use-after-free *across processes*, which the
        CUDA allocator cannot see. So: drain the local stream, barrier over the
        group, drain again, then drop the last references and let the
        symmetric-memory allocator reclaim them.

        Any surviving reference to the buffers (a saved view, a traced graph)
        pins the allocation, which is why this class does not hand out views of
        the window.
        """
        if self._closed:
            return
        self._closed = True
        torch.cuda.synchronize()
        try:
            self._h_ctl.barrier()
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001 - teardown must not mask the real error
            pass
        self._h_data = None
        self._h_ctl = None
        self._data = None
        self._ctl = None
        self._buf_ptrs = []
        self._ctl_ptrs = []

    def __enter__(self) -> IcpExchange:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"IcpExchange(rank={self.rank}/{self.world}, T={self.T}, "
                f"Hg={self.Hg}, Hl={self.Hl}, slots={self.slots}, "
                f"opts={self.opts})")


def _validate_mask(opts: int) -> None:
    """Reject the masks the host side of the kernel would reject, with context.

    Every rule here is mirrored by a ``TORCH_CHECK`` in ``csrc/k5_exchange.cu``,
    which remains the real gate. Catching them here buys two things: the message
    names the bit and the reason instead of an integer, and it fires at
    **construction** rather than at first launch -- which under a cudagraph may
    be the first *captured* launch.

    The rules, in the order the kernel applies them:

    * a negative mask is not a mask (checked first, so a garbage argument is
      never blamed on a named bit -- ``-1 & OPT_SELF_ELIDE`` is 1);
    * un-dispatched bits are refused **by name**: ``OPT_SELF_ELIDE`` (retired
      with mask 513) and ``OPT_NODIV`` (implemented, never dispatched here). An
      anonymous "unsupported bit" would be true and useless;
    * any other unimplemented bit is an error, never a silent fall back to the
      control -- a silent fallback runs and times the baseline under the
      optimisation's name, and because every rank must run the same protocol, a
      rank that quietly dropped a bit its peers kept would desynchronise the
      handshake;
    * ``OPT_LAMPORT`` excludes ``OPT_HIER``. Lamport deletes the
      release/acquire handshake outright and ``OPT_HIER`` exists only to
      optimise one, so the combination is dead code in a measured mask;
    * finally, the mask must be one the kernel actually instantiates. A
      supported-bit combination that no ``case`` builds is exactly the failure
      the kernel is organised to prevent.
    """
    if opts < 0:
        raise ValueError(f"opts mask must be non-negative; got {opts}")
    if opts & OPT_SELF_ELIDE:
        raise ValueError(
            f"opts {opts} sets OPT_SELF_ELIDE = 1, which was RETIRED together "
            "with mask 513 (OPT_LAMPORT | OPT_SELF_ELIDE), its only "
            "instantiation: it measured 0.09-0.11 us/layer at C=2 and a null "
            "at C=4, and 513 sat within +/-0.4 us of plain 512. The bit value "
            "stays reserved and the kernel's device code still implements it; "
            "see the OPT_SELF_ELIDE refusal in csrc/k5_exchange.cu for how to "
            "bring it back."
        )
    if opts & OPT_NODIV:
        raise ValueError(
            f"opts {opts} sets OPT_NODIV = 256, which is implemented in the "
            "kernel's device code and instantiated by no dispatch case. It is "
            "dormant rather than removed: it hoists the per-element division "
            "out of both publish loops, but its register saving did not "
            "reproduce across architectures. Mask 768 was this "
            "package's LAMPORT preset before this revision; it is 512 now. See "
            "the OPT_NODIV refusal in csrc/k5_exchange.cu for what landing it "
            "would take."
        )
    if opts & ~SUPPORTED_OPTS:
        raise ValueError(
            f"opts {opts} sets a bit this revision does not implement "
            f"(supported: {SUPPORTED_OPTS} = OPT_NOFENCE | OPT_HIER | "
            "OPT_LAMPORT). The remaining values are reserved for the upstream "
            "harness bits of the same numbering and are not silently ignored."
        )
    if opts & OPT_LAMPORT and opts & OPT_HIER:
        raise ValueError(
            "OPT_LAMPORT removes the handshake entirely; OPT_HIER optimises a "
            "handshake that is no longer there and would be silently dead code"
        )
    if int(opts) not in INSTANTIATED_MASKS:
        raise ValueError(
            f"opts mask {opts} is not instantiated. The kernel builds exactly "
            f"{sorted(INSTANTIATED_MASKS)}: 0 (the control), 12 "
            "(MaskPreset.HANDSHAKE) and 512 (MaskPreset.LAMPORT). Masks 4 and 8 "
            "are the single-bit ablation arms -- both bits still ship, inside "
            "12 -- and were dropped because neither is best at every token "
            "count alone while 12 is, and 12 is in turn beaten by 512 in 24 of "
            "24 measured cells."
        )


# --------------------------------------------------------------------------
# functional wrapper (CONTRACT C7's Impl B signature)
# --------------------------------------------------------------------------

_CACHE: dict[tuple, IcpExchange] = {}


def exchange_and_merge(cand: torch.Tensor, group: dist.ProcessGroup, *,
                       out: torch.Tensor | None = None,
                       head_offset: int, num_heads_local: int,
                       layer_idx: int = 0, slots: int = 57,
                       preset: int = MaskPreset.LAMPORT,
                       forced: torch.Tensor | None = None,
                       n_ordinary: torch.Tensor | None = None,
                       capacity: int | None = None,
                       ) -> torch.Tensor:
    """CONTRACT C7 Impl B, behind the same signature as the NCCL Impl A.

    Convenience over :class:`IcpExchange` for scripts and tests. It caches one
    exchange per ``(group, T, H_group, H_local, slots, preset)`` because
    allocating and rendezvousing symmetric memory per call would be both slow
    and collective. That cache makes this **unsafe to call for the first time
    inside a cudagraph capture** -- construct the :class:`IcpExchange` yourself
    for anything captured, or warm this up first.

    ``head_offset`` is checked against C7's ``icp_rank * num_heads_local``
    rather than used: no other value is expressible, and a caller passing a
    different one has a bug the kernel would otherwise turn into a silent
    head-window shift.

    ``forced`` and ``n_ordinary`` are forwarded unchanged; see
    :meth:`IcpExchange.exchange_and_merge` for what omitting them costs. They
    are **not** part of the cache key, because they are per-call data rather
    than part of the symmetric window's shape.

    ``capacity`` is the window to build, defaulting to ``cand``'s own row count.
    Give it once -- the largest extent this process will ever present -- and
    every shorter ``cand`` then runs as a prefix through the **same** window
    instead of rendezvousing another one. It is the cache key's token entry for
    exactly that reason: the extent is per call, the window is not.
    """
    tokens, heads_group = int(cand.shape[0]), int(cand.shape[1])
    window_tokens = tokens if capacity is None else int(capacity)
    # Every argument that sizes the symmetric window is part of the key, so a
    # changed shape builds a new exchange instead of misusing an old one. The
    # group *object* is the key, not `id(group)`: the key must keep the group
    # alive, or a collected group's id could be reused by a different one and
    # silently hand back the wrong symmetric window.
    cache_key = (group, window_tokens, heads_group, int(num_heads_local),
                 int(slots), int(preset), cand.device.index)
    exchange = _CACHE.get(cache_key)
    if exchange is None:
        exchange = IcpExchange(group, tokens=window_tokens,
                               heads_group=heads_group,
                               heads_local=int(num_heads_local),
                               slots=int(slots), preset=preset,
                               device=cand.device)
        _CACHE[cache_key] = exchange
    if int(head_offset) != exchange.head_offset:
        raise ValueError(
            f"C7 pins head_offset = icp_rank * H_local = "
            f"{exchange.head_offset}; got {head_offset}"
        )
    return exchange.exchange_and_merge(cand, out=out, layer_idx=layer_idx,
                                       forced=forced, n_ordinary=n_ordinary,
                                       num_tokens=tokens)


def close_all() -> None:
    """Close every cached :func:`exchange_and_merge` window. Collective."""
    for exchange in list(_CACHE.values()):
        exchange.close()
    _CACHE.clear()
