"""CPU byte and score-bound gates for the TP2/P128 NVFP4 compound page layout."""

import pytest

torch = pytest.importorskip("torch")

from fmha_sm100.icp.compound_page import (
    CompoundPageError,
    CompoundPageLayout,
    SparseMainFormat,
    icp_local_read_bounds,
)

PAGE_BYTES = 45_056
MAIN_BYTES = 36_864
INDEX_BYTES = 8_192


@pytest.fixture
def layout():
    return CompoundPageLayout.build(tp_size=2, main_format=SparseMainFormat.NVFP4)


def test_page_geometry_is_the_current_packed_main_plus_owned_index(layout):
    assert layout.main_shape == (4, 128, 72)
    assert layout.main_span_bytes == MAIN_BYTES
    assert layout.index_shape == (64, 128)
    assert layout.index_span_bytes == INDEX_BYTES
    assert layout.index_offset_bytes == MAIN_BYTES
    assert layout.compound_span_bytes == PAGE_BYTES


@pytest.mark.parametrize("rank", [0, 1])
def test_read_bounds_match_independently_enumerated_owned_tokens(rank):
    positions = [-1, 0, 1, 62, 63, 64, 65, 126, 127, 128, 129, 191, 255, 511]
    lengths = [0, 1, 63, 64, 65, 127, 128, 129, 192, 257, 600]
    pairs = [(p, n) for p in positions for n in lengths]
    nvalid, forced = icp_local_read_bounds(
        torch.tensor([p for p, _ in pairs], dtype=torch.int64),
        torch.tensor([n for _, n in pairs], dtype=torch.int32),
        physical_page_tokens=128,
        world_size=2,
        rank=rank,
    )
    expected_counts, expected_forced = [], []
    for position, length in pairs:
        visible_tokens = range(max(0, min(position + 1, length)))
        owned_pages = {
            token // 128 for token in visible_tokens if (token % 128) // 64 == rank
        }
        expected_counts.append(len(owned_pages))
        current_page = position // 128
        expected_forced.append(current_page if current_page in owned_pages else -1)
    assert nvalid.dtype == torch.int32
    assert forced.dtype == torch.int32
    assert nvalid.tolist() == expected_counts
    assert forced.tolist() == expected_forced


@pytest.mark.parametrize("rank", [0, 1])
def test_native_context_bounds_are_uncapped_and_tail_exact(rank):
    nvalid, forced = icp_local_read_bounds(
        torch.tensor([1_048_575, 1_048_512, 1_048_512, 1_048_575]),
        torch.tensor([1_048_576, 1_048_513, 1_048_512, 0]),
        physical_page_tokens=128,
        world_size=2,
        rank=rank,
    )
    expected = [8192, 8192, 8192, 0] if rank == 0 else [8192, 8192, 8191, 0]
    assert nvalid.tolist() == expected
    expected_forced = [8191, 8191, 8191, -1] if rank == 0 else [8191, 8191, -1, -1]
    assert forced.tolist() == expected_forced


@pytest.mark.parametrize("tp_size", [0, 1, 3, 4, 8])
def test_rejects_unqualified_tp_width(tp_size):
    with pytest.raises(CompoundPageError, match="TP2"):
        CompoundPageLayout.build(tp_size=tp_size, main_format=SparseMainFormat.NVFP4)


@pytest.mark.parametrize("page_tokens", [64, 256, 512])
def test_rejects_unqualified_page_size(page_tokens):
    with pytest.raises(CompoundPageError, match="P128"):
        CompoundPageLayout.build(
            tp_size=2,
            main_format=SparseMainFormat.NVFP4,
            physical_page_tokens=page_tokens,
        )


@pytest.mark.parametrize("main_format", ["fp8_e4m3", "nvfp4_4over6"])
def test_rejects_unqualified_sparse_format(main_format):
    with pytest.raises((CompoundPageError, ValueError)):
        CompoundPageLayout.build(tp_size=2, main_format=main_format)
