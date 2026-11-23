import pytest

from rag.domain.metrics import hit_at_k, percentile, reciprocal_rank
from rag.domain.models import Chunk, SourceRef


def test_hit_at_k_looks_only_at_the_first_k_positions():
    relevant_at_3 = [False, False, True, False]

    assert hit_at_k(relevant_at_3, 5) == 1
    assert hit_at_k(relevant_at_3, 2) == 0
    assert hit_at_k([], 5) == 0


def test_reciprocal_rank_rewards_finding_the_answer_higher():
    assert reciprocal_rank([True, False]) == 1.0
    assert reciprocal_rank([False, False, True, True]) == pytest.approx(1 / 3)
    assert reciprocal_rank([False, False]) == 0.0


def test_percentile_uses_the_nearest_rank():
    latencies = [100.0, 200.0, 300.0, 400.0, 5000.0]

    assert percentile(latencies, 50) == 300.0
    assert percentile(latencies, 95) == 5000.0
    with pytest.raises(ValueError):
        percentile([], 50)


def make_chunk(source: str, section: str | None, page: int | None = None) -> Chunk:
    return Chunk(chunk_id="x", text="t", source=source, page=page, section=section, position=0)


def test_source_ref_without_section_accepts_any_chunk_of_the_file():
    ref = SourceRef("tutorial/query-params.md")

    assert ref.matches(make_chunk("tutorial/query-params.md", "Query > Opcionales"))
    assert not ref.matches(make_chunk("tutorial/path-params.md", "Query > Opcionales"))


def test_source_ref_with_section_or_page_requires_an_exact_match():
    by_section = SourceRef("query.md", section="Query > Opcionales")
    by_page = SourceRef("manual.pdf", page=4)

    assert by_section.matches(make_chunk("query.md", "Query > Opcionales"))
    assert not by_section.matches(make_chunk("query.md", "Query > Opcionales > Ejemplo"))
    assert by_page.matches(make_chunk("manual.pdf", None, page=4))
    assert not by_page.matches(make_chunk("manual.pdf", None, page=5))
