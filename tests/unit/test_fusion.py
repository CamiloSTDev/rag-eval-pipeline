import pytest

from rag.domain.fusion import reciprocal_rank_fusion
from rag.domain.models import Chunk, RetrievedChunk


def ranking(*ids: str, score: float = 0.9) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            Chunk(chunk_id=i, text=i, source="doc.md", page=None, section=None, position=0), score
        )
        for i in ids
    ]


def test_fusion_matches_the_scores_computed_by_hand():
    # a: 1.º vectorial y 3.º BM25 → 1/61 + 1/63; c: 3.º y 1.º → lo mismo; b y d: 1/62
    fused = reciprocal_rank_fusion([ranking("a", "b", "c"), ranking("c", "d", "a")], k=60)

    assert [result.chunk.chunk_id for result in fused] == ["a", "c", "b", "d"]
    assert [result.score for result in fused] == pytest.approx(
        [1 / 61 + 1 / 63, 1 / 61 + 1 / 63, 1 / 62, 1 / 62]
    )


def test_consensus_beats_a_single_first_place():
    # y es 2.º en las dos listas; x y z son 1.º en una sola
    fused = reciprocal_rank_fusion([ranking("x", "y"), ranking("z", "y")], k=60)

    assert fused[0].chunk.chunk_id == "y"


def test_only_positions_matter_not_the_original_scores():
    tiny_scores = ranking("a", "b", score=0.01)
    huge_scores = ranking("b", "a", score=25.0)

    fused = reciprocal_rank_fusion([tiny_scores, huge_scores], k=60)

    assert fused[0].score == pytest.approx(fused[1].score)


def test_a_smaller_k_gives_more_weight_to_the_first_places():
    rankings = [ranking("x", "y"), ranking("z", "y")]

    with_k60 = reciprocal_rank_fusion(rankings, k=60)
    with_k1 = reciprocal_rank_fusion(rankings, k=1)

    # y (2.º en las dos) sigue ganando, pero con k=1 un 1.er puesto vale mucho más que un 2.º
    # (1/2 contra 1/3), así que su ventaja sobre x se achica: de 1,97 veces a 1,33
    assert with_k60[0].chunk.chunk_id == with_k1[0].chunk.chunk_id == "y"
    assert with_k60[0].score / with_k60[1].score == pytest.approx(2 / 62 / (1 / 61))
    assert with_k1[0].score / with_k1[1].score == pytest.approx((2 / 3) / (1 / 2))
