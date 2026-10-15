from rag.application.config import ChunkingConfig
from rag.application.use_cases.ingest_corpus import IngestCorpus, IngestReport
from rag.domain.models import Document
from tests.fakes import FakeEmbedder, FakeVectorStore

DOCUMENTS = [
    Document(
        text="# Guía\n\nIntroducción.\n\n## Instalación\n\nEjecuta el instalador.",
        source="guia.md",
        title="Guía",
    ),
    Document(text="Texto sin títulos.", source="notas.md", title="Notas"),
]


def count_words(text: str) -> int:
    return len(text.split())


def make_ingest(embedder, store, batch_size=100):
    config = ChunkingConfig(max_tokens=50, overlap_tokens=5)
    return IngestCorpus(embedder, store, count_words, config, batch_size=batch_size)


def test_ingest_stores_every_chunk_and_reports_the_totals():
    store = FakeVectorStore()

    report = make_ingest(FakeEmbedder(), store)(DOCUMENTS)

    assert report == IngestReport(documents=2, chunks=3)
    assert sorted(chunk.text for chunk, _ in store.rows.values()) == [
        "Ejecuta el instalador.",
        "Introducción.",
        "Texto sin títulos.",
    ]


def test_text_sent_to_the_model_starts_with_the_section_or_the_title():
    embedder = FakeEmbedder()

    make_ingest(embedder, FakeVectorStore())(DOCUMENTS)

    assert embedder.batches == [
        [
            "Guía\n\nIntroducción.",
            "Guía > Instalación\n\nEjecuta el instalador.",
            "Notas\n\nTexto sin títulos.",
        ]
    ]


def test_ingesting_twice_does_not_duplicate_chunks():
    store = FakeVectorStore()
    ingest = make_ingest(FakeEmbedder(), store)

    ingest(DOCUMENTS)
    ingest(DOCUMENTS)

    assert len(store.rows) == 3


def test_chunks_are_embedded_in_batches():
    embedder = FakeEmbedder()

    make_ingest(embedder, FakeVectorStore(), batch_size=2)(DOCUMENTS)

    assert [len(batch) for batch in embedder.batches] == [2, 1]
