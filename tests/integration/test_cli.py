import pytest

from rag import bootstrap
from rag.interfaces import cli


def test_ingest_if_empty_skips_an_index_that_already_has_chunks(monkeypatch, capsys):
    monkeypatch.setattr(bootstrap, "build_index_size", lambda settings: lambda: 1117)

    def must_not_ingest(settings):
        raise AssertionError("no debía reingerir")

    monkeypatch.setattr(bootstrap, "build_ingest_corpus", must_not_ingest)

    assert cli.main(["ingest", "--if-empty"]) == 0
    assert "ya tiene 1117 chunks" in capsys.readouterr().out


def test_ingest_if_empty_ingests_an_empty_index(monkeypatch):
    monkeypatch.setattr(bootstrap, "build_index_size", lambda settings: lambda: 0)

    class Ingested(Exception):
        pass

    def ingest(settings):
        raise Ingested

    monkeypatch.setattr(bootstrap, "build_ingest_corpus", ingest)

    with pytest.raises(Ingested):
        cli.main(["ingest", "--if-empty"])
