import pymupdf

from rag.infrastructure.loaders import load_corpus
from rag.infrastructure.loaders.html import load_html
from rag.infrastructure.loaders.markdown import clean_markdown, load_markdown
from rag.infrastructure.loaders.pdf import load_pdf

MKDOCS_PAGE = """\
---
include_yaml:
  sponsors: data/sponsors.yml
---

# Parámetros de Query { #query-parameters }

Se <dfn title="a datos de Python">parsean</dfn> solos. Ver [la guía](https://x.org) y `<form>`.

{* ../../docs_src/query_params/tutorial001_py310.py hl[9] *}

<!-- sponsors -->
<img src="logo.png">

/// tip | Consejo

Usa tipos.

///

```Python
{!../../docs_src/app.py!}
```

```Python
# un comentario
x = "<b>no tocar</b>"
```
"""

CLEAN_PAGE = """\
# Parámetros de Query

Se parsean (a datos de Python) solos. Ver la guía y `<form>`.

**Consejo**

Usa tipos.

```Python
# un comentario
x = "<b>no tocar</b>"
```"""


def test_clean_markdown_removes_mkdocs_syntax_and_leaves_code_untouched():
    assert clean_markdown(MKDOCS_PAGE) == CLEAN_PAGE


def test_load_markdown_takes_the_title_from_the_first_heading(tmp_path):
    path = tmp_path / "query-params.md"
    path.write_text(MKDOCS_PAGE, encoding="utf-8")

    [document] = load_markdown(path, "tutorial/query-params.md")

    assert document.title == "Parámetros de Query"
    assert document.source == "tutorial/query-params.md"
    assert document.page is None


def test_load_corpus_reads_supported_files_in_order_and_skips_excluded(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ["guia.md", "sub/otra.md", "sub/_prueba.md", "banner.md"]:
        (tmp_path / name).write_text("# Título\n\nTexto.", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("MIT", encoding="utf-8")

    documents = load_corpus(tmp_path, exclude=["_*.md", "banner.md"])

    assert [document.source for document in documents] == ["guia.md", "sub/otra.md"]


def test_load_pdf_gives_one_document_per_page_without_repeated_header_or_footer(tmp_path):
    path = tmp_path / "manual.pdf"
    with pymupdf.open() as pdf:
        for number in range(1, 4):
            page = pdf.new_page()
            page.insert_text((72, 50), "Manual de prueba")
            page.insert_text((72, 300), f"La infor-\nmación importante número {number}.")
            page.insert_text((72, 800), f"Página {number}")
        pdf.set_metadata({"title": "Manual"})
        pdf.save(path)

    documents = load_pdf(path, "manual.pdf")

    assert [document.page for document in documents] == [1, 2, 3]
    assert [document.text for document in documents] == [
        f"La información importante número {number}." for number in range(1, 4)
    ]
    assert all(document.title == "Manual" for document in documents)


def test_load_html_keeps_the_main_content_as_markdown(tmp_path):
    paragraph = "FastAPI es un framework web moderno para construir APIs con Python. " * 4
    path = tmp_path / "guia.html"
    path.write_text(
        f"""<html><head><title>Guía</title></head><body>
        <nav>Inicio | Tutorial | Ayuda</nav>
        <main><article>
          <h1>Primeros pasos</h1><p>{paragraph}</p>
          <h2>Instalación</h2><p>{paragraph}</p>
        </article></main>
        <footer>Pie de página del sitio</footer>
        </body></html>""",
        encoding="utf-8",
    )

    [document] = load_html(path, "guia.html")

    assert document.text.startswith("# Primeros pasos\n\nFastAPI es un framework")
    assert "\n## Instalación\n" in document.text
    assert "Inicio | Tutorial" not in document.text
    assert "Pie de página" not in document.text
