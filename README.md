# RAG con evaluación

Asistente de preguntas y respuestas sobre la documentación de FastAPI en español, con búsqueda híbrida, citas obligatorias y una evaluación medible que compara configuraciones.

Estado: en construcción (semana 1 de 4). Ya funcionan la ingesta y la búsqueda vectorial por línea de comandos.

## Preparar el entorno

Requiere [uv](https://docs.astral.sh/uv/).

```
uv sync
```

Copia `.env.example` como `.env` y pon tu clave de OpenAI en `OPENAI_API_KEY`.

## Uso

Construye el índice vectorial a partir del corpus:

```
uv run rag ingest
```

Busca los fragmentos más parecidos a una pregunta:

```
uv run rag search "¿Cómo declaro un parámetro de query opcional?"
```

## Desarrollo

```
uv run pytest -q
uv run ruff check .
uv run mypy
```

## Corpus

La procedencia y la licencia de los documentos están en [data/PROCEDENCIA.md](data/PROCEDENCIA.md).
