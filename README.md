# RAG con evaluación

Asistente de preguntas y respuestas sobre la documentación de FastAPI en español, con búsqueda híbrida, citas obligatorias y una evaluación medible que compara configuraciones.

Estado: en construcción (semana 2 de 4). Ya funcionan la ingesta, la búsqueda vectorial y las respuestas con citas por línea de comandos.

## Preparar el entorno

Requiere [uv](https://docs.astral.sh/uv/).

```
uv sync
```

Copia `.env.example` como `.env` y completa:

- `OPENAI_API_KEY`: tu clave de OpenAI, para los embeddings.
- `LLM_PROVIDER` y `LLM_MODEL`: el proveedor del modelo que redacta las respuestas (`openai` o `deepseek`) y el nombre exacto del modelo según su documentación. Con DeepSeek, añade `DEEPSEEK_API_KEY`.

## Uso

Construye el índice vectorial a partir del corpus:

```
uv run rag ingest
```

Busca los fragmentos más parecidos a una pregunta:

```
uv run rag search "¿Cómo declaro un parámetro de query opcional?"
```

Haz una pregunta y recibe una respuesta con citas a la documentación:

```
uv run rag ask "¿Cómo declaro un parámetro de query opcional?"
```

## Desarrollo

```
uv run pytest -q
uv run ruff check .
uv run mypy
```

## Corpus

La procedencia y la licencia de los documentos están en [data/PROCEDENCIA.md](data/PROCEDENCIA.md).
