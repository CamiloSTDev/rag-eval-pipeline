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

## Evaluación

El golden set va en `data/eval/golden.jsonl`, una pregunta por línea en JSON. Hay un ejemplo en [data/eval/golden.example.jsonl](data/eval/golden.example.jsonl). Cada línea lleva:

- `id`, `question` y `reference_answer` (la respuesta verificada a mano).
- `relevant_sources`: dónde está la respuesta, como lista de `{"source": ..., "section": ...}`. `source` es la ruta dentro de `data/raw`; `section` es la ruta exacta de títulos y es opcional (`page` también, para PDF).
- `type`: `single` (un chunk), `multi` (varios) o `unanswerable` (sin respuesta en el corpus; no lleva `relevant_sources`).
- `holdout`: `true` en las preguntas que se reservan para la evaluación final.

Corre una evaluación con un nombre de experimento:

```
uv run rag eval E0
```

Guarda `results/E0.csv`, con una fila por pregunta, y `results/E0.json`, con la configuración usada y el resumen: Hit@5, Hit@10, MRR, aciertos de rechazo, citas sin fuente, latencia p50/p95 y tokens. Con `--holdout` evalúa solo las preguntas reservadas.

## Desarrollo

```
uv run pytest -q
uv run ruff check .
uv run mypy
```

## Corpus

La procedencia y la licencia de los documentos están en [data/PROCEDENCIA.md](data/PROCEDENCIA.md).
