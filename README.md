# RAG con evaluación

[![CI](https://github.com/CamiloSTDev/rag-eval-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/CamiloSTDev/rag-eval-pipeline/actions/workflows/ci.yml)

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

Compara las dos ramas de la búsqueda con `--mode vector`, `--mode bm25` (por palabras) o `--mode hybrid` (las dos fusionadas con Reciprocal Rank Fusion). Para que `ask` y `eval` usen la búsqueda híbrida, pon `USE_BM25=true` en `.env`.

Haz una pregunta y recibe una respuesta con citas a la documentación:

```
uv run rag ask "¿Cómo declaro un parámetro de query opcional?"
```

## API

```
uv run uvicorn rag.interfaces.api:app
```

| Endpoint | Qué hace |
| --- | --- |
| `POST /ask` | Recibe `{"question": "..."}` (1 a 1.000 caracteres) y devuelve la respuesta, sus citas, las citas sin fuente, si rechazó, la similitud vectorial, la latencia por etapa y los tokens |
| `POST /ingest` | Reingesta el corpus configurado. Exige el header `X-Admin-Key` con el valor de `ADMIN_API_KEY`; sin esa variable, el endpoint está desactivado |
| `GET /health` | `200` con la cantidad de chunks indexados, o `503` si el índice está vacío |
| `GET /version` | Versión del paquete, configuración activa y un hash corto de ella |

Errores: `422` si la pregunta no es válida, `503` si el índice está vacío y `502` si falla el proveedor del modelo o devuelve una respuesta ilegible. La documentación interactiva queda en `/docs`.

## Evaluación

El golden set va en `data/eval/golden.jsonl`, una pregunta por línea en JSON. Hay un ejemplo en [data/eval/golden.example.jsonl](data/eval/golden.example.jsonl). Cada línea lleva:

- `id`, `question` y `reference_answer` (la respuesta verificada a mano).
- `relevant_sources`: dónde está la respuesta, como lista de `{"source": ..., "section": ..., "evidence": ...}`. `source` es la ruta dentro de `data/raw`; `section`, la ruta exacta de títulos (opcional, igual que `page` para PDF); `evidence`, una cita literal que respalda la respuesta (opcional). Si varias secciones responden la pregunta, encontrar cualquiera cuenta como acierto.
- `type`: `single` (un chunk), `multi` (varios) o `unanswerable` (sin respuesta en el corpus; no lleva `relevant_sources`).
- `holdout`: `true` en las preguntas que se reservan para la evaluación final.
- `verified`: `true` cuando una persona revisó la pregunta, la respuesta y sus fuentes.

Valida el golden set contra el corpus limpio, sin necesitar claves. Comprueba que cada fuente, sección y cita exista tal cual; `--show` lo imprime en formato legible para revisarlo:

```
uv run rag golden --show
```

Corre una evaluación con un nombre de experimento:

```
uv run rag eval E0
```

Guarda `results/E0.csv`, con una fila por pregunta, y `results/E0.json`, con la configuración usada y el resumen: Hit@5, Hit@10, MRR, aciertos de rechazo, citas sin fuente, latencia p50/p95 y tokens. Con `--holdout` evalúa solo las preguntas reservadas.

Con `JUDGE_PROVIDER` y `JUDGE_MODEL` configurados, un segundo modelo actúa como juez y mide la **fidelidad**: qué fracción de las afirmaciones de cada respuesta respaldan sus fuentes. También mide la **relevancia**: si la respuesta contesta la pregunta. El juez debe ser un modelo distinto del que genera. Las afirmaciones sin respaldo quedan en el CSV, para revisar a mano unas 10 respuestas y comprobar que coincides con el juez.

## Desarrollo

```
uv run pytest -q
uv run ruff check .
uv run mypy
```

## Corpus

La procedencia y la licencia de los documentos están en [data/PROCEDENCIA.md](data/PROCEDENCIA.md).
