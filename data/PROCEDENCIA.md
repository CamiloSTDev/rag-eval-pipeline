# Procedencia de los datos

Todo lo que está en `data/raw/` viene de fuentes públicas. Este archivo registra de dónde salió cada carpeta, en qué versión y con qué licencia, y cómo se construyó y revisó el golden set de `data/eval/`.

## `raw/fastapi-es/`

| Dato | Valor |
| --- | --- |
| Contenido | Documentación de FastAPI en español |
| Origen | https://github.com/fastapi/fastapi, carpeta `docs/es/docs` |
| Versión | commit `f5c6e9b4f9cadc1bf0f9a1fd672e621206b63007` (rama `master`, 8 de octubre de 2026) |
| Descargado | 13 de octubre de 2026 |
| Tamaño | 125 archivos Markdown, 803.184 bytes |
| Licencia | MIT, Copyright (c) 2018 Sebastián Ramírez. Copia en `raw/fastapi-es/LICENSE` |
| Modificaciones | Ninguna: los archivos son copia exacta del repositorio |

Notas sobre el contenido:

- Según el sitio de FastAPI, la traducción al español está hecha con IA y guiada por humanos.
- Los ejemplos de código no están dentro del Markdown. Cada uno aparece como una directiva `{* ../../docs_src/... *}` que apunta a un archivo de la carpeta `docs_src` del repositorio original, que no se descargó.

## `eval/golden.jsonl`

Las preguntas de referencia contra las que se mide el sistema. No vienen de una fuente externa: se redactaron para este proyecto a partir del corpus.

| Paso | Quién | Qué se hizo |
| --- | --- | --- |
| Redacción (22 nov 2026) | Claude | 46 preguntas (28 de un chunk, 10 de varios, 8 sin respuesta; 10 de hold-out), cada respuesta con una cita literal de la sección que la respalda |
| Validación mecánica | `rag golden` | Cada fuente, sección y cita existe tal cual en el corpus limpio |
| Revisión (25 nov 2026) | Dos agentes independientes (Claude Opus y Claude Sonnet), por delegación del dueño del proyecto | Cada pregunta revisada por los dos por separado, sin acceso a los resultados del sistema. Leyeron las secciones completas y buscaron en todo el corpus otras secciones que también respondieran |
| Consolidación | Claude | Se agregaron 54 fuentes que también responden, se completaron 9 respuestas de referencia y se reemplazó una pregunta sin respuesta (Azure) porque el corpus la respondía en parte |

`verified: true` significa que la pregunta pasó esa revisión, no que la revisó una persona. Los revisores son de la misma familia de modelos que la autora, así que sus errores no son del todo independientes: una revisión humana sigue siendo recomendable.
