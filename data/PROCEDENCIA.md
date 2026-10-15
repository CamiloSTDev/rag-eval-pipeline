# Procedencia del corpus

Todo lo que está en `data/raw/` viene de fuentes públicas. Este archivo registra de dónde salió cada carpeta, en qué versión y con qué licencia.

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
