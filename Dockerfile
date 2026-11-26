# Imagen de la API. La misma versión de Python que .python-version y la misma de uv que en
# desarrollo, con número exacto: una etiqueta suelta ("3-slim", "latest") cambia sin aviso.
FROM python:3.14.3-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /bin/uv

# UV_COMPILE_BYTECODE: compila los .pyc al construir, así el primer arranque es más rápido.
# UV_LINK_MODE=copy: copia en vez de enlazar (el caché de uv no vive en la imagen final).
# UV_PYTHON_DOWNLOADS=never: usa el Python de la imagen, nunca descarga otro.
# TIKTOKEN_CACHE_DIR: el tokenizador se descarga al construir y queda dentro de la imagen.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Primero solo las dependencias: Docker reutiliza esta capa mientras uv.lock no cambie,
# y cambiar el código no obliga a reinstalar chromadb y compañía.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# El corpus y el golden set viajan en la imagen; el índice no: se arma al arrancar y vive en
# un volumen (ver docker-compose.yml), así no se reindexa en cada reinicio.
COPY data/raw ./data/raw
COPY data/eval ./data/eval
COPY data/PROCEDENCIA.md ./data/

# Sin privilegios: si alguien explota la API, no es root dentro del contenedor
RUN useradd --create-home --uid 10001 app \
    && mkdir -p data/index results \
    && chown app:app data/index results
USER app

EXPOSE 8000

# Ingiere solo si el índice está vacío (la primera vez) y después levanta la API.
# exec: uvicorn reemplaza al shell y recibe las señales de docker stop.
CMD ["sh", "-c", "rag ingest --if-empty && exec uvicorn rag.interfaces.api:app --host 0.0.0.0 --port 8000"]
