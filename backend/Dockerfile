# ---------- builder: install Python deps into an isolated prefix ----------
FROM python:3.14-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Use --user so deps land in /root/.local and we can copy a clean tree to the
# runtime image without dragging the build layer along.
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# ---------- runtime ----------
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/root/.local/bin:$PATH \
    ENVIRONMENT=production

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
