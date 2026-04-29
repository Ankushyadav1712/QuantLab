from fastapi import FastAPI

app = FastAPI(title="QuantLab")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
