from fastapi import FastAPI
from routers import predict

app = FastAPI(
    title="Bella Tavola API",
    description=(
        "API do restaurante Bella Tavola com endpoint de predição de risco de cancelamento. "
        "Construída como referência para o curso de MLOps — CDIA CD2 2026."
    ),
    version="0.1.0",
)

app.include_router(
    predict.router,
    prefix="/ml",
    tags=["ML"],
)


@app.get("/", tags=["Geral"])
async def root():
    return {
        "restaurante": "Bella Tavola",
        "versao": "0.1.0",
        "docs": "/docs",
    }
