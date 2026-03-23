"""
Router de ML — endpoint de predição de risco de cancelamento.

Estado atual: predição mockada (valores fixos).
O comentário "# TODO (e03)" marca onde o mock deve ser substituído
pelo modelo real após o encontro 03.
"""

import numpy as np
from fastapi import APIRouter
from model_utils import get_model
from models.predict import PredictInput, PredictOutput

router = APIRouter()


@router.post("/predict", response_model=PredictOutput)
async def predict(input: PredictInput):
    """
    Recebe as features do pedido e retorna a predição de risco de cancelamento.

    A ordem dos campos no array numpy abaixo deve ser idêntica
    à ordem das features usada no treinamento (gerar_dados.py do e03).
    """

    # Monta o array de features na mesma ordem do treinamento.
    # Se a ordem mudar aqui sem mudar no treino (ou vice-versa),
    # o modelo vai predizer com features embaralhadas — sem erro aparente.
    features = np.array([[
        input.valor_pedido,           # feature 0
        input.hora_pedido,            # feature 1
        input.num_itens,              # feature 2
        input.historico_cancelamentos,  # feature 3
        input.distancia_entrega,      # feature 4
    ]])

    model = get_model()

    # ------------------------------------------------------------------
    # TODO (e03): substituir o bloco mock pelo bloco real abaixo
    # ------------------------------------------------------------------
    # BLOCO REAL — descomente após publicar o modelo no Hub (e03)
    #
    # prediction = int(model.predict(features)[0])
    # probability = float(model.predict_proba(features)[0][1])
    # label = "cancelamento provável" if prediction == 1 else "pedido normal"
    # model_version = os.environ.get("HF_REPO_ID", "desconhecido")
    #
    # return PredictOutput(
    #     prediction=prediction,
    #     probability=round(probability, 4),
    #     label=label,
    #     model_version=model_version,
    # )
    # ------------------------------------------------------------------

    # BLOCO MOCK — remove após o e03
    # Retorna sempre "pedido normal" com probabilidade baixa.
    # Útil para testar o contrato da API antes do modelo existir.
    _ = features  # evita warning de variável não usada
    _ = model

    return PredictOutput(
        prediction=0,
        probability=0.12,
        label="pedido normal",
        model_version="mock",
    )


@router.get("/health")
async def health():
    """
    Verifica o status da API e do modelo.

    Distingue dois estados:
    - model: "ok"   → modelo carregado e funcional
    - model: "mock" → endpoint ativo mas usando dados mockados (pré e03)
    """
    model = get_model()

    if model is None:
        model_status = "mock"
        model_info = "Modelo não disponível — usando predições mockadas (pré e03)"
    else:
        # Smoke test: tenta uma predição com input neutro
        try:
            test_input = np.zeros((1, 5))
            model.predict(test_input)
            model_status = "ok"
            model_info = "Modelo carregado e funcional"
        except Exception as e:
            model_status = "degraded"
            model_info = str(e)

    return {
        "api": "ok",
        "model": model_status,
        "model_info": model_info,
    }
