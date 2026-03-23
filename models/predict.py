from pydantic import BaseModel, Field


class PredictInput(BaseModel):
    """
    Features esperadas pelo modelo de risco de cancelamento.
    A ordem dos campos aqui deve ser idêntica à ordem usada
    para montar o array numpy em routers/predict.py.
    """

    valor_pedido: float = Field(
        gt=0,
        description="Valor total do pedido em reais",
        examples=[85.0],
    )
    hora_pedido: int = Field(
        ge=0,
        le=23,
        description="Hora do dia em que o pedido foi feito (0-23)",
        examples=[20],
    )
    num_itens: int = Field(
        ge=1,
        description="Quantidade de pratos no pedido",
        examples=[3],
    )
    historico_cancelamentos: int = Field(
        ge=0,
        description="Número de cancelamentos anteriores do cliente",
        examples=[0],
    )
    distancia_entrega: float = Field(
        ge=0,
        description="Distância até o endereço de entrega em km",
        examples=[2.5],
    )


class PredictOutput(BaseModel):
    """
    Resposta do endpoint de predição.
    """

    prediction: int = Field(
        description="Classe predita: 0 = pedido normal, 1 = cancelamento provável",
    )
    probability: float = Field(
        description="Probabilidade de cancelamento (classe 1), entre 0 e 1",
    )
    label: str = Field(
        description="Rótulo legível da predição",
    )
    model_version: str = Field(
        description="Versão do modelo usado. 'mock' enquanto o modelo real não está disponível.",
    )
