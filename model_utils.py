"""
Utilitários para carregamento do modelo de ML.

Estado atual: load_model() retorna None — o modelo ainda não existe.
O endpoint /ml/predict usa dados mockados enquanto isso.

Quando o modelo estiver publicado no Hugging Face Hub (encontro 03),
substitua o corpo de load_model() pela implementação real abaixo.
"""

import os
import logging

logger = logging.getLogger(__name__)

# Repo ID do modelo no Hugging Face Hub.
# Configurado via variável de ambiente HF_REPO_ID.
# Exemplo: "seu-usuario/bella-tavola-model"
HF_REPO_ID = os.environ.get("HF_REPO_ID", "")


def load_model(force_download: bool = False):
    """
    Carrega o modelo do Hugging Face Hub.

    Retorna o modelo treinado, ou None se o modelo ainda não estiver
    disponível (durante o desenvolvimento, antes do encontro 03).

    Args:
        force_download: Se True, ignora o cache local e baixa novamente.

    Returns:
        Modelo sklearn carregado, ou None.

    # ------------------------------------------------------------------
    # IMPLEMENTAÇÃO REAL — descomente após o encontro 03
    # ------------------------------------------------------------------
    # import joblib
    # from huggingface_hub import hf_hub_download, login
    #
    # token = os.environ.get("HF_TOKEN")
    # if token:
    #     login(token=token)
    #
    # if not HF_REPO_ID:
    #     raise ValueError(
    #         "HF_REPO_ID não configurado. "
    #         "Defina a variável de ambiente HF_REPO_ID com o repo do modelo."
    #     )
    #
    # local_path = hf_hub_download(
    #     repo_id=HF_REPO_ID,
    #     filename="model.pkl",
    #     force_download=force_download,
    # )
    # model = joblib.load(local_path)
    # logger.info("Modelo carregado de: %s", local_path)
    # return model
    # ------------------------------------------------------------------
    """
    logger.warning(
        "load_model() chamado, mas o modelo ainda não está disponível. "
        "Retornando None — o endpoint usará dados mockados. "
        "Implemente a carga real após o encontro 03."
    )
    return None


# Modelo carregado em memória — None até o encontro 03.
# Lazy loading: carregado na primeira requisição via get_model().
_model = None


def get_model():
    """
    Retorna o modelo em memória, carregando na primeira chamada.

    O padrão lazy loading evita baixar o modelo a cada requisição
    e evita falha na inicialização da API quando o modelo não existe.
    """
    global _model
    if _model is None:
        _model = load_model()
    return _model
