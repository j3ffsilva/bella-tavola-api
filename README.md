# bella-tavola-api

API do restaurante Bella Tavola com endpoint de predição de risco de cancelamento de pedidos.

Construída como referência para o curso **MLOps — CDIA CD2 2026**.

---

## Contexto do curso

Este repositório cobre o **encontro 02** — integração de um endpoint de ML em uma API FastAPI.

| Encontro | O que acontece |
|---|---|
| **e02 (este repo)** | Endpoint `/ml/predict` existe, mas retorna predição mockada |
| **e03** | Modelo treinado e publicado no Hugging Face Hub |
| **e03 → e02** | Mock substituído pela chamada real ao Hub |

---

## Estrutura do projeto

```
bella-tavola-api/
├── main.py              # Inicialização da API e registro dos routers
├── model_utils.py       # Carregamento do modelo (mock → Hub no e03)
├── routers/
│   └── predict.py       # Endpoint POST /ml/predict e GET /ml/health
├── models/
│   └── predict.py       # PredictInput e PredictOutput (schemas Pydantic)
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Como rodar localmente

**1. Clone o repositório e entre na pasta:**

```bash
git clone https://github.com/seu-usuario/bella-tavola-api.git
cd bella-tavola-api
```

**2. Crie e ative um ambiente virtual:**

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\activate
```

**3. Instale as dependências:**

```bash
pip install -r requirements.txt
```

**4. Configure as variáveis de ambiente:**

```bash
cp .env.example .env
# Edite .env com seus valores (necessário apenas a partir do e03)
```

**5. Suba a API:**

```bash
uvicorn main:app --reload
```

**6. Acesse a documentação:**

- Swagger UI: http://localhost:8000/docs
- Health check: http://localhost:8000/ml/health

---

## Endpoints

### `POST /ml/predict`

Recebe as features do pedido e retorna a predição de risco de cancelamento.

**Request body:**

```json
{
  "valor_pedido": 85.0,
  "hora_pedido": 20,
  "num_itens": 3,
  "historico_cancelamentos": 0,
  "distancia_entrega": 2.5
}
```

**Response:**

```json
{
  "prediction": 0,
  "probability": 0.12,
  "label": "pedido normal",
  "model_version": "mock"
}
```

> **Nota:** enquanto o modelo não estiver disponível (pré e03), o endpoint retorna sempre `prediction: 0` e `model_version: "mock"`. Isso é intencional — o contrato da API já está definido e testável.

### `GET /ml/health`

Verifica o status da API e do modelo.

```json
{
  "api": "ok",
  "model": "mock",
  "model_info": "Modelo não disponível — usando predições mockadas (pré e03)"
}
```

Após o e03, `model` passará a retornar `"ok"` quando o modelo estiver carregado corretamente.

---

## Integrando o modelo real (após o e03)

Após publicar o modelo no Hugging Face Hub no encontro 03:

**1. Configure as variáveis de ambiente em `.env`:**

```
HF_TOKEN=hf_seu_token_aqui
HF_REPO_ID=seu-usuario/bella-tavola-model
```

**2. Descomente as dependências em `requirements.txt`:**

```
huggingface-hub==0.29.3
joblib==1.4.2
```

**3. Em `model_utils.py`, descomente o bloco de implementação real** e remova o bloco mock.

**4. Em `routers/predict.py`, descomente o bloco real** marcado com `# TODO (e03)` e remova o bloco mock.

**5. Reinstale as dependências e suba a API novamente.**

---

## Rodando os testes

```bash
pytest tests/ -v
```

> A pasta `tests/` com os testes de integração será adicionada durante os cadernos da Semana 3 (Partes 1, 2 e 3).
