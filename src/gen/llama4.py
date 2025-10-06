import os
import json
import subprocess
import time
import gc
import requests
import pandas as pd

PROJECT_ID   = ""
REGION       = "us-east5"
SERVICE_HOST = "us-east5-aiplatform.googleapis.com"
MODEL_NAME   = "meta/llama-4-scout-17b-16e-instruct-maas"
BASE_FILE    = "new_gens_0425_enriched.csv"
OUTPUT_FILE  = ""

new_generator_model = "llama-4-scout-enriched" 

API_URL = (
    f"https://{SERVICE_HOST}/v1beta1/"
    f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/openapi/chat/completions"
)

SYSTEM_MSG = (
    "Você é um analista financeiro com formação em economia que escreve para um público de investidores geral."
)

TEMPERATURE   = 0.3            
TOP_P         = 0.4
MAX_NEW_TOKENS = 2000       

def get_access_token() -> str:
    """Obtém token de acesso via gcloud CLI (ADC)"""
    return (
        subprocess.check_output(["gcloud", "auth", "print-access-token"])
        .decode()
        .strip()
    )

def vertex_chat_completion(prompt: str) -> str:
    """Envia o prompt para o modelo via Vertex AI e devolve apenas o texto gerado."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {get_access_token()}",
    }

    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "messages": [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user",   "content": prompt},
        ],
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    
    return data["choices"][0]["message"]["content"]

def get_user_prompt(company: str, fr: str) -> str:
    return f"""### Instrução:
Escreva em Português uma análise sobre a Empresa considerando o Fato Relevante e siga o Formato determinado. A análise deve conter TODOS os elementos especificados no seguinte Formato.

### Formato:
<Título>
<Sentença que Resuma a análise>
<Corpo da análise>
<Considerações Finais>

### Empresa:
{company}

### Fato Relevante:
{fr}

### Resposta: """


df = pd.read_csv(BASE_FILE)

gpt4o_rows = df[df["generator_model"] == "gpt4o"].iloc[:-2].copy()
gpt4o_rows["generator_model"] = new_generator_model

# zera colunas de métricas a partir de 'BLEU_score'
metric_cols = df.columns.tolist()
metrics_start = metric_cols.index("BLEU_score")
#gpt4o_rows[metric_cols[metrics_start:]] = 0

generated = []

for idx, row in gpt4o_rows.iterrows():
    
    company = row["company"]
    fr      = row["enriched_material_fact"]

    prompt = get_user_prompt(company, fr)
    try:
        text = vertex_chat_completion(prompt)
    except requests.HTTPError as e:
        print(f"[{idx}] Falha na chamada da API: {e}")
        text = ""  # opcional: marcar erro e continuar
    generated.append(text)
    print(f"[{idx}] concluído.")


    gc.collect()

gpt4o_rows["generated_text"] = generated

# concatena e grava
df_final = pd.concat([df, gpt4o_rows], ignore_index=True)
df_final.to_csv(OUTPUT_FILE, index=False)
print("Arquivo salvo!")
