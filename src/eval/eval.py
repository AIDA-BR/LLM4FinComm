"""
Atualiza e ANEXA métricas (BLEU, ROUGE, BERTScore, CTC) no JSON de entrada.
Para cada item, calcula as métricas para TODAS as colunas 'generated_text*'
e salva de volta um campo 'metrics' por coluna avaliada, preservando o resto.

Opcional: filtrar por modelos em TARGET_MODELS (ONLY_TARGET_MODELS=True).
Também marca 'best_output' por item com base no maior BERTScore_F1.
"""

from typing import Dict, Any, List
import json
import torch
from evaluate import load

# ctc_score conforme seu ambiente
from ctc_score import FactualConsistencyScorer, DialogScorer

# ------------------------------------------------------------------
# CONFIGURAÇÕES
# ------------------------------------------------------------------
INPUT_JSON   = "translated_new_gens_0601_nometrics.json"
OUTPUT_JSON  = "translated_new_gens_0601_with_metrics.json"

# Se True, só calcula para os modelos listados abaixo.
ONLY_TARGET_MODELS = True
TARGET_MODELS = [
    "mistral-7b",
    "llama3-8b",
    "gemma-3-12b",
    "gpt4o",
    "sabia3",
    "mistral-3-24b",
    "gemma-3-27b",
    "llama-4-scout",
]

# Colunas de referência/grounding
REFERENCE_COL = ""
FACT_COL      = ""

# Prefixos de colunas de saídas geradas (varia conforme seu dataset)
GENERATED_PREFIXES = ["generated_text"]

# Modelo para BERTScore via evaluate (multilíngue para inglês/pt)
BERTSCORE_MODEL_TYPE = "bert-base-multilingual-cased"

# ------------------------------------------------------------------
# CARREGA MÉTRICAS
# ------------------------------------------------------------------
bleu       = load("bleu")
rouge      = load("rouge")
bertscore  = load("bertscore")
device     = "cuda" if torch.cuda.is_available() else "cpu"

factual_scorer = FactualConsistencyScorer(align="E-bert", device=device)
dialog_scorer  = DialogScorer(align="E-bert", device=device)

# ------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# ------------------------------------------------------------------
def is_non_empty_text(x: Any) -> bool:
    return isinstance(x, str) and x.strip() != ""

def find_generated_cols(example: Dict[str, Any]) -> List[str]:
    cols = []
    for k in example.keys():
        for pref in GENERATED_PREFIXES:
            if k.startswith(pref):
                cols.append(k)
                break
    return sorted(cols)

def compute_all_metrics(pred: str, reference: str, fact: str) -> Dict[str, Any]:
    # As libs esperam listas
    predictions = [pred]
    references  = [reference]

    # BLEU
    bleu_val = bleu.compute(predictions=predictions, references=references)["bleu"]

    # ROUGE
    rouge_vals = rouge.compute(predictions=predictions, references=references)
    # garante chaves esperadas
    r1 = float(rouge_vals.get("rouge1", 0.0))
    r2 = float(rouge_vals.get("rouge2", 0.0))
    rL = float(rouge_vals.get("rougeL", 0.0))
    rS = float(rouge_vals.get("rougeLsum", 0.0))

    # BERTScore (via evaluate)
    bs = bertscore.compute(
        predictions=predictions,
        references=references,
        model_type=BERTSCORE_MODEL_TYPE,
        lang=None,  # deixe None para auto (usa model_type)
        rescale_with_baseline=False,
    )
    bsP = float(bs["precision"][0])
    bsR = float(bs["recall"][0])
    bsF = float(bs["f1"][0])

    # CTC — robustez com fallback se 'fact' vazio
    # groundedness (com base no 'fact' e no 'reference')
    try:
        grounded_fact = float(dialog_scorer.score(fact=fact or "", dialog_history=[], hypo=pred, aspect="groundedness"))
    except Exception:
        grounded_fact = None

    try:
        grounded_ref  = float(dialog_scorer.score(fact=reference or "", dialog_history=[], hypo=pred, aspect="groundedness"))
    except Exception:
        grounded_ref = None

    # factual consistency (com base no 'fact' e também no 'reference' como comparação)
    try:
        factual_fact = float(factual_scorer.score(grounding=fact or "", hypo=pred))
    except Exception:
        factual_fact = None

    try:
        factual_ref  = float(factual_scorer.score(grounding=reference or "", hypo=pred))
    except Exception:
        factual_ref = None

    return {
        "BLEU_score": float(bleu_val),
        "ROUGE": {
            "rouge1": r1, "rouge2": r2, "rougeL": rL, "rougeLsum": rS
        },
        "BERTScore": {
            "precision": bsP, "recall": bsR, "f1": bsF
        },
        "CTC": {
            "groundedness_fact": grounded_fact,
            "groundedness_ref": grounded_ref,
            "factual_fact": factual_fact,
            "factual_ref": factual_ref
        }
    }

# ------------------------------------------------------------------
# PROCESSAMENTO
# ------------------------------------------------------------------
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    data: List[Dict[str, Any]] = json.load(f)

if not isinstance(data, list) or not data:
    raise ValueError("O JSON deve ser uma lista não vazia de objetos.")

generated_cols = find_generated_cols(data[0])
if not generated_cols:
    raise ValueError("Não foram encontradas colunas 'generated_text*' no JSON.")

updated = 0
for i, item in enumerate(data):
    model_name = item.get("generator_model", None)
    if ONLY_TARGET_MODELS and (model_name not in TARGET_MODELS):
        continue

    reference = item.get(REFERENCE_COL, "")
    fact      = item.get(FACT_COL, "")

    if not is_non_empty_text(reference):
        # Sem referência não dá para computar as métricas baseadas em ref
        # (mantemos o item intacto)
        continue

    # Garante campo para anexar as métricas
    if "metrics" not in item or not isinstance(item["metrics"], dict):
        item["metrics"] = {}

    best_f1 = -1.0
    best_col = None

    for col in generated_cols:
        pred = item.get(col, None)
        if not is_non_empty_text(pred):
            continue

        try:
            m = compute_all_metrics(pred=pred, reference=reference, fact=fact)
        except Exception as e:
            # Se alguma métrica falhar, seguimos sem travar o pipeline
            m = {"error": f"Falha ao calcular métricas: {type(e).__name__}: {e}"}

        item["metrics"][col] = m

        # Atualiza "melhor" por BERTScore F1 se disponível
        f1 = m.get("BERTScore", {}).get("f1", None) if isinstance(m, dict) else None
        if isinstance(f1, (float, int)) and f1 > best_f1:
            best_f1 = float(f1)
            best_col = col

    # marca o melhor output do item
    if best_col is not None:
        item["best_output"] = {
            "column": best_col,
            "criterion": "BERTScore_F1",
            "value": best_f1
        }

    updated += 1

print(f"Itens processados (com filtro aplicado={ONLY_TARGET_MODELS}): {updated} de {len(data)}")

# ------------------------------------------------------------------
# SALVA JSON ENRIQUECIDO
# ------------------------------------------------------------------
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Arquivo salvo com métricas em: {OUTPUT_JSON}")
