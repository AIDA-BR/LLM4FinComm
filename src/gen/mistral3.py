import pandas as pd
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
import gc

# Configurar modelo Ollama via LlamaIndex
model_name = "mistral-small3.1:24b"  
llm = Ollama(model=model_name, request_timeout=120.0, temperature=0.3)

system_msg = 'Você é um analista financeiro com formação em economia que escreve para um público de investidores geral'

def get_user_prompt(company, fr):
    return f'''### Instrução:
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

### Resposta: '''

# Carregar o CSV
df = pd.read_csv("")

gpt4o_rows = df[df['generator_model'] == 'gpt4o'].iloc[:-2].copy()

# Atualizar a coluna generator_model
gpt4o_rows['generator_model'] = "mistral-3-24b"

# Resetar métricas
metric_cols = df.columns.tolist()
metric_start_idx = metric_cols.index('BLEU_score')
metrics_to_reset = metric_cols[metric_start_idx:]
gpt4o_rows[metrics_to_reset] = 0

# Lista para armazenar os textos gerados
generated_texts = []

# Iterar sobre as linhas para gerar o texto
for idx, row in gpt4o_rows.iterrows():
    empresa = row['company']
    fr_conteudo = row['material fact']

    instruct = get_user_prompt(empresa, fr_conteudo)

    # Construir as mensagens para o Ollama (LlamaIndex)
    messages = [
        ChatMessage(role="system", content=system_msg),
        ChatMessage(role="user", content=instruct),
    ]

    # Gerar resposta com Ollama
    try:
        response = llm.chat(messages)
        generated_text = response.message.content.strip()
    except Exception as e:
        generated_text = f"Error during generation: {e}"

    generated_texts.append(generated_text)

    gc.collect()
    print("Done! ", idx)

# Atualizar o dataframe com os textos gerados
gpt4o_rows['generated_text'] = generated_texts

# Concatenar ao dataframe original
df = pd.concat([df, gpt4o_rows], ignore_index=True)

# Salvar o CSV atualizado
df.to_csv("basis_updated.csv", index=False)
