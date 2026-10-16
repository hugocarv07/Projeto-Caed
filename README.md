# CAEd · Complexidade textual e geração de textos por mutação

Aplicação web que (1) classifica um texto curto em português em 4 níveis de complexidade
(Ensino Fundamental I, Fundamental II, Médio, Superior) e (2) gera novas versões do texto por
mutações (troca de palavras, troca de gênero, paráfrase), mostrando quanto a complexidade prevista mudou.
Metodologia e resultados: **RELATORIO.md**.

## Como executar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download pt_core_news_sm
python app.py
```

Os modelos treinados (`models/`), os dados processados (`data/processed/`) e o WordNet em português
(`data/nltk_data/`) já acompanham o projeto: **não é preciso retreinar** para usar o app.

## Reproduzir todo o pipeline (a partir do dataset original)

```bash
python scripts/prepare_data.py /caminho/dataset_caed   # limpeza, chunks, features (spaCy), léxico de formas   (~1 min)
python scripts/build_vectors.py                        # vetores PPMI-SVD usados no filtro de sinônimos       (~1 min)
python scripts/train.py                                # compara 10 modelos, salva models/*.joblib           (~10 min)
python scripts/evaluate_mutations.py 60                # efeito das mutações na complexidade                  (~4 min)
python scripts/make_report_assets.py                   # figuras e importância de features
python -m pytest -q tests                              # testes (canonicidade e mutações)
```

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `caed/syllables.py` | **Silabificação e canonicidade** de palavras (feature exigida no enunciado) |
| `caed/data.py` | Limpeza do dataset e criação das instâncias (chunks) |
| `caed/features.py` | Features superficiais, silábicas/canonicidade e sintáticas (spaCy) |
| `caed/model.py` | Modelos comparados e wrapper de inferência (probabilidades + escore contínuo 0–3) |
| `caed/mutations.py`, `morph.py`, `lexicon.py` | Operadores de mutação, regras morfológicas, léxico (OpenWN-PT, vetores) |
| `app.py`, `templates/index.html` | Aplicação web (Flask) |
| `scripts/` | Pipeline de dados, treino e avaliação |
| `results/` | Métricas (json), figuras e exemplos de mutações |

## Uso rápido da API

```bash
curl -s -X POST localhost:5000/api/mutate -H 'Content-Type: application/json' \
  -d '{"text":"O texto foi lido pelo aluno.","ops":["paraphrase"],"rate":1,"seed":1}'
```
