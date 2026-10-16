"""Passo 1: limpa o dataset, gera chunks, extrai features (com spaCy) e o léxico de formas flexionadas.

Uso: python scripts/prepare_data.py /caminho/dataset_caed
Saídas: data/processed/chunks.csv, data/processed/features.csv, data/processed/lexicon.json
"""
import json, os, sys, time
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import spacy
from caed.data import build_dataset
from caed.features import extract

root = sys.argv[1] if len(sys.argv) > 1 else "dataset_caed"
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed")
os.makedirs(out, exist_ok=True)

df = build_dataset(root)
df.to_csv(os.path.join(out, "chunks.csv"), index=False)
print(f"{len(df)} chunks de {df.file.nunique()} arquivos", flush=True)

nlp = spacy.load("pt_core_news_sm", disable=["ner"])
rows, forms, freq = [], defaultdict(Counter), Counter()
t0 = time.time()
for i, doc in enumerate(nlp.pipe(df.text.tolist(), batch_size=64, n_process=2)):
    rows.append(extract(df.text.iloc[i], doc))
    for t in doc:
        if t.is_alpha:
            freq[t.text.lower()] += 1
            m = t.morph.to_dict()
            key = "|".join([t.lemma_.lower(), t.pos_] + [f"{k}={m[k]}" for k in sorted(m) if k in ("Gender", "Number", "Tense", "Mood", "Person", "VerbForm")])
            forms[key][t.text.lower()] += 1
    if i % 1000 == 0:
        print(i, f"{time.time()-t0:.0f}s", flush=True)
feat = pd.DataFrame(rows)
feat.insert(0, "chunk_id", df.chunk_id.values)
feat.to_csv(os.path.join(out, "features.csv"), index=False)
json.dump({"forms": {k: c.most_common(1)[0][0] for k, c in forms.items()},
           "freq": {w: c for w, c in freq.items() if c >= 2}},
          open(os.path.join(out, "lexicon.json"), "w"), ensure_ascii=False)
print("ok", feat.shape, f"{time.time()-t0:.0f}s")
