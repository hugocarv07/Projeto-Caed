"""Passo 3: quanto cada mutação altera a complexidade prevista?

Os modelos aqui são treinados SÓ nos blocos 0-3; as mutações são aplicadas a textos do bloco 4 (nunca vistos).
Para cada operador (e combinações) mede-se a variação do escore contínuo E[nível] = Σ p_k·k  (0..3) e a
taxa de troca de classe, com dois modelos: TF-IDF de caracteres (melhor na validação) e HistGB sobre features.
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from sklearn.base import clone
from caed.model import make_models, numeric_cols, ComplexityClassifier
from caed.mutations import Mutator

N_PER_LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 40
P = os.path.join(ROOT, "data", "processed")
df = pd.read_csv(os.path.join(P, "chunks.csv")).drop(columns=["n_words"]).merge(pd.read_csv(os.path.join(P, "features.csv")), on="chunk_id")
cols = numeric_cols(pd.read_csv(os.path.join(P, "features.csv")))
models = make_models(cols)
dev, test = df[df.block < 4], df[df.block == 4]
import spacy
lite = spacy.load("pt_core_news_sm", disable=["ner"])
mut = Mutator()
clfs = {}
for key, name in [("TF-IDF chars", "TF-IDF chars · LR"), ("HistGB features", "HistGB · features")]:
    t0 = time.time()
    clfs[key] = ComplexityClassifier(clone(models[name]).fit(dev, dev.level.values), cols, nlp=lite)
    print("treinado", key, f"{time.time()-t0:.0f}s", flush=True)

sample = pd.concat([g.sample(min(N_PER_LEVEL, len(g)), random_state=0) for _, g in test.groupby("level")]).reset_index(drop=True)
texts = sample.text.tolist()
base = {k: c.predict_proba(texts) for k, c in clfs.items()}
print("amostra:", len(texts), "textos", flush=True)

configs = {
    "troca de palavras (sinônimos+antônimos)": dict(ops=("word",), word_mode="mixed", direction="random"),
    "troca de palavras → só sinônimos": dict(ops=("word",), word_mode="synonym", direction="random"),
    "troca de palavras → mais simples": dict(ops=("word",), word_mode="synonym", direction="simpler"),
    "troca de palavras → mais difícil": dict(ops=("word",), word_mode="synonym", direction="harder"),
    "troca de gênero": dict(ops=("gender",)),
    "paráfrase": dict(ops=("paraphrase",)),
    "as três (aleatório)": dict(ops=("paraphrase", "gender", "word")),
}
RATE = 0.3
out, examples, cnt = {}, [], {}
for name, cfg in configs.items():
    t0 = time.time()
    mutated, nchg = [], []
    for i, t in enumerate(texts):
        r = mut.mutate(t, rate=RATE, seed=1000 + i, **{k: v for k, v in cfg.items()})
        mutated.append(r.text); nchg.append(len(r.changes))
        if len(examples) < 400 and r.changes:
            examples.append(dict(config=name, level=int(sample.level[i]), before=t, after=r.text, changes=r.changes))
    changed = np.array([m != t for m, t in zip(mutated, texts)])
    nwords = sample.n_words.values
    res = dict(pct_texts_changed=float(changed.mean() * 100), changes_per_100_words=float(np.sum(nchg) / nwords.sum() * 100))
    for k, c in clfs.items():
        p2 = c.predict_proba(mutated)
        d = p2 @ np.arange(4) - base[k] @ np.arange(4)
        res[k] = dict(mean_delta=float(d.mean()), std_delta=float(d.std()), mean_abs_delta=float(np.abs(d).mean()),
                      mean_delta_changed_only=float(d[changed].mean()) if changed.any() else 0.0,
                      pct_class_changed=float((p2.argmax(1) != base[k].argmax(1)).mean() * 100),
                      pct_up=float((p2.argmax(1) > base[k].argmax(1)).mean() * 100),
                      pct_down=float((p2.argmax(1) < base[k].argmax(1)).mean() * 100),
                      by_level={LV: float(d[sample.level.values == l].mean()) for l, LV in enumerate(["EF-I", "EF-II", "EM", "ES"])})
    out[name] = res
    print(f"{name:42s} alterados={res['pct_texts_changed']:.0f}% " + " | ".join(
        f"{k}: Δ={res[k]['mean_delta']:+.3f} (|Δ|={res[k]['mean_abs_delta']:.3f}, troca classe {res[k]['pct_class_changed']:.0f}%)" for k in clfs)
        + f"  [{time.time()-t0:.0f}s]", flush=True)

# varredura de intensidade (as três mutações)
sweep = {}
for rate in (0.1, 0.3, 0.6, 1.0):
    mutated = [mut.mutate(t, ops=("paraphrase", "gender", "word"), rate=rate, seed=2000 + i).text for i, t in enumerate(texts)]
    sweep[str(rate)] = {k: float(np.mean(c.predict_proba(mutated) @ np.arange(4) - base[k] @ np.arange(4))) for k, c in clfs.items()}
    sweep[str(rate)]["pct_changed"] = float(np.mean([m != t for m, t in zip(mutated, texts)]) * 100)
    print("intensidade", rate, sweep[str(rate)], flush=True)

# escala de referência: diferença média de escore entre níveis consecutivos (para interpretar os Δ)
ref = {k: [float((base[k] @ np.arange(4))[sample.level.values == l].mean()) for l in range(4)] for k in clfs}
json.dump(dict(n=len(texts), rate=RATE, results=out, sweep=sweep, mean_score_by_level=ref), open(os.path.join(ROOT, "results", "mutation_results.json"), "w"), ensure_ascii=False, indent=1)
json.dump(examples, open(os.path.join(ROOT, "results", "mutation_examples.json"), "w"), ensure_ascii=False, indent=1)
print("escore médio por nível", ref)
