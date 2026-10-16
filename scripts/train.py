"""Passo 2: treina e compara modelos. Protocolo:

 * dev = blocos 0-3, teste final = bloco 4 (nunca usado para escolher modelo);
 * validação cruzada de 4 dobras em dev, cada dobra = um bloco de arquivos contíguos (≈ material didático);
 * para quantificar o vazamento, também rodamos uma divisão aleatória de chunks (ERRADA de propósito).
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from caed.model import make_models, numeric_cols, ComplexityClassifier

P = os.path.join(ROOT, "data", "processed")
chunks = pd.read_csv(os.path.join(P, "chunks.csv"))
feat = pd.read_csv(os.path.join(P, "features.csv"))
df = chunks.drop(columns=["n_words"]).merge(feat, on="chunk_id")
cols = [c for c in numeric_cols(feat)]
print(len(df), "instâncias,", len(cols), "features numéricas", flush=True)
y = df.level.values
models = make_models(cols)


def metrics(yt, yp):
    return dict(acc=accuracy_score(yt, yp), f1=f1_score(yt, yp, average="macro"),
                mae=float(np.mean(np.abs(yt - yp))), adj=float(np.mean(np.abs(yt - yp) <= 1)))


def run(name, tr, te):
    m = clone(models[name]).fit(df.iloc[tr], y[tr])
    return m, m.predict(df.iloc[te])


res = {}
dev = np.where(df.block < 4)[0]
test = np.where(df.block == 4)[0]
for name in models:
    t0 = time.time()
    folds = []
    for b in range(4):
        tr, te = np.where((df.block < 4) & (df.block != b))[0], np.where(df.block == b)[0]
        _, yp = run(name, tr, te)
        folds.append(metrics(y[te], yp))
    cv = {k: float(np.mean([f[k] for f in folds])) for k in folds[0]}
    cv_std = float(np.std([f["f1"] for f in folds]))
    m, yp = run(name, dev, test)
    res[name] = dict(cv=cv, cv_f1_std=cv_std, test=metrics(y[test], yp),
                     test_confusion=confusion_matrix(y[test], yp).tolist())
    print(f"{name:45s} CV F1={cv['f1']:.3f}±{cv_std:.3f} acc={cv['acc']:.3f} | TESTE F1={res[name]['test']['f1']:.3f} "
          f"acc={res[name]['test']['acc']:.3f} MAE={res[name]['test']['mae']:.3f} ({time.time()-t0:.0f}s)", flush=True)

# vazamento: divisão aleatória por chunk (mesmo arquivo/material nos dois lados)
leak = {}
tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=0, stratify=y)
for name in ["LR · features", "HistGB · features", "TF-IDF palavras · LR", "TF-IDF chars · LR", "Híbrido: features + TF-IDF palavras · LR"]:
    _, yp = run(name, tr, te)
    leak[name] = metrics(y[te], yp)
    print("split ALEATÓRIO (vazamento)", name, f"F1={leak[name]['f1']:.3f}", flush=True)
res["_leak_random_split"] = leak

# escolha: melhor F1 médio na validação cruzada por blocos
cands = {k: v for k, v in res.items() if not k.startswith("_") and "baseline" not in k}
best = max(cands, key=lambda k: cands[k]["cv"]["f1"])
res["_best"] = best
print("MELHOR (CV):", best)
json.dump(res, open(os.path.join(ROOT, "results", "model_results.json"), "w"), ensure_ascii=False, indent=1)

# modelo final: refit com todos os dados, e importâncias
final = clone(models[best]).fit(df, y)
ComplexityClassifier(final, cols).save(os.path.join(ROOT, "models", "complexity.joblib"))
feat_name = "HistGB · features"
ComplexityClassifier(clone(models[feat_name]).fit(df, y), cols).save(os.path.join(ROOT, "models", "complexity_features.joblib"))
print("modelos salvos")
