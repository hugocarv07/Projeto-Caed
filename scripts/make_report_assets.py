"""Passo 4: gera figuras e tabelas do relatório a partir de results/*.json (+ importância de features)."""
import json, os, sys, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from caed.model import make_models, numeric_cols

R = os.path.join(ROOT, "results")
res = json.load(open(os.path.join(R, "model_results.json")))
mut = json.load(open(os.path.join(R, "mutation_results.json")))
INK, ACC, ACC2, MUT = "#1d2433", "#1f5eff", "#e8590c", "#9aa3b2"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUT, "text.color": INK,
                     "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK})

# 1) comparação de modelos
names = [k for k in res if not k.startswith("_")]
names = sorted(names, key=lambda k: res[k]["cv"]["f1"])
fig, ax = plt.subplots(figsize=(8.2, 4.4))
y = np.arange(len(names))
ax.barh(y - 0.2, [res[k]["cv"]["f1"] for k in names], 0.4, color=ACC, label="validação cruzada por blocos (F1 macro)")
ax.barh(y + 0.2, [res[k]["test"]["f1"] for k in names], 0.4, color=ACC2, label="teste final, bloco 4 (F1 macro)")
leak = res["_leak_random_split"]
for k, v in leak.items():
    ax.plot(v["f1"], names.index(k) - 0.2, marker="D", color="black", ms=5, ls="", label="divisão ALEATÓRIA (vazamento)" if k == list(leak)[0] else None)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5); ax.set_xlim(0, 1)
ax.set_xlabel("F1 macro"); ax.legend(loc="lower right", fontsize=8, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig_models.png"), dpi=160); plt.close()

# 2) matrizes de confusão (teste final)
labels = ["EF I", "EF II", "EM", "ES"]
fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.8))
for ax, k in zip(axs, ["TF-IDF chars · LR", "HistGB · features"]):
    m = np.array(res[k]["test_confusion"]); mn = m / m.sum(1, keepdims=True)
    ax.imshow(mn, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{mn[i, j]:.2f}", ha="center", va="center", color="white" if mn[i, j] > .5 else INK, fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(labels); ax.set_yticks(range(4)); ax.set_yticklabels(labels)
    ax.set_title(k, fontsize=10); ax.set_xlabel("previsto"); ax.set_ylabel("real")
plt.tight_layout(); plt.savefig(os.path.join(R, "fig_confusion.png"), dpi=160); plt.close()

# 3) efeito das mutações
rs = mut["results"]
order = list(rs)
fig, ax = plt.subplots(figsize=(8.2, 4.2))
y = np.arange(len(order))
for off, key, col in ((-0.2, "TF-IDF chars", ACC), (0.2, "HistGB features", ACC2)):
    ax.barh(y + off, [rs[o][key]["mean_delta"] for o in order], 0.4, color=col, label=key)
    ax.errorbar([rs[o][key]["mean_delta"] for o in order], y + off, xerr=[rs[o][key]["std_delta"] / np.sqrt(mut["n"]) * 1.96 for o in order], fmt="none", ecolor=INK, lw=1)
ax.axvline(0, color=MUT, lw=1); ax.set_yticks(y); ax.set_yticklabels(order, fontsize=8.5); ax.invert_yaxis()
ax.set_xlabel("Δ médio do escore de complexidade (0–3), IC 95%"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig_mutations.png"), dpi=160); plt.close()

# 4) importância de features (HistGB treinado em dev, permutação no bloco 4)
P = os.path.join(ROOT, "data", "processed")
df = pd.read_csv(os.path.join(P, "chunks.csv")).drop(columns=["n_words"]).merge(pd.read_csv(os.path.join(P, "features.csv")), on="chunk_id")
cols = numeric_cols(pd.read_csv(os.path.join(P, "features.csv")))
dev, test = df[df.block < 4], df[df.block == 4]
m = clone(make_models(cols)["HistGB · features"]).fit(dev, dev.level.values)
pi = permutation_importance(m, test[cols + ["text"]], test.level.values, scoring="f1_macro", n_repeats=5, random_state=0, n_jobs=1)
imp = pd.Series(pi.importances_mean, index=cols + ["text"]).drop("text").sort_values(ascending=False)
imp.head(15).to_csv(os.path.join(R, "feature_importance_top15.csv"), header=["queda_F1_ao_embaralhar"])
fig, ax = plt.subplots(figsize=(7.2, 4.2))
top = imp.head(12)[::-1]
ax.barh(top.index, top.values, color=[ACC2 if (i.startswith(("canonical", "noncanon", "shape_"))) else ACC for i in top.index])
ax.set_xlabel("queda do F1 macro ao embaralhar a feature (laranja = canonicidade/sílabas)")
plt.tight_layout(); plt.savefig(os.path.join(R, "fig_importance.png"), dpi=160); plt.close()
print(imp.head(15).round(4).to_string())

# 5) canonicidade por nível
g = df.groupby("level")[["canonical_word_ratio", "canonical_type_ratio", "shape_CV", "noncanon_syll_per_word"]].mean().round(3)
g.to_csv(os.path.join(R, "canonicity_by_level.csv")); print(g)
