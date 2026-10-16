"""Vetores distribucionais (PPMI + SVD) treinados no corpus, SEM usar rótulos.

Usados só pelo operador de troca de palavras, para filtrar sinônimos do WordNet que tenham outro sentido
(ex.: 'anos' -> 'sonos'). Saída: data/processed/vectors.npz
"""
import os, re, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from collections import Counter
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

texts = pd.read_csv(os.path.join(ROOT, "data", "processed", "chunks.csv")).text.tolist()
toks = [re.findall(r"[a-zà-ÿç]+", t.lower()) for t in texts]
freq = Counter(w for t in toks for w in t)
vocab = [w for w, c in freq.items() if c >= 4]
idx = {w: i for i, w in enumerate(vocab)}
W = 5
rows, cols = [], []
for t in toks:
    ids = [idx[w] for w in t if w in idx]
    for i, a in enumerate(ids):
        for j in range(max(0, i - W), min(len(ids), i + W + 1)):
            if j != i:
                rows.append(a); cols.append(ids[j])
C = sparse.coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(vocab),) * 2).tocsr()
tot = C.sum()
rs = np.asarray(C.sum(1)).ravel()
cs = np.asarray(C.sum(0)).ravel() ** 0.75
cs = cs / cs.sum() * tot
C = C.tocoo()
pmi = np.log((C.data * tot) / (rs[C.row] * cs[C.col] + 1e-9) + 1e-9)
keep = pmi > 0
P = sparse.coo_matrix((pmi[keep], (C.row[keep], C.col[keep])), shape=C.shape).tocsr()
V = TruncatedSVD(150, random_state=0).fit_transform(P)
V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
np.savez_compressed(os.path.join(ROOT, "data", "processed", "vectors.npz"), vocab=np.array(vocab), vec=V.astype(np.float16))
print("vocab", len(vocab), "dim", V.shape[1])
