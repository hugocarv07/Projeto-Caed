"""Aplicação web: classifica a complexidade de um texto e gera novas versões por mutação.

    python app.py            ->  http://127.0.0.1:5000
"""
import difflib
import os
import re

from flask import Flask, jsonify, render_template, request

from caed.data import LEVEL_NAMES
from caed.model import ComplexityClassifier
from caed.mutations import Mutator
from caed.syllables import canonicity, words_of

ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

MAIN = ComplexityClassifier.load(os.path.join(ROOT, "models", "complexity.joblib"))
FEAT_PATH = os.path.join(ROOT, "models", "complexity_features.joblib")
FEAT = ComplexityClassifier.load(FEAT_PATH) if os.path.exists(FEAT_PATH) else None
MUT = Mutator()


TOKEN = re.compile(r"\s+|\w+|[^\w\s]", re.UNICODE)


def analyze(text: str) -> dict:
    res = {"main": MAIN.analyze(text)}
    if FEAT is not None:
        res["features"] = FEAT.analyze(text)
    ws = words_of(text)
    infos = [canonicity(w) for w in ws]
    can = [i for i in infos if i["shapes"]]
    res["canonicity"] = {
        "ratio": (sum(i["canonical"] for i in can) / len(can)) if can else 0.0,
        "n_words": len(ws),
        "words": [{"w": i["word"], "syl": "·".join(i["syllables"]), "ok": i["canonical"]} for i in infos],
    }
    return res


def diff_ops(a: str, b: str) -> list:
    ta, tb = TOKEN.findall(a), TOKEN.findall(b)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        if tag == "equal":
            out.append({"t": "eq", "x": "".join(ta[i1:i2])})
        else:
            if i2 > i1:
                out.append({"t": "del", "x": "".join(ta[i1:i2])})
            if j2 > j1:
                out.append({"t": "ins", "x": "".join(tb[j1:j2])})
    return out


@app.get("/")
def index():
    return render_template("index.html", levels=LEVEL_NAMES)


@app.post("/api/analyze")
def api_analyze():
    text = (request.json or {}).get("text", "").strip()
    if len(text) < 10:
        return jsonify(error="Digite um texto um pouco maior."), 400
    return jsonify(analyze(text))


@app.post("/api/mutate")
def api_mutate():
    d = request.json or {}
    text = d.get("text", "").strip()
    if len(text) < 10:
        return jsonify(error="Digite um texto um pouco maior."), 400
    ops = [o for o in d.get("ops", []) if o in ("word", "gender", "paraphrase")]
    if not ops:
        return jsonify(error="Escolha ao menos um operador de mutação."), 400
    seed = d.get("seed")
    r = MUT.mutate(text, ops=ops, rate=float(d.get("rate", 0.3)), seed=int(seed) if seed not in (None, "") else None,
                   word_mode=d.get("word_mode", "mixed"), direction=d.get("direction", "random"))
    before, after = analyze(text), analyze(r.text)
    return jsonify(text=r.text, changes=r.changes, diff=diff_ops(text, r.text), before=before, after=after)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
