"""Extração de features de complexidade textual (superficiais, silábicas, lexicais e sintáticas)."""
from __future__ import annotations
import math
import re
from collections import Counter
import numpy as np

from .syllables import syllabify, words_of
from .data import split_sentences

SHAPES = ["CV", "V", "CVC", "CCV", "CVV", "CCVC", "VC", "CVVC", "other"]
POS_LIST = ["NOUN", "VERB", "ADJ", "ADV", "ADP", "DET", "PRON", "CCONJ", "SCONJ", "PROPN", "NUM", "AUX"]
SUBORD_DEPS = {"acl", "acl:relcl", "advcl", "ccomp", "xcomp", "csubj"}

_syl_cache: dict[str, tuple] = {}


def _word_syll_info(w: str):
    """(n_sílabas, [formas], canônica?) com cache."""
    k = w.lower()
    r = _syl_cache.get(k)
    if r is None:
        sy = syllabify(k)
        shapes = [s.shape for s in sy]
        r = (len(sy), shapes, bool(shapes) and all(s == "CV" for s in shapes))
        _syl_cache[k] = r
    return r


def surface_features(text: str) -> dict:
    sents = split_sentences(text) or [text]
    words = words_of(text)
    n = max(len(words), 1)
    wl = [len(w) for w in words] or [0]
    sl = [len(words_of(s)) for s in sents] or [0]
    infos = [_word_syll_info(w) for w in words]
    nsyl = [i[0] for i in infos] or [0]
    canon_flags = [i[2] for i in infos if i[0] > 0]
    shape_counts = Counter(s for i in infos for s in i[1])
    tot_syl = max(sum(shape_counts.values()), 1)
    lower = [w.lower() for w in words]
    types = set(lower)
    asl, asw = float(np.mean(sl)), float(np.mean(nsyl))

    win = 40
    if len(lower) >= win:
        mattr = float(np.mean([len(set(lower[i:i + win])) / win for i in range(0, len(lower) - win + 1, 5)]))
    else:
        mattr = len(types) / n
    caps_mid = sum(1 for s in sents for w in words_of(s)[1:] if w[:1].isupper())
    f = {
        "n_words": len(words), "n_sents": len(sents),
        "words_per_sent": asl, "words_per_sent_std": float(np.std(sl)), "words_per_sent_max": float(np.max(sl)),
        "long_sent_ratio": float(np.mean([x > 25 for x in sl])),
        "chars_per_word": float(np.mean(wl)),
        "long_word_ratio": float(np.mean([x >= 7 for x in wl])), "vlong_word_ratio": float(np.mean([x >= 10 for x in wl])),
        "syll_per_word": asw, "poly_word_ratio": float(np.mean([x >= 4 for x in nsyl])),
        "flesch_pt": 248.835 - 1.015 * asl - 84.6 * asw,
        # --- canonicidade (feature exigida pelo enunciado) ---
        "canonical_word_ratio": float(np.mean(canon_flags)) if canon_flags else 0.0,
        "canonical_type_ratio": float(np.mean([_word_syll_info(w)[2] for w in types])) if types else 0.0,
        "noncanon_syll_per_word": sum(1 for i in infos for s in i[1] if s != "CV") / n,
        "ttr": len(types) / n, "root_ttr": len(types) / math.sqrt(n), "mattr": mattr,
        "commas_per_sent": text.count(",") / len(sents), "semicolons_per_sent": text.count(";") / len(sents),
        "colons_per_sent": text.count(":") / len(sents), "parens_per_sent": text.count("(") / len(sents),
        "quotes_per_sent": (text.count('"') + text.count("“")) / len(sents),
        "dashes_per_sent": (text.count("—") + text.count("–")) / len(sents),
        "digit_ratio": sum(c.isdigit() for c in text) / max(len(text), 1),
        "caps_mid_ratio": caps_mid / n,
        "hyphen_word_ratio": sum("-" in w for w in words) / n,
        "mente_ratio": sum(w.lower().endswith("mente") for w in words) / n,
    }
    for s in SHAPES:
        f[f"shape_{s}"] = (sum(c for k, c in shape_counts.items() if (k if k in SHAPES else "other") == s)) / tot_syl
    return f


def _depth(tok):
    d = 0
    while tok.head is not tok:
        tok = tok.head; d += 1
        if d > 60: break
    return d


def syntactic_features(doc) -> dict:
    toks = [t for t in doc if not t.is_space]
    words = [t for t in toks if t.is_alpha] or toks
    n = max(len(words), 1)
    nsent = max(len(list(doc.sents)), 1)
    pos = Counter(t.pos_ for t in words)
    f = {f"pos_{p}": pos.get(p, 0) / n for p in POS_LIST}
    content = pos["NOUN"] + pos["VERB"] + pos["ADJ"] + pos["ADV"] + pos["PROPN"]
    f["lexical_density"] = content / n
    deps = Counter(t.dep_ for t in toks)
    f["subord_per_sent"] = sum(deps.get(d, 0) for d in SUBORD_DEPS) / nsent
    f["relcl_per_sent"] = deps.get("acl:relcl", 0) / nsent
    f["advcl_per_sent"] = deps.get("advcl", 0) / nsent
    f["coord_per_sent"] = deps.get("conj", 0) / nsent
    f["appos_per_sent"] = deps.get("appos", 0) / nsent
    f["passive_per_sent"] = sum(c for d, c in deps.items() if d.endswith(":pass")) / nsent
    dist = [abs(t.i - t.head.i) for t in toks if t.head is not t and t.pos_ != "PUNCT"]
    f["mean_dep_dist"] = float(np.mean(dist)) if dist else 0.0
    f["max_dep_dist"] = float(np.max(dist)) if dist else 0.0
    f["mean_tree_depth"] = float(np.mean([_depth(t) for t in toks])) if toks else 0.0
    f["max_tree_depth"] = float(np.max([_depth(t) for t in toks])) if toks else 0.0
    verbs = [t for t in words if t.pos_ in ("VERB", "AUX")]
    nv = max(len(verbs), 1)
    mc = Counter()
    for t in verbs:
        m = t.morph.to_dict()
        for k in ("Mood", "Tense", "VerbForm", "Person"):
            if k in m:
                mc[f"{k}={m[k].split(',')[0]}"] += 1
    f["verb_subj"] = mc["Mood=Sub"] / nv
    f["verb_fut"] = mc["Tense=Fut"] / nv
    f["verb_ger"] = mc["VerbForm=Ger"] / nv
    f["verb_inf"] = mc["VerbForm=Inf"] / nv
    f["verb_part"] = mc["VerbForm=Part"] / nv
    f["verb_pres"] = mc["Tense=Pres"] / nv
    f["verb_past"] = (mc["Tense=Past"] + mc["Tense=Imp"] + mc["Tense=Pqp"]) / nv
    f["verb_p1_2"] = (mc["Person=1"] + mc["Person=2"]) / nv
    f["verbs_per_sent"] = len(verbs) / nsent
    return f


def extract(text: str, doc) -> dict:
    f = surface_features(text)
    f.update(syntactic_features(doc))
    return f


def feature_names() -> list[str]:
    import spacy
    nlp = spacy.blank("pt")
    return list(surface_features("Um teste simples. Outro teste.").keys())  # reordenado em runtime
