"""Recursos lexicais: sinônimos/antônimos (OpenWN-PT via OMW) e banco de formas flexionadas do próprio corpus."""
from __future__ import annotations
import json
import math
import os
import re
from collections import defaultdict

from .syllables import is_canonical, syllabify

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_SIM = 0.30
NLTK_DATA = os.path.join(ROOT, "data", "nltk_data")
POS_MAP = {"NOUN": "n", "VERB": "v", "ADJ": "a", "ADV": "r"}


def _strip_accents(w: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn")


def _spelling_variant(a: str, b: str) -> bool:
    """'protecção'~'proteção', 'freqüentes'~'frequentes': variantes ortográficas não são trocas interessantes."""
    import difflib
    from .morph import pluralize
    a2, b2 = _strip_accents(a.lower()), _strip_accents(b.lower())
    if pluralize(a.lower()) == b.lower() or pluralize(b.lower()) == a.lower():
        return True
    return a2 == b2 or difflib.SequenceMatcher(None, a2, b2).ratio() > 0.88


def feat_key(lemma: str, pos: str, morph: dict) -> str:
    keep = ("Gender", "Number", "Tense", "Mood", "Person", "VerbForm")
    return "|".join([lemma.lower(), pos] + [f"{k}={morph[k]}" for k in sorted(morph) if k in keep])


class Lexicon:
    def __init__(self, processed_dir: str | None = None):
        processed_dir = processed_dir or os.path.join(ROOT, "data", "processed")
        lex = json.load(open(os.path.join(processed_dir, "lexicon.json"), encoding="utf-8"))
        self.forms: dict[str, str] = lex["forms"]
        self.freq: dict[str, int] = lex["freq"]
        self._load_wordnet()
        self._load_vectors(processed_dir)


    def _load_vectors(self, processed_dir):
        import numpy as np
        path = os.path.join(processed_dir, "vectors.npz")
        self.vec_idx, self.vec = {}, None
        if os.path.exists(path):
            d = np.load(path)
            self.vec_idx = {w: i for i, w in enumerate(d["vocab"])}
            self.vec = d["vec"].astype("float32")

    def sim(self, a: str, b: str) -> float | None:
        if self.vec is None or a not in self.vec_idx or b not in self.vec_idx:
            return None
        return float(self.vec[self.vec_idx[a]] @ self.vec[self.vec_idx[b]])

    def context_fit(self, word: str, context: list[str]) -> float | None:
        """Cosseno entre a palavra e o vetor médio das palavras de contexto (mede se 'cabe' na frase)."""
        import numpy as np
        if self.vec is None or word not in self.vec_idx:
            return None
        ids = [self.vec_idx[c] for c in context if c in self.vec_idx and c != word]
        if len(ids) < 2:
            return None
        m = self.vec[ids].mean(0)
        n = np.linalg.norm(m)
        return float(self.vec[self.vec_idx[word]] @ m / n) if n > 0 else None


    def _load_wordnet(self):
        tab = os.path.join(NLTK_DATA, "corpora", "omw-1.4", "por", "wn-data-por.tab")
        self.syn_lemmas = defaultdict(list)
        self.lemma_syns = defaultdict(set)
        for line in open(tab, encoding="utf-8"):
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[1] != "lemma":
                continue
            sid, lemma = parts[0], parts[2].lower()
            pos = sid[-1].replace("s", "a")
            if "_" in lemma or " " in lemma or not re.fullmatch(r"[a-zà-ÿç]+", lemma):
                continue
            sid = sid[:-1] + pos
            self.syn_lemmas[sid].append(lemma)
            self.lemma_syns[(lemma, pos)].add(sid)
        self._ant_cache: dict = {}
        try:
            import nltk
            if NLTK_DATA not in nltk.data.path:
                nltk.data.path.insert(0, NLTK_DATA)
            from nltk.corpus import wordnet as wn
            self.wn = wn
            wn.ensure_loaded()
        except Exception:
            self.wn = None

    def f(self, w: str) -> int:
        return self.freq.get(w.lower(), 0)

    def synonyms(self, lemma: str, pos: str) -> list[str]:
        """Duas camadas: primeiro sinônimos estritos (Jaccard>=0.2); se nada, camada relaxada (>=0.06) só com palavras frequentes."""
        res = self._synonyms(lemma, pos, 3, 0.2)
        if not res:
            res = self._synonyms(lemma, pos, 15, 0.06)
        res = [c for c in res if not _spelling_variant(lemma, c)]
        if self.vec is not None:
            scored = [(c, self.sim(lemma.lower(), c)) for c in res]
            res = [c for c, sc in sorted(scored, key=lambda x: -(x[1] or 0)) if sc is not None and sc >= MIN_SIM]
        if pos == "ADJ":
            norm = []
            for c in res:
                if c.endswith("a") and c[:-1] + "o" in self.freq:
                    c = c[:-1] + "o"
                if c != lemma.lower() and c not in norm:
                    norm.append(c)
            res = norm
        return res

    def _synonyms(self, lemma: str, pos: str, min_freq: int, min_jaccard: float) -> list[str]:
        """Sinônimos ordenados por (compartilhamento de sentidos x frequência).

        Sem desambiguação de sentido, 'casa' traria 'classe'. Como atenuação, exige-se que o candidato
        compartilhe uma fração mínima (Jaccard) dos synsets do lema original: sinônimos "de verdade"
        costumam compartilhar vários sentidos; palavras ligadas por um sentido raro, quase nenhum.
        """
        p = POS_MAP.get(pos)
        src = self.lemma_syns.get((lemma.lower(), p), set())
        out = {}
        for sid in src:
            for l in self.syn_lemmas[sid]:
                if l == lemma.lower() or self.f(l) < min_freq or l in out:
                    continue
                cand = self.lemma_syns.get((l, p), set())
                j = len(src & cand) / max(len(src | cand), 1)
                if j >= min_jaccard:
                    out[l] = j * math.log1p(self.f(l))
        return sorted(out, key=out.get, reverse=True)

    def antonyms(self, lemma: str, pos: str, min_freq: int = 3) -> list[str]:
        """Antônimos: relação de antonímia do WordNet inglês (mesmos offsets) projetada nos lemas PT."""
        p = POS_MAP.get(pos)
        if self.wn is None or p not in ("a", "r"):
            return []
        key = (lemma.lower(), p)
        if key in self._ant_cache:
            return self._ant_cache[key]
        res = {}
        for sid in self.lemma_syns.get(key, ()):
            off, ps = int(sid[:8]), sid[-1]
            for pp in ([ps] + (["s"] if ps == "a" else [])):
                try:
                    syn = self.wn.synset_from_pos_and_offset(pp, off)
                except Exception:
                    continue
                for l in syn.lemmas():
                    for a in l.antonyms():
                        asid = f"{a.synset().offset():08d}-{'a' if a.synset().pos() in 'as' else a.synset().pos()}"
                        for pl in self.syn_lemmas.get(asid, ()):
                            if pl != lemma.lower() and self.f(pl) >= min_freq:
                                res[pl] = self.f(pl)
        self._ant_cache[key] = sorted(res, key=res.get, reverse=True)
        return self._ant_cache[key]

    def difficulty(self, w: str) -> float:
        w = w.lower()
        nsyl = max(len(syllabify(w)), 1)
        return (0.35 * len(w) + 0.6 * nsyl + (0 if is_canonical(w) else 1.0)
                - 0.45 * math.log1p(self.f(w)))

    def form(self, lemma: str, pos: str, **morph) -> str | None:
        return self.forms.get(feat_key(lemma, pos, morph))
