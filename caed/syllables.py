"""Silabificação do português (heurística baseada em regras) e canonicidade de palavras.

Palavra canônica = todas as sílabas são do tipo CV (consoante + vogal):
cabeça, canela, janela, luta, sopa  -> canônicas
abajur, lápis, cansado, asa, alheio -> não canônicas

Decisões de projeto (ortografia -> estrutura silábica):
  * dígrafos ch, lh, nh, rr, ss contam como UMA consoante (cachorro = ca-cho-rro -> CV-CV-CV, canônica).
  * 'qu'/'gu' antes de e/i (que, guia) = uma consoante (o 'u' é mudo); antes de a/o (quando, água)
    o 'u' é semivogal -> a sílaba deixa de ser CV.
  * 'ç' e 'x' são consoantes simples; 'h' sozinho é tratado como consoante (hoje -> ho-je);
  * vogais nasais til (ã, õ) são uma vogal (maçã = ma-çã, CV-CV);
  * m/n em coda (campo, ontem) e s/r/l/z finais (lápis, mar) formam coda -> sílaba CVC -> não canônica;
  * ditongos e hiatos formam sílabas com núcleo V+V ou sílaba só-vogal (asa, pai, mãe) -> não canônicas.
"""
from __future__ import annotations
import re
import unicodedata
from dataclasses import dataclass

VOWELS = set("aeiouáéíóúâêôãõàü")
STRONG = set("aeoáéóâêôãõà")           # vogais "fortes" (formam hiato entre si)
ACCENTED_WEAK = set("íú")              # i/u tônicos -> hiato com a vogal anterior
DIGRAPHS = ("ch", "lh", "nh", "rr", "ss")
# encontros consonantais que podem iniciar sílaba (onset complexo)
VALID_ONSETS = {"bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr", "pl", "pr", "tr", "vr"}
_LETTERS = re.compile(r"[a-zçáéíóúâêôãõàü]+")


def _tokenize(word: str):
    """Quebra a palavra em unidades ('C'|'V', texto). Dígrafos viram uma unidade 'C'."""
    w = word.lower()
    units = []
    i = 0
    while i < len(w):
        ch = w[i]
        two = w[i:i + 2]
        if two in DIGRAPHS:
            units.append(("C", two)); i += 2; continue
        if two in ("qu", "gu") and i + 2 < len(w) and w[i + 2] in "eéêií":
            units.append(("C", two)); i += 2; continue          # u mudo
        if ch in VOWELS:
            units.append(("V", ch))
        else:
            units.append(("C", ch))
        i += 1
    return units


def _group_nuclei(units):
    """Agrupa vogais adjacentes em núcleos (ditongo) ou separa (hiato)."""
    out = []
    for kind, t in units:
        if kind == "V" and out and out[-1][0] == "V":
            prev = out[-1][1]
            last = prev[-1]
            hiatus = (last in STRONG and t in STRONG) or t in ACCENTED_WEAK or last in ACCENTED_WEAK
            if hiatus:
                out.append((kind, t))
            else:
                out[-1] = ("V", prev + t)          # ditongo/tritongo
        else:
            out.append((kind, t))
    return out


@dataclass
class Syllable:
    onset: str
    nucleus: str
    coda: str

    @property
    def text(self):
        return self.onset + self.nucleus + self.coda

    @property
    def shape(self):
        """Forma CV: 'C' por unidade consonantal, 'V' por vogal (ditongo -> 'VV')."""
        c = lambda s: "".join("C" for _ in _split_cons(s))
        return c(self.onset) + "V" * len(self.nucleus) + c(self.coda)


def _split_cons(s: str):
    """Divide uma string de consoantes em unidades (dígrafos contam 1)."""
    res, i = [], 0
    while i < len(s):
        if s[i:i + 2] in DIGRAPHS or (s[i:i + 2] in ("qu", "gu") and len(s[i:]) == 2):
            res.append(s[i:i + 2]); i += 2
        else:
            res.append(s[i]); i += 1
    return res


def syllabify(word: str) -> list[Syllable]:
    """Divide uma palavra em sílabas pelo princípio do onset máximo."""
    word = unicodedata.normalize("NFC", word.lower())
    units = _group_nuclei(_tokenize("".join(_LETTERS.findall(word))))
    # índices dos núcleos
    nuc = [i for i, (k, _) in enumerate(units) if k == "V"]
    if not nuc:
        return []                                   # sem vogal (sigla etc.)
    sylls = []
    for n, ni in enumerate(nuc):
        prev_end = nuc[n - 1] if n else -1
        next_start = nuc[n + 1] if n + 1 < len(nuc) else len(units)
        cons_before = [t for _, t in units[prev_end + 1:ni]]
        # onset: se é a 1ª sílaba, todas as consoantes iniciais; senão decide-se a divisão
        if n == 0:
            onset = "".join(cons_before)
        else:
            onset = ""   # preenchido pelo particionamento da sílaba anterior
        sylls.append([onset, units[ni][1], ""])
    # distribui consoantes intervocálicas
    for n in range(len(nuc) - 1):
        cons = [t for _, t in units[nuc[n] + 1:nuc[n + 1]]]
        if len(cons) == 0:
            continue
        if len(cons) == 1:
            split = 0
        elif len(cons) == 2 and (cons[0] + cons[1]) in VALID_ONSETS:
            split = 0
        elif len(cons) >= 3 and (cons[-2] + cons[-1]) in VALID_ONSETS:
            split = len(cons) - 2
        else:
            split = len(cons) - 1
        sylls[n][2] = "".join(cons[:split])
        sylls[n + 1][0] = "".join(cons[split:])
    # coda da última sílaba
    tail = [t for _, t in units[nuc[-1] + 1:]]
    sylls[-1][2] = "".join(tail)
    return [Syllable(*s) for s in sylls]


def syllable_shapes(word: str) -> list[str]:
    return [s.shape for s in syllabify(word)]


def is_canonical(word: str) -> bool:
    """True se TODAS as sílabas da palavra são CV."""
    shapes = syllable_shapes(word)
    return bool(shapes) and all(s == "CV" for s in shapes)


def canonicity(word: str) -> dict:
    """Classifica a palavra e devolve detalhes (útil para a interface)."""
    sy = syllabify(word)
    shapes = [s.shape for s in sy]
    non_cv = [s.text for s in sy if s.shape != "CV"]
    return {
        "word": word,
        "syllables": [s.text for s in sy],
        "shapes": shapes,
        "canonical": bool(shapes) and not non_cv,
        "non_canonical_syllables": non_cv,
    }


def words_of(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÿ]+(?:-[A-Za-zÀ-ÿ]+)*", text)
