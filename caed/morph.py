"""Regras morfológicas do português usadas pelas mutações (flexão de gênero/número e conjugação regular)."""
from __future__ import annotations

SPECIAL_S_PLURAL = {"irmão", "cidadão", "cristão", "mão", "pagão", "cortesão", "anão", "órgão", "órfão", "grão", "chão", "vão", "são"}
INVARIANT_ADJ_OR = {"maior", "menor", "melhor", "pior", "superior", "inferior", "exterior", "interior", "anterior", "posterior", "major"}
ADJ_FEM_SPECIAL = {"bom": "boa", "mau": "má", "europeu": "europeia", "hebreu": "hebreia", "judeu": "judia", "espanhol": "espanhola"}

IRREGULAR_3 = {  # lema: (pres 3sg, pres 3pl, pret 3sg, pret 3pl, imp 3sg, imp 3pl, fut 3sg, fut 3pl)
    "ler": ("lê", "leem", "leu", "leram", "lia", "liam", "lerá", "lerão"),
    "fazer": ("faz", "fazem", "fez", "fizeram", "fazia", "faziam", "fará", "farão"),
    "dizer": ("diz", "dizem", "disse", "disseram", "dizia", "diziam", "dirá", "dirão"),
    "trazer": ("traz", "trazem", "trouxe", "trouxeram", "trazia", "traziam", "trará", "trarão"),
    "ver": ("vê", "veem", "viu", "viram", "via", "viam", "verá", "verão"),
    "dar": ("dá", "dão", "deu", "deram", "dava", "davam", "dará", "darão"),
    "ter": ("tem", "têm", "teve", "tiveram", "tinha", "tinham", "terá", "terão"),
    "poder": ("pode", "podem", "pôde", "puderam", "podia", "podiam", "poderá", "poderão"),
    "querer": ("quer", "querem", "quis", "quiseram", "queria", "queriam", "quererá", "quererão"),
    "saber": ("sabe", "sabem", "soube", "souberam", "sabia", "sabiam", "saberá", "saberão"),
    "pôr": ("põe", "põem", "pôs", "puseram", "punha", "punham", "porá", "porão"),
    "vir": ("vem", "vêm", "veio", "vieram", "vinha", "vinham", "virá", "virão"),
    "ir": ("vai", "vão", "foi", "foram", "ia", "iam", "irá", "irão"),
    "ser": ("é", "são", "foi", "foram", "era", "eram", "será", "serão"),
    "estar": ("está", "estão", "esteve", "estiveram", "estava", "estavam", "estará", "estarão"),
    "construir": ("constrói", "constroem", "construiu", "construíram", "construía", "construíam", "construirá", "construirão"),
    "ouvir": ("ouve", "ouvem", "ouviu", "ouviram", "ouvia", "ouviam", "ouvirá", "ouvirão"),
    "pedir": ("pede", "pedem", "pediu", "pediram", "pedia", "pediam", "pedirá", "pedirão"),
    "medir": ("mede", "medem", "mediu", "mediram", "media", "mediam", "medirá", "medirão"),
}
IRREGULAR_PART = {"fazer": "feito", "dizer": "dito", "ver": "visto", "pôr": "posto", "abrir": "aberto", "escrever": "escrito",
                  "vir": "vindo", "cobrir": "coberto", "descobrir": "descoberto", "ganhar": "ganho", "gastar": "gasto",
                  "ler": "lido", "ter": "tido", "trazer": "trazido", "ser": "sido"}
_TENSE_IDX = {("Pres", "Sing"): 0, ("Pres", "Plur"): 1, ("Past", "Sing"): 2, ("Past", "Plur"): 3,
              ("Imp", "Sing"): 4, ("Imp", "Plur"): 5, ("Fut", "Sing"): 6, ("Fut", "Plur"): 7}
_REG = {  # sufixo por conjugação
    "ar": {"Pres": ("a", "am"), "Past": ("ou", "aram"), "Imp": ("ava", "avam"), "Fut": ("ará", "arão")},
    "er": {"Pres": ("e", "em"), "Past": ("eu", "eram"), "Imp": ("ia", "iam"), "Fut": ("erá", "erão")},
    "ir": {"Pres": ("e", "em"), "Past": ("iu", "iram"), "Imp": ("ia", "iam"), "Fut": ("irá", "irão")},
}


def pluralize(w: str) -> str:
    wl = w.lower()
    if wl.endswith("ão"):
        return w + "s" if wl in SPECIAL_S_PLURAL else w[:-2] + "ões"
    if wl.endswith("ã"):
        return w + "s"
    if wl[-1] in "aeiouáéíóúâêôàõ":
        return w + "s"
    if wl.endswith("m"):
        return w[:-1] + "ns"
    if wl.endswith(("r", "z")):
        return w + "es"
    if wl.endswith("s"):
        if wl.endswith("ês"):
            return w[:-2] + "eses"
        return w if any(c in wl for c in "áéíóúâêô") or wl.endswith(("is", "us")) else w + "es"
    if wl.endswith("al"):
        return w[:-2] + "ais"
    if wl.endswith("el"):
        return w[:-2] + "éis"
    if wl.endswith("ol"):
        return w[:-2] + "óis"
    if wl.endswith("ul"):
        return w[:-2] + "uis"
    if wl.endswith("il"):
        return (w[:-2] + "eis") if any(c in wl for c in "áéíóú") else (w[:-2] + "is")
    return w + "s"


def adj_gender(masc: str, gender: str) -> str:
    """Feminino/masculino singular de um adjetivo a partir da forma masculina."""
    if gender != "Fem":
        return masc
    m = masc.lower()
    if m in ADJ_FEM_SPECIAL:
        return ADJ_FEM_SPECIAL[m]
    if m in INVARIANT_ADJ_OR:
        return masc
    if m.endswith("ês"):
        return masc[:-2] + "esa"
    if m.endswith("or"):
        return masc + "a"
    if m.endswith("ão"):
        return masc[:-2] + "ã"
    if m.endswith("u"):
        return masc + "a"
    if m.endswith("o"):
        return masc[:-1] + "a"
    return masc


def inflect_adj(masc: str, gender: str, number: str) -> str:
    f = adj_gender(masc, gender)
    return pluralize(f) if number == "Plur" else f


def participle_masc(inf: str) -> str | None:
    if inf in IRREGULAR_PART:
        return IRREGULAR_PART[inf]
    if inf.endswith("ar"):
        return inf[:-2] + "ado"
    if inf.endswith(("er", "ir")):
        return inf[:-2] + "ido"
    if inf.endswith("ôr"):
        return None
    return None


def participle(inf: str, gender: str, number: str) -> str | None:
    pm = participle_masc(inf)
    return inflect_adj(pm, gender, number) if pm else None


def conjugate(inf: str, tense: str, number: str) -> str | None:
    """3ª pessoa do indicativo: Pres / Past (pretérito perfeito) / Imp / Fut."""
    idx = _TENSE_IDX.get((tense, number))
    if idx is None:
        return None
    if inf in IRREGULAR_3:
        return IRREGULAR_3[inf][idx]
    for end, tbl in _REG.items():
        if inf.endswith(end) and len(inf) > 3:
            suf = tbl[tense][0 if number == "Sing" else 1]
            stem = inf[:-2]
            if end == "er" and stem.endswith(("ce", "ge")) is False and tense == "Pres" and number == "Sing":
                pass
            return stem + suf
    return None


def match_case(src: str, new: str) -> str:
    if len(src) > 1 and src.isupper():
        return new.upper()
    if src[:1].isupper():
        return new[:1].upper() + new[1:]
    return new


_REV_PART = {v: k for k, v in IRREGULAR_PART.items()}


def participle_to_inf(word: str, freq: dict | None = None) -> str | None:
    """Particípio (qualquer gênero/número) -> infinitivo."""
    w = word.lower()
    if w.endswith("s"):
        w = w[:-1]
    if w.endswith("a"):
        w = w[:-1] + "o"
    if w in _REV_PART:
        return _REV_PART[w]
    if w.endswith("ado"):
        return w[:-3] + "ar"
    if w.endswith("ido"):
        stem = w[:-3]
        for suf in ("er", "ir"):                    # ambíguo: escolhe o que existe no corpus
            if freq is None or (stem + suf) in freq:
                return stem + suf
    return None
