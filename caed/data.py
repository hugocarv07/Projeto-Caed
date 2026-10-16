"""Leitura, limpeza e segmentação do dataset do CAEd em instâncias (chunks)."""
from __future__ import annotations
import os
import re
import pandas as pd

LEVELS = {
    "1_Ensino_Fundamental_I": 0,
    "2_Ensino_Fundamental_II": 1,
    "3_Ensino_Medio": 2,
    "4_Ensino_Superior": 3,
}
LEVEL_NAMES = ["Ensino Fundamental I", "Ensino Fundamental II", "Ensino Médio", "Ensino Superior"]

_SENT_SPLIT = re.compile(r'(?<=[.!?…])["”»)]*\s+(?=["“«(]?[A-ZÀ-Ý0-9])')
_END_PUNCT = re.compile(r'[.!?…:;"”»)]\s*$')


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def n_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def clean_lines(raw: str) -> list[str]:
    """Remove ruído de extração: títulos, índices, caminhos de navegação (ex.: 'Cálculo/Geometria').

    Sem esta limpeza o modelo poderia aprender "artefatos de formatação" (30% das linhas de
    Ensino Médio/Superior são títulos sem pontuação; no EF-I quase nenhuma) em vez de complexidade.
    """
    kept = []
    for line in raw.replace("﻿", "").split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        w = n_words(line)
        letters = sum(c.isalpha() for c in line)
        if letters / max(len(line), 1) < 0.6:
            continue
        if not _END_PUNCT.search(line) and w < 25:
            continue
        if "/" in line and w <= 12 and not re.search(r"[.!?]\s", line):
            continue
        if w < 6:
            continue
        kept.append(line)
    return kept


def make_chunks(lines: list[str], min_words: int = 30, target: int = 55, max_words: int = 130) -> list[str]:
    """Agrupa sentenças consecutivas (dentro do mesmo parágrafo) em textos curtos de ~target palavras."""
    chunks = []
    for line in lines:
        cur, cw = [], 0
        for s in split_sentences(line):
            sw = n_words(s)
            if cur and cw + sw > max_words:
                chunks.append(" ".join(cur)); cur, cw = [], 0
            cur.append(s); cw += sw
            if cw >= target:
                chunks.append(" ".join(cur)); cur, cw = [], 0
        if cur:
            if cw >= min_words:
                chunks.append(" ".join(cur))
            elif chunks and cw:
                chunks[-1] = chunks[-1] + " " + " ".join(cur)
    return [c for c in chunks if n_words(c) >= min_words]


def build_dataset(root: str, n_blocks: int = 5) -> pd.DataFrame:
    """Gera o DataFrame de instâncias.

    Colunas: chunk_id, file, file_idx, level, block, text, n_words.
    `block` agrupa arquivos contíguos (por nº de ordem dentro do nível): arquivos vizinhos vêm do mesmo
    material didático, então particionar por blocos aproxima uma separação "por material" e evita
    vazamento entre treino e teste.
    """
    rows = []
    for d, lvl in LEVELS.items():
        files = sorted(os.listdir(os.path.join(root, d)), key=lambda f: int(f.split("_")[0]))
        files = [f for f in files if f.endswith(".txt")]
        for pos, f in enumerate(files):
            raw = open(os.path.join(root, d, f), encoding="utf-8-sig").read()
            for k, ch in enumerate(make_chunks(clean_lines(raw))):
                rows.append(dict(chunk_id=f"{f[:-4]}#{k}", file=f"{d}/{f}", file_idx=pos, file_rank=pos / len(files),
                                 level=lvl, block=min(int(pos / len(files) * n_blocks), n_blocks - 1),
                                 text=ch, n_words=n_words(ch)))
    return pd.DataFrame(rows)
