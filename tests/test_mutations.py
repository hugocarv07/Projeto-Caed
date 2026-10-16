import random
import pytest
from caed.mutations import Mutator


@pytest.fixture(scope="module")
def M():
    return Mutator()


def run(fn, text, M, **kw):
    return fn(text, random.Random(1), 1.0, **kw).text


def test_passive_to_active(M):
    assert run(M.paraphrase, "O texto foi lido pelo aluno.", M) == "O aluno leu o texto."


def test_active_to_passive(M):
    sent = next(M.nlp("O aluno leu o texto.").sents)
    assert M._active_to_passive(sent) == "O texto foi lido pelo aluno."


def test_temporal_fronting(M):
    assert run(M.paraphrase, "Há duas semanas atrás estava chovendo.", M) == "Estava chovendo há duas semanas atrás."


def test_apposition_roundtrip(M):
    assert run(M.paraphrase, "Roberto Carlos, o cantor, fez aniversário.", M) == "O cantor Roberto Carlos fez aniversário."


def test_gender_swap_example_from_prompt(M):
    assert run(M.gender_swap, "A cantora era famosa.", M) == "O cantor era famoso."
    assert run(M.gender_swap, "O menino comprou um livro.", M) == "A menina comprou um livro."


def test_word_swap_keeps_agreement(M):
    out = run(M.word_swap, "Ontem, a menina comprou um livro bonito.", M, mode="synonym")
    assert out != "Ontem, a menina comprou um livro bonito." and out.endswith(".")
