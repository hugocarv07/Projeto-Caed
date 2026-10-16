from caed.syllables import is_canonical, syllabify, canonicity

CANON = ["cabeça", "canela", "janela", "luta", "sopa", "bola", "casa", "chuva", "carro", "cavalo",
         "maçã", "bebe", "queda", "gula", "cachorro", "sapato", "menino"]
NON = ["abajur", "lápis", "cansado", "asa", "alheio", "pai", "mãe", "quando", "campo", "mar", "história",
       "escola", "problema", "ônibus", "água", "ideia", "criança", "leite", "ao"]

def test_canonical():
    for w in CANON:
        assert is_canonical(w), (w, canonicity(w))

def test_non_canonical():
    for w in NON:
        assert not is_canonical(w), (w, canonicity(w))

def test_syllable_division():
    assert [s.text for s in syllabify("janela")] == ["ja", "ne", "la"]
    assert [s.text for s in syllabify("cansado")] == ["can", "sa", "do"]
    assert [s.text for s in syllabify("abajur")] == ["a", "ba", "jur"]
    assert [s.text for s in syllabify("problema")] == ["pro", "ble", "ma"]
    assert [s.text for s in syllabify("lápis")] == ["lá", "pis"]
