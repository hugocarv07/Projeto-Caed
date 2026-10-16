"""Operadores de mutação de texto: troca de palavras, troca de gênero e paráfrase.

Todos usam heurísticas simples sobre a análise de dependências do spaCy + léxicos (OpenWN-PT) e
regras morfológicas do português. Cada operador devolve a lista de alterações (antes/depois) para
que a interface possa mostrar o que mudou.
"""
from __future__ import annotations
import random
import re
from dataclasses import dataclass, field

from . import morph
from .lexicon import Lexicon, POS_MAP, feat_key

# --------------------------------------------------------------------------------------------
# Recursos para troca de gênero
# --------------------------------------------------------------------------------------------
_PAIRS_TXT = """menino menina|garoto garota|rapaz moça|homem mulher|pai mãe|filho filha|irmão irmã|tio tia|avô avó|primo prima
sobrinho sobrinha|neto neta|marido esposa|namorado namorada|amigo amiga|vizinho vizinha|professor professora|aluno aluna
cantor cantora|ator atriz|autor autora|escritor escritora|pintor pintora|diretor diretora|médico médica|enfermeiro enfermeira
cozinheiro cozinheira|padeiro padeira|rei rainha|príncipe princesa|senhor senhora|dono dona|patrão patroa|leitor leitora
jogador jogadora|vendedor vendedora|trabalhador trabalhadora|governador governadora|deputado deputada|cidadão cidadã
gato gata|cachorro cachorra|cavalo égua|boi vaca|galo galinha|leão leoa|tigre tigresa|lobo loba|urso ursa|pato pata|porco porca
herói heroína|poeta poetisa|conde condessa|duque duquesa|bruxo bruxa|noivo noiva|viúvo viúva|aluno aluna|doutor doutora
engenheiro engenheira|advogado advogada|juiz juíza|prefeito prefeita|vereador vereadora|músico música|bailarino bailarina
moço moça|velho velha|rapazinho rapazinha|menininho menininha|padrinho madrinha|afilhado afilhada|genro nora|sogro sogra"""
M2F = {}
for pair in _PAIRS_TXT.replace("\n", "|").split("|"):
    a, b = pair.split()
    M2F[a] = b
F2M = {v: k for k, v in M2F.items()}
INVARIANT_NOUNS = {"estudante", "motorista", "artista", "cliente", "colega", "jovem", "chefe", "presidente", "turista",
                   "dentista", "jornalista", "pianista", "atleta", "líder", "intérprete", "agente", "comerciante",
                   "gerente", "paciente", "participante", "habitante", "adolescente", "cientista", "guitarrista",
                   "pessoa_"}  # 'pessoa_' é só sentinela
NAMES = {"joão": "maria", "pedro": "ana", "paulo": "paula", "carlos": "carla", "marcos": "marta", "antônio": "antônia",
         "luís": "luísa", "josé": "josefa", "lucas": "lúcia", "rafael": "rafaela", "gabriel": "gabriela", "daniel": "daniela"}
NAMES.update({v: k for k, v in list(NAMES.items())})
NAMES["maria"] = "joão"; NAMES["ana"] = "pedro"

DET_FLIP = {"o": "a", "os": "as", "um": "uma", "uns": "umas", "do": "da", "dos": "das", "no": "na", "nos": "nas",
            "ao": "à", "aos": "às", "pelo": "pela", "pelos": "pelas", "esse": "essa", "esses": "essas", "este": "esta",
            "estes": "estas", "aquele": "aquela", "aqueles": "aquelas", "dele": "dela", "deles": "delas", "ele": "ela",
            "eles": "elas", "num": "numa", "nuns": "numas", "dum": "duma", "duns": "dumas", "todo": "toda", "todos": "todas",
            "outro": "outra", "outros": "outras", "mesmo": "mesma", "mesmos": "mesmas", "algum": "alguma", "alguns": "algumas",
            "nenhum": "nenhuma", "dois": "duas", "muito": "muita", "muitos": "muitas", "pouco": "pouca", "poucos": "poucas",
            "certo": "certa", "certos": "certas", "seu": "sua", "seus": "suas", "meu": "minha", "meus": "minhas",
            "nosso": "nossa", "nossos": "nossas", "tal": "tal"}
DET_FLIP.update({v: k for k, v in list(DET_FLIP.items()) if v != k})
DET_FLIP["a"] = "o"; DET_FLIP["as"] = "os"; DET_FLIP["à"] = "ao"
PRON_ONLY = {"ele", "ela", "eles", "elas", "dele", "dela", "deles", "delas"}   # pronomes de 3ª pessoa (referem-se à pessoa)
POSSESSIVE = {"seu", "sua", "seus", "suas", "meu", "minha", "meus", "minhas", "nosso", "nossa", "nossos", "nossas"}

TEMPORAL = set("""hoje ontem amanhã agora depois antes sempre então ano anos mês meses dia dias semana semanas hora horas
manhã tarde noite século séculos época tempo vez vezes momento madrugada verão inverno primavera outono domingo segunda terça
quarta quinta sexta sábado janeiro fevereiro março abril maio junho julho agosto setembro outubro novembro dezembro
infância adolescência""".split())
NEGATIONS = {"não", "nunca", "jamais", "nem", "sem", "nenhum", "nenhuma"}
SKIP_VERBS = {"existir", "ser", "estar", "ter", "haver", "ir", "fazer", "ficar", "poder", "dever", "vir", "ver", "dar", "dizer",
              "parecer", "precisar", "querer", "saber", "chamar", "levar", "passar", "tornar", "deixar"}
SKIP_WORDS = {"muito", "mais", "menos", "bem", "mal", "já", "ainda", "também", "só", "apenas", "assim", "aqui", "ali",
              "lá", "onde", "como", "quando", "porque", "outro", "outra", "mesmo", "mesma", "próprio", "próprio"}


@dataclass
class MutationResult:
    text: str
    changes: list = field(default_factory=list)


def _cap_first(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _lower_first(tokens_text: str, first_tok) -> str:
    if first_tok.pos_ == "PROPN" or (len(first_tok.text) > 1 and first_tok.text.isupper()) or first_tok.ent_type_:
        return tokens_text
    return tokens_text[:1].lower() + tokens_text[1:]


def _txt(tokens) -> str:
    return "".join(t.text_with_ws for t in tokens).strip()


def _morph(t) -> dict:
    return {k: v.split(",")[0] for k, v in t.morph.to_dict().items()}


class Mutator:
    def __init__(self, nlp=None, lexicon: Lexicon | None = None):
        if nlp is None:
            import spacy
            nlp = spacy.load("pt_core_news_sm")
        self.nlp = nlp
        self.lex = lexicon or Lexicon()

    # ------------------------------------------------------------------ API
    def mutate(self, text: str, ops=("paraphrase", "gender", "word"), rate: float = 0.3, seed: int | None = None,
               word_mode: str = "mixed", direction: str = "random") -> MutationResult:
        """Aplica os operadores em sequência (paráfrase -> gênero -> palavras) e devolve texto + alterações."""
        rng = random.Random(seed)
        out_paras, changes = [], []
        for para in text.split("\n"):
            if not para.strip():
                out_paras.append(para); continue
            cur = para
            if "paraphrase" in ops:
                r = self.paraphrase(cur, rng, rate); cur = r.text; changes += r.changes
            if "gender" in ops:
                r = self.gender_swap(cur, rng, rate); cur = r.text; changes += r.changes
            if "word" in ops:
                r = self.word_swap(cur, rng, rate, word_mode, direction); cur = r.text; changes += r.changes
            out_paras.append(cur)
        return MutationResult("\n".join(out_paras), changes)

    # ------------------------------------------------------------------ 1) TROCA DE PALAVRAS
    def _inflect_candidate(self, cand: str, t) -> str | None:
        m, pos, L = _morph(t), t.pos_, self.lex
        if pos == "NOUN":
            g, n = m.get("Gender"), m.get("Number", "Sing")
            if g is None or L.form(cand, "NOUN", Gender=g, Number="Sing") is None:     # mesmo gênero (atestado no corpus)
                return None
            if n == "Sing":
                return cand
            return L.form(cand, "NOUN", Gender=g, Number="Plur") or morph.pluralize(cand)
        if pos == "ADJ":
            g, n = m.get("Gender", "Masc"), m.get("Number", "Sing")
            return L.form(cand, "ADJ", Gender=g, Number=n) or morph.inflect_adj(cand, g, n)
        if pos == "VERB":
            if m.get("VerbForm") == "Inf":
                return cand
            return L.form(cand, "VERB", **{k: v for k, v in m.items() if k in ("Gender", "Number", "Tense", "Mood", "Person", "VerbForm")})
        if pos == "ADV":
            return cand
        return None

    def word_swap(self, text: str, rng: random.Random, rate: float, mode: str = "mixed", direction: str = "random") -> MutationResult:
        """Troca palavras de conteúdo por sinônimos (e, no modo 'mixed'/'antonym', antônimos de adjetivos/advérbios).

        direction: 'random' | 'simpler' (escolhe a alternativa lexicalmente mais fácil) | 'harder'.
        """
        doc = self.nlp(text)
        repl, changes = {}, []
        for t in doc:
            if (not t.is_alpha or t.pos_ not in POS_MAP or t.ent_type_ or len(t.text) < 3
                    or t.lemma_.lower() in SKIP_VERBS or t.lower_ in SKIP_WORDS or t.dep_ in ("fixed", "flat:name")
                    or any(c.dep_ == "fixed" for c in t.children) or (t.text[0].isupper() and not t.is_sent_start)):
                continue
            if rng.random() > rate:
                continue
            lemma = t.lemma_.lower()
            ant = mode == "antonym" or (mode == "mixed" and t.pos_ in ("ADJ", "ADV") and rng.random() < 0.25)
            if ant:
                if any(x.lower_ in NEGATIONS for x in doc[max(0, t.i - 3):t.i]):
                    continue                                 # evita dupla negação
                cands = self.lex.antonyms(lemma, t.pos_)
            else:
                cands = self.lex.synonyms(lemma, t.pos_)
                if t.pos_ == "ADV":                          # advérbios: só -mente <-> -mente (o resto é muito ruidoso)
                    cands = [c for c in cands if c.endswith("mente") and lemma.endswith("mente")]
            if not ant and self.lex.vec is not None:        # desambiguação por contexto: o candidato tem que "caber" na frase
                ctx = [w.lower_ for w in doc[max(t.sent.start, t.i - 8):min(t.sent.end, t.i + 9)] if w.is_alpha and w.i != t.i]
                fo = self.lex.context_fit(lemma, ctx)
                if fo is not None:
                    keep = []
                    for c in cands:
                        fc = self.lex.context_fit(c, ctx)
                        if fc is not None and fc >= max(0.8 * fo, 0.05):
                            keep.append(c)
                    cands = keep
            cands = [c for c in cands if c not in SKIP_VERBS and c not in ("ser", "estar", "haver", "existir")]
            forms = []
            for c in cands[:12]:
                f = self._inflect_candidate(c, t)
                if f and f.lower() != t.lower_ and re.fullmatch(r"[a-zà-ÿç]+", f):
                    forms.append(f)
            if not forms:
                continue
            d0 = self.lex.difficulty(t.lower_)
            if direction == "simpler":
                f = min(forms, key=self.lex.difficulty)
                if self.lex.difficulty(f) >= d0 and not ant:
                    continue
            elif direction == "harder":
                f = max(forms, key=self.lex.difficulty)
                if self.lex.difficulty(f) <= d0 and not ant:
                    continue
            else:
                f = rng.choice(forms[:3])
            repl[t.i] = morph.match_case(t.text, f)
            changes.append({"op": "word", "kind": "antônimo" if ant else "sinônimo", "before": t.text, "after": repl[t.i]})
        out = "".join(repl.get(t.i, t.text) + t.whitespace_ for t in doc)
        return MutationResult(out, changes)

    # ------------------------------------------------------------------ 2) TROCA DE GÊNERO
    def _noun_pair(self, t):
        """Devolve (lema-alvo, gênero-alvo) se o substantivo for animado/trocável."""
        g = _morph(t).get("Gender")
        lem, low = t.lemma_.lower(), t.lower_
        for key in (lem, low):
            if key in M2F and g == "Masc":
                return M2F[key], "Fem"
            if key in F2M and g == "Fem":
                return F2M[key], "Masc"
            if key in INVARIANT_NOUNS and g in ("Masc", "Fem"):
                return key, ("Fem" if g == "Masc" else "Masc")
        return None

    def _adj_base(self, t):
        """O tagger pequeno marca muitos particípios/adjetivos como NOUN (e erra o lema). Reconhece-os pelo léxico:
        o masculino é atestado como ADJ no corpus e a própria forma NÃO é atestada como substantivo."""
        w = t.lower_
        n = "Plur" if w.endswith("s") else "Sing"
        base = w[:-1] if n == "Plur" else w
        g = "Fem" if base.endswith("a") else "Masc"
        if base.endswith("ora"):
            masc = base[:-1]
        elif base.endswith("esa"):
            masc = base[:-3] + "ês"
        elif g == "Fem":
            masc = base[:-1] + "o"
        else:
            masc = base
        if self.lex.form(masc, "ADJ", Gender="Masc", Number="Sing") and not self.lex.form(base, "NOUN", Gender=g, Number="Sing"):
            return masc
        return None

    def _flip(self, t, target: str, force_adj: bool = False) -> str | None:
        """Nova forma de um determinante/adjetivo/particípio/numeral para o gênero-alvo."""
        low = t.lower_
        m = _morph(t)
        if t.pos_ in ("DET", "ADP", "PRON", "NUM") or low in DET_FLIP:
            new = DET_FLIP.get(low)
            if new is None:
                return None
            return morph.match_case(t.text, new) if _form_gender(new) == target else None
        if t.pos_ == "ADJ" or force_adj:
            lem = t.lemma_.lower()
            n = m.get("Number") or ("Plur" if t.lower_.endswith("s") else "Sing")
            if force_adj and re.search(r"(ar|er|ir)$", lem):          # particípio com lema = infinitivo
                form = morph.participle(lem, target, n)
                return morph.match_case(t.text, form) if form else None
            if force_adj and (b := self._adj_base(t)):
                lem = b
            return morph.match_case(t.text, morph.inflect_adj(lem, target, n))
        if t.pos_ in ("VERB", "AUX") and m.get("VerbForm") == "Part":
            form = self.lex.form(t.lemma_.lower(), "VERB", Gender=target, Number=m.get("Number", "Sing"), VerbForm="Part") \
                or morph.participle(t.lemma_.lower(), target, m.get("Number", "Sing"))
            return morph.match_case(t.text, form) if form else None
        return None

    def gender_swap(self, text: str, rng: random.Random, rate: float) -> MutationResult:
        """Troca o gênero de frases inteiras: substantivos animados (lista de pares), pronomes ele/ela,
        e propaga a concordância (artigos, adjetivos, particípios, predicativos) pela árvore de dependências."""
        doc = self.nlp(text)
        repl, changes = {}, []
        for sent in doc.sents:
            if rng.random() > rate:
                continue
            local, targets = {}, []
            for t in sent:
                if t.pos_ == "NOUN" and (pr := self._noun_pair(t)):
                    if t.dep_ == "appos" and t.head.pos_ == "PROPN" and t.head.lower_ not in NAMES:
                        continue                               # 'Roberto Carlos, o cantor,': referente não trocável
                    if any(c.dep_ in ("appos", "flat:name") and c.pos_ == "PROPN" and c.lower_ not in NAMES for c in t.children):
                        continue                               # 'o cantor Roberto Carlos': nome próprio não trocável -> não troca
                    targets.append((t, pr))
                elif t.pos_ == "PRON" and t.lower_ in PRON_ONLY:
                    targets.append((t, None))
                elif t.pos_ == "PROPN" and t.lower_ in NAMES and not t.dep_.startswith("flat"):
                    targets.append((t, ("name", NAMES[t.lower_])))
            if not targets or all(pr is None for _, pr in targets):
                continue                                       # só pronome: o antecedente está em outra frase -> incoerente
            if any(t.ent_type_ == "PER" and t.lower_ not in NAMES for t in sent):
                continue                                       # nome de pessoa desconhecido: não dá para trocar junto
            if any(_cop_mismatch(t) for t in sent):
                continue                                       # 'Um tenor é um cantor': só um dos lados é trocável
            for t, pr in targets:
                if pr is None:                                 # pronome
                    new = DET_FLIP.get(t.lower_)
                    tgt = _form_gender(new)
                    local[t.i] = morph.match_case(t.text, new)
                elif pr[0] == "name":
                    tgt = _form_gender_name(pr[1])
                    local[t.i] = morph.match_case(t.text, pr[1])
                else:
                    lemma_new, tgt = pr
                    n = _morph(t).get("Number", "Sing")
                    if n == "Plur":
                        if lemma_new in ("mãe",) or t.lemma_.lower() == "pai":
                            continue                           # 'pais' é ambíguo
                        new = self.lex.form(lemma_new, "NOUN", Gender=tgt, Number="Plur") or morph.pluralize(lemma_new)
                    else:
                        new = lemma_new
                    local[t.i] = morph.match_case(t.text, new)
                # concordância
                agree = [c for c in t.children if c.dep_ in ("det", "amod", "case", "nummod", "acl", "det:poss")]
                agree += [cc for c in list(agree) if c.pos_ == "ADJ" for cc in c.children if cc.dep_ == "conj" and (cc.pos_ == "ADJ" or _morph(cc).get("VerbForm") == "Part")]
                h = t.head
                adjlike = set()
                if t.dep_ in ("nsubj", "nsubj:pass") and h.i != t.i:
                    def predicative(x):        # o tagger pequeno às vezes marca predicativos como NOUN; sem determinante = adjetivo
                        return (x.pos_ == "ADJ" or _morph(x).get("VerbForm") == "Part"
                                or (x.pos_ == "NOUN" and not any(k.dep_ == "det" for k in x.children)
                                    and x.lemma_.lower() not in M2F and x.lemma_.lower() not in F2M
                                    and any(k.dep_ == "cop" for k in h.children)))
                    stack, heads = [h], []
                    while stack:                                   # predicativo + todos os coordenados em cadeia
                        x = stack.pop()
                        if predicative(x):
                            heads.append(x); adjlike.add(x.i)
                        stack += [k for k in x.children if k.dep_ == "conj"]
                    agree += heads
                if h.pos_ == "VERB" and t.dep_ == "nsubj":         # predicativo secundário: 'Ela chegou cansada e feliz'
                    stack = [k for k in h.children if k.dep_ in ("obj", "xcomp", "amod", "obl") and k.pos_ in ("NOUN", "ADJ")
                             and not any(d.dep_ == "det" for d in k.children)]
                    while stack:
                        x = stack.pop()
                        if x.i not in adjlike and (x.pos_ == "ADJ" or self._adj_base(x)) and x.lemma_.lower() not in M2F | F2M:
                            agree.append(x); adjlike.add(x.i)
                            stack += [k for k in x.children if k.dep_ == "conj"]
                for c in agree:
                    if c.i in local or c.lower_ in POSSESSIVE:      # possessivo concorda com o possuído, não com o possuidor
                        continue
                    f = self._flip(c, tgt, force_adj=c.i in adjlike)
                    if f and f.lower() != c.lower_:
                        local[c.i] = f
            for i, new in local.items():
                if new != doc[i].text:
                    repl[i] = new
                    changes.append({"op": "gender", "before": doc[i].text, "after": new})
        out = "".join(repl.get(t.i, t.text) + t.whitespace_ for t in doc)
        return MutationResult(out, changes)

    # ------------------------------------------------------------------ 3) PARÁFRASE
    def paraphrase(self, text: str, rng: random.Random, rate: float) -> MutationResult:
        """Reescrita sintática (uma transformação por sentença, escolhida entre as aplicáveis):
        passiva->ativa, ativa->passiva, deslocamento de adjunto temporal, aposto, divisão e junção de orações."""
        doc = self.nlp(text)
        sents = list(doc.sents)
        out, changes, i = [], [], 0          # out: (texto, espaço-final-original) -> preserva o espaçamento do original
        while i < len(sents):
            s = sents[i]
            full = s.text_with_ws
            ws = full[len(full.rstrip()):]                      # todo o espaço final (inclui tokens de espaço duplo)
            sentence = full.strip()
            applied = None
            if rng.random() <= rate:
                ops = [self._passive_to_active, self._active_to_passive, self._move_adjunct, self._apposition,
                       self._split_clause]
                rng.shuffle(ops)
                for op in ops:
                    r = op(s)
                    if r and r != sentence:
                        applied = (op.__name__.strip("_"), r)
                        break
                if applied is None and i + 1 < len(sents):      # junção de duas frases curtas
                    r = self._merge(s, sents[i + 1])
                    if r:
                        f2 = sents[i + 1].text_with_ws
                        out.append((r, f2[len(f2.rstrip()):]))
                        changes.append({"op": "paraphrase", "kind": "junção de frases", "before": sentence + " " + sents[i + 1].text.strip(), "after": r})
                        i += 2
                        continue
            if applied:
                out.append((applied[1], ws))
                changes.append({"op": "paraphrase", "kind": _KIND[applied[0]], "before": sentence, "after": applied[1]})
            else:
                out.append((sentence, ws))
            i += 1
        return MutationResult("".join(t + w for t, w in out), changes)

    def _inf(self, tok) -> str | None:
        """Infinitivo do verbo; corrige lemas ruins de particípios irregulares ('escritos' -> 'escrever')."""
        lem = tok.lemma_.lower()
        if re.search(r"(ar|er|ir|ôr)$", lem):
            return lem
        return morph.participle_to_inf(tok.lower_, self.lex.freq)

    # -- passiva -> ativa:  "O texto foi lido pelo aluno." -> "O aluno leu o texto."
    def _passive_to_active(self, s):
        root = s.root
        if root.pos_ != "VERB" or _morph(root).get("VerbForm") != "Part":
            return None
        subj = [c for c in root.children if c.dep_ == "nsubj:pass"]
        aux = [c for c in root.children if c.dep_ == "aux:pass"]
        ag = [c for c in root.children if c.dep_ == "obl:agent"]
        if len(subj) != 1 or len(aux) != 1 or len(ag) != 1:
            return None
        S, X, A = subj[0], aux[0], ag[0]
        case = [c for c in A.children if c.dep_ == "case" and c.lower_ in ("por", "pelo", "pela", "pelos", "pelas")]
        if len(case) != 1:
            return None
        am = _morph(X)
        tense = {"Past": "Past", "Imp": "Imp", "Pres": "Pres", "Fut": "Fut"}.get(am.get("Tense"))
        if tense is None or am.get("Person", "3") != "3":
            return None
        Ss, As = sorted(S.subtree, key=lambda t: t.i), sorted(A.subtree, key=lambda t: t.i)
        if not (Ss[-1].i < X.i < root.i < As[0].i) or len(Ss) > 14 or len(As) > 14:
            return None
        plural = _morph(A).get("Number") == "Plur" or any(c.dep_ == "conj" for c in A.children)
        inf = self._inf(root)
        if not inf:
            return None
        verb = self.lex.form(inf, "VERB", Mood="Ind", Number="Plur" if plural else "Sing", Person="3",
                             Tense=tense, VerbForm="Fin") or morph.conjugate(inf, tense, "Plur" if plural else "Sing")
        if not verb:
            return None
        art = {"pelo": "o", "pela": "a", "pelos": "os", "pelas": "as"}
        ag_txt = "".join((art[t.lower_] + t.whitespace_) if t.i == case[0].i and t.lower_ in art else ("" if t.i == case[0].i else t.text_with_ws) for t in As).strip()
        prefix = s[:Ss[0].i - s.start]
        between = [t for t in s if root.i < t.i < As[0].i and t.pos_ != "PUNCT"]
        after = [t for t in s if t.i > As[-1].i and not (t.pos_ == "PUNCT" and t.i == s.end - 1)]
        end_punct = s[-1].text if s[-1].pos_ == "PUNCT" else "."
        subj_txt = _lower_first(_txt(Ss), Ss[0]) if not len(prefix) else _txt(Ss)
        first = ag_txt if not len(prefix) else ag_txt
        parts = ([_txt(prefix)] if len(prefix) else []) + [first, verb, subj_txt] + ([_txt(between)] if between else []) + ([_txt(after).rstrip(".")] if after else [])
        res = " ".join(p for p in parts if p).replace(" ,", ",")
        res = _cap_first(res) if not len(prefix) else res
        return re.sub(r"\s+([,;:.!?])", r"\1", res).rstrip(".!?…") + end_punct

    # -- ativa -> passiva:  "O aluno leu o texto." -> "O texto foi lido pelo aluno."
    def _active_to_passive(self, s):
        root = s.root
        rm = _morph(root)
        if root.pos_ != "VERB" or rm.get("VerbForm") != "Fin" or rm.get("Mood") != "Ind" or rm.get("Person") != "3":
            return None
        tense = rm.get("Tense")
        if tense not in ("Past", "Imp", "Pres", "Fut") or root.lemma_.lower() in SKIP_VERBS | {"haver", "ter"}:
            return None
        subj = [c for c in root.children if c.dep_ == "nsubj"]
        obj = [c for c in root.children if c.dep_ == "obj"]
        if len(subj) != 1 or len(obj) != 1 or any(c.dep_ in ("aux", "aux:pass", "expl", "expl:pass", "xcomp", "ccomp", "iobj") for c in root.children):
            return None
        S, O = subj[0], obj[0]
        ok_subj = S.pos_ in ("NOUN", "PROPN") or S.lower_ in ("ele", "ela", "eles", "elas")
        if not ok_subj or O.pos_ not in ("NOUN", "PROPN") or _morph(O).get("Gender") is None:
            return None
        Ss, Os = sorted(S.subtree, key=lambda t: t.i), sorted(O.subtree, key=lambda t: t.i)
        if not (Ss[-1].i + 1 == root.i and Os[0].i == root.i + 1) or len(Ss) > 12 or len(Os) > 12:
            return None
        if s[:Ss[0].i - s.start] and len(s[:Ss[0].i - s.start]) > 0 and Ss[0].i != s.start:
            return None
        on, og = _morph(O).get("Number", "Sing"), _morph(O).get("Gender")
        aux = morph.conjugate("ser", tense, "Plur" if on == "Plur" else "Sing")
        part = self.lex.form(root.lemma_.lower(), "VERB", Gender=og, Number=on, VerbForm="Part") or morph.participle(root.lemma_.lower(), og, on)
        if not aux or not part:
            return None
        first = Ss[0]
        contr = {"o": "pelo", "a": "pela", "os": "pelos", "as": "pelas"}
        if first.dep_ == "det" and first.lower_ in contr:
            agent = contr[first.lower_] + " " + _txt(Ss[1:])
        elif S.lower_ in ("ele", "ela", "eles", "elas"):
            agent = "por " + S.text
        else:
            agent = "por " + _txt(Ss)
        after = [t for t in s if t.i > Os[-1].i and not (t.pos_ == "PUNCT" and t.i == s.end - 1)]
        end_punct = s[-1].text if s[-1].pos_ == "PUNCT" else "."
        obj_txt = _cap_first(_txt(Os))
        res = f"{obj_txt} {aux} {part} {agent}" + ((" " + _txt(after).rstrip(".")) if after else "")
        return re.sub(r"\s+([,;:.!?])", r"\1", res) + end_punct

    # -- deslocamento de adjunto:  "Ontem, ele chegou." <-> "Ele chegou ontem."
    def _move_adjunct(self, s):
        text = s.text.strip()
        m = re.match(r"^(H[áa]|Faz)\s+((?:\w+\s+)?\w+)\s+(dias?|semanas?|meses|m[êe]s|anos?|horas?|minutos?)(\s+atr[áa]s)?\s*,?\s+(.+?)([.!?…])$", text)
        if m:                                                    # "Há duas semanas atrás estava chovendo."
            head = f"há {m.group(2)} {m.group(3)}{m.group(4) or ''}"
            rest = m.group(5)
            return _cap_first(rest) + " " + head + m.group(6)
        root = s.root
        if root.pos_ not in ("VERB", "ADJ", "NOUN", "AUX"):
            return None
        end_punct = s[-1].text if s[-1].pos_ == "PUNCT" else "."
        # (a) adjunto inicial seguido de vírgula -> final
        lead = [c for c in root.children if c.i < root.i and c.dep_ in ("advmod", "obl", "obl:tmod") and c.pos_ in ("ADV", "NOUN", "PROPN", "NUM")]
        for c in lead:
            sub = sorted(c.subtree, key=lambda t: t.i)
            if sub and sub[-1].text == ",":                      # o spaCy pendura a vírgula no adjunto
                sub = sub[:-1]
            if sub and sub[0].i == s.start and sub[-1].i + 1 < s.end and s.doc[sub[-1].i + 1].text == "," and len(sub) <= 8:
                rest = [t for t in s if t.i > sub[-1].i + 1 and not (t.pos_ == "PUNCT" and t.i == s.end - 1)]
                if not rest or rest[0].pos_ == "PUNCT":
                    continue
                adj = _lower_first(_txt(sub), sub[0])
                return _cap_first(_txt(rest)) + " " + adj + end_punct
        # (b) adjunto temporal final -> início
        trail = [c for c in root.children if c.i > root.i and c.dep_ in ("advmod", "obl", "obl:tmod")]
        for c in trail:
            sub = sorted(c.subtree, key=lambda t: t.i)
            last = s[-2] if s[-1].pos_ == "PUNCT" else s[-1]
            if sub[-1].i == last.i and len(sub) <= 6 and any(t.lower_ in TEMPORAL for t in sub) and sub[0].i > root.i + 0:
                rest = [t for t in s if t.i < sub[0].i]
                if len(rest) < 3:
                    continue
                restxt = _lower_first(_txt(rest), rest[0]).rstrip(",")
                return _cap_first(_txt(sub)) + ", " + restxt + end_punct
        return None

    # -- aposto:  "O cantor Roberto Carlos fez aniversário." <-> "Roberto Carlos, o cantor, fez aniversário."
    def _apposition(self, s):
        text = s.text.strip()
        m = re.match(r"^((?:[A-ZÀ-Ý][\wÀ-ÿ]+)(?:\s+(?:de|da|do|dos|das|[A-ZÀ-Ý][\wÀ-ÿ]+))*),\s+(o|a|os|as|um|uma)\s+([\wÀ-ÿ]+(?:\s+[a-zà-ÿ]+)?),\s+(.+)$", text)
        if m:
            return f"{_cap_first(m.group(2))} {m.group(3)} {m.group(1)} {m.group(4)}"
        for t in s:
            if t.pos_ == "NOUN" and any(c.dep_ == "appos" and c.pos_ == "PROPN" and c.i == t.i + 1 for c in t.children):
                det = [c for c in t.children if c.dep_ == "det" and c.i == t.i - 1 and c.lower_ in ("o", "a", "os", "as", "um", "uma")]
                if not det or t.dep_ not in ("nsubj", "nsubj:pass", "obj"):
                    continue
                ap = next(c for c in t.children if c.dep_ == "appos")
                name = sorted(ap.subtree, key=lambda x: x.i)
                if any(x.pos_ not in ("PROPN", "ADP", "X") for x in name):
                    continue
                start, end = det[0].i, name[-1].i
                before, after = s.doc[s.start:start], s.doc[end + 1:s.end]
                nxt = s.doc[end + 1].text if end + 1 < s.end else ""
                sep = "" if nxt in (",", ".", ";") else ","
                new = (_txt(before) + " " if len(before) else "") + _txt(name) + ", " + _lower_first(det[0].text + " " + t.text, det[0]) + sep + " " + _txt(after)
                new = re.sub(r"\s+([,;:.!?])", r"\1", new).strip()
                return _cap_first(new)
        return None

    # -- divisão de oração coordenada:  "X, mas Y." -> "X. Mas Y."
    def _split_clause(self, s):
        root = s.root
        if root.pos_ not in ("VERB", "AUX", "ADJ", "NOUN"):
            return None
        for c in root.children:
            if c.dep_ == "conj" and c.pos_ in ("VERB", "AUX") and any(x.dep_ == "nsubj" for x in c.children):
                cc = [x for x in c.children if x.dep_ == "cc" and x.i < c.i]
                if not cc:
                    continue
                cc = cc[0]
                left = [t for t in s if t.i < cc.i]
                if left and left[-1].text == ",":
                    left = left[:-1]
                right = [t for t in s if t.i > cc.i and not (t.pos_ == "PUNCT" and t.i == s.end - 1)]
                if len(left) < 5 or len(right) < 5:
                    continue
                end_punct = s[-1].text if s[-1].pos_ == "PUNCT" else "."
                head = (cc.text + " ") if cc.lower_ in ("mas", "porém", "contudo", "todavia", "pois", "porque") and cc.lower_ not in ("pois", "porque") else ""
                return _txt(left).rstrip(",;") + ". " + _cap_first(head + _lower_first(_txt(right), right[0])) + end_punct
        return None

    # -- junção de duas frases curtas:  "A. B." -> "A, e b."
    def _merge(self, s1, s2):
        n1, n2 = len([t for t in s1 if t.is_alpha]), len([t for t in s2 if t.is_alpha])
        if n1 > 9 or n2 > 9 or n1 < 3 or n2 < 3 or s2[0].pos_ == "PROPN":
            return None
        if s1[-1].text not in (".",) or s2.root.pos_ not in ("VERB", "AUX", "ADJ"):
            return None
        if not any(c.dep_ == "nsubj" for c in s2.root.children):
            return None
        a, b = _txt(s1[:-1]), _txt(s2)
        return a + ", e " + _lower_first(b, s2[0])


_KIND = {"passive_to_active": "passiva → ativa", "active_to_passive": "ativa → passiva", "move_adjunct": "deslocamento de adjunto",
         "apposition": "aposto", "split_clause": "divisão de oração"}


def _swappable(t) -> bool:
    l = t.lemma_.lower()
    return l in M2F or l in F2M or l in INVARIANT_NOUNS


def _cop_mismatch(t) -> bool:
    """Predicativo nominal (com cópula) e sujeito substantivo: ou os dois são trocáveis, ou nenhum."""
    if t.pos_ != "NOUN" or not any(k.dep_ == "cop" for k in t.children):
        return False
    subj = [k for k in t.children if k.dep_ == "nsubj" and k.pos_ == "NOUN"]
    return bool(subj) and _swappable(t) != _swappable(subj[0])


def _form_gender(word: str) -> str | None:
    fem = {"a", "as", "uma", "umas", "da", "das", "na", "nas", "à", "às", "pela", "pelas", "essa", "essas", "esta", "estas",
           "aquela", "aquelas", "dela", "delas", "ela", "elas", "numa", "numas", "duma", "dumas", "toda", "todas", "outra",
           "outras", "mesma", "mesmas", "alguma", "algumas", "nenhuma", "duas", "muita", "muitas", "pouca", "poucas", "certa",
           "certas", "sua", "suas", "minha", "minhas", "nossa", "nossas"}
    return "Fem" if word in fem else "Masc"


def _form_gender_name(name: str) -> str:
    return "Fem" if name.endswith("a") and name not in ("luca",) else "Masc"

