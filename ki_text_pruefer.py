#!/usr/bin/env python3
"""
Lokaler KI-Text-Prüfer v11 (Deutsch) — Tiefere Linguistik, Informationstheorie & lokale Perplexität

Verbesserungen in v5:
- Tiefere spaCy-Linguistik (mit Score-Anbindung):
  * Syntaktische Baumtiefe (Dependency Tree Depth & deren Streuung über Sätze)
  * Wortarten-Variabilität (POS-Entropie)
  * Named-Entity-Dichte (konkrete Personen, Orte, Organisationen)
  * Lexikalische Dichte (jetzt aktiv im Score gewichtet)
- Informationstheorie (immer ohne Zusatzmodell lokal verfügbar):
  * Wort-Entropie (Shannon-Entropie) & zlib-Komprimierungsrate (Redundanzmaß)
- Optionale lokale Perplexitäts-Messung (Transformers / PyTorch, z. B. dbmdz/german-gpt2):
  * Misst auf Wortebene, wie vorhersagbar der Text für ein lokales deutsches Sprachmodell ist
  * Läuft standardmäßig selektiv nur auf Grauzonen-Passagen (spart Rechenzeit)
- Alle Features aus v4 bleiben erhalten:
  * Punktgenaues Inline-Highlighting, Dark-Mode-sichere Farben, Hover-Erklärungen, PDF/TXT, Ollama-Option
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import threading
import urllib.error
import urllib.request
import zlib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    import spacy
except ImportError:
    spacy = None

_TRANSFORMERS_IMPORT_ERROR: Optional[str] = None
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    _TRANSFORMERS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Konfiguration: Sensitivität & deutsche Muster
# ---------------------------------------------------------------------------

CONFIG_PATH = Path.home() / ".ki_text_pruefer_config.json"


def load_user_config() -> dict:
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_user_config(updates: dict) -> None:
    try:
        cfg = load_user_config()
        cfg.update(updates)
        CONFIG_PATH.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass


def get_saved_ppl_model(default: str = "dbmdz/german-gpt2") -> Tuple[str, bool]:
    """
    Gibt (modell_oder_ordner, ist_gespeicherter_lokaler_ordner) zurück.
    """
    cfg = load_user_config()
    saved = str(cfg.get("ppl_model", "")).strip()
    if not saved:
        return default, False
    p = Path(saved).expanduser()
    if p.is_dir():
        return str(p), True
    return saved, False


SENSITIVITY_PROFILES = {
    "Konservativ": {
        "gray_threshold": 26.0,
        "strong_threshold": 56.0,
        "multiplier": 0.9,
        "desc": "Weniger Fehlalarme bei stark normierten Fachtexten",
    },
    "Ausgewogen": {
        "gray_threshold": 18.0,
        "strong_threshold": 42.0,
        "multiplier": 1.05,
        "desc": "Standard-Kalibrierung für allgemeine & akademische Texte",
    },
    "Streng": {
        "gray_threshold": 12.0,
        "strong_threshold": 32.0,
        "multiplier": 1.25,
        "desc": "Maximale Empfindlichkeit (erkennt auch subtilere KI-Muster)",
    },
}

GERMAN_AI_PHRASES = [
    r"\bes ist wichtig zu beachten,?\s+dass\b",
    r"\bes gilt zu beachten,?\s+dass\b",
    r"\bzusammenfassend lässt sich (sagen|festhalten|feststellen)\b",
    r"\babschließend lässt sich (sagen|festhalten|feststellen)\b",
    r"\bes gibt zahlreiche (autoren|studien|ansätze|faktoren|möglichkeiten|aspekte)\b",
    r"\bspielt eine (zentrale|entscheidende|schlüssel|maßgebliche|wesentliche|wichtige) ?rolle\b",
    r"\bstellt einen (entscheidenden|wichtigen|bedeutenden|zentralen) (wendepunkt|meilenstein|faktor|aspekt) dar\b",
    r"\bin der heutigen (schnelllebigen|digitalen|vernetzten|modernen|globalisierten) (welt|zeit|gesellschaft|landschaft|arbeitswelt)\b",
    r"\bvor diesem hintergrund\b",
    r"\bdarüber hinaus\b",
    r"\bdes weiteren\b",
    r"\beinerseits\b.{0,140}\bandererseits\b",
    r"\bsowohl\b.{1,80}\bals auch\b",
    r"\bchancen und (risiken|herausforderungen)\b",
    r"\bvorteile und nachteile\b",
    r"\bpotenziale und risiken\b",
    r"\bunterstreicht die (bedeutung|relevanz|wichtigkeit|notwendigkeit)\b",
    r"\bhebt die (relevanz|bedeutung|wichtigkeit) hervor\b",
    r"\bträgt maßgeblich dazu bei\b",
    r"\bträgt dazu bei,?\s+dass\b",
    r"\baufgrund (dieser|der genannten) (herausforderungen|entwicklungen|faktoren)\b",
    r"\btrotz dieser (herausforderungen|entwicklungen)\b",
    r"\bblick in die zukunft\b",
    r"\bals ki-sprachmodell\b",
    r"\bals großes sprachmodell\b",
    r"\bin einer zeit, in der\b",
    r"\bes bleibt abzuwarten, (wie|ob|inwieweit)\b",
    r"\bbietet (ein|eine|enormes|großes|vielfältiges) potenzial\b",
    r"\bermöglicht es,?\s+(prozesse|abläufe|daten|informationen)\b",
    r"\bvon (entscheidender|zentraler|großer|wesentlicher) bedeutung\b",
    r"\bim mittelpunkt (steht|stehen)\b",
    r"\bin den letzten jahren hat sich\b",
    r"\bim zuge der (digitalisierung|globalisierung|entwicklung)\b",
]

MODERN_AI_STEM_PATTERNS = [
    r"\bvielschichtig[a-zäöüß]{0,4}\b",
    r"\bganzheitlich[a-zäöüß]{0,4}\b",
    r"\bnahtlos[a-zäöüß]{0,4}\b",
    r"\bwegweisend[a-zäöüß]{0,4}\b",
    r"\bbahnbrechend[a-zäöüß]{0,4}\b",
    r"\btiefgreifend[a-zäöüß]{0,4}\b",
    r"\bfacettenreich[a-zäöüß]{0,4}\b",
    r"\bunverzichtbar[a-zäöüß]{0,4}\b",
    r"\bmaßgeblich[a-zäöüß]{0,4}\b",
    r"\bmaßgeschneidert[a-zäöüß]{0,4}\b",
    r"\bparadigmenwechsel[a-zäöüß]{0,3}\b",
    r"\bkatalysator[a-zäöüß]{0,3}\b",
    r"\b[a-zäöüß]{3,}landschaft\b",
    r"\bsynergie[a-zäöüß]{0,3}\b",
    r"\bumfassend[a-zäöüß]{0,4}\b",
    r"\bvielfältig[a-zäöüß]{0,4}\b",
    r"\beffizient[a-zäöüß]{0,4}\b",
    r"\binnovativ[a-zäöüß]{0,4}\b",
    r"\bnachhaltig[a-zäöüß]{0,4}\b",
    r"\bdynamisch[a-zäöüß]{0,4}\b",
    r"\bkomplex[a-zäöüß]{0,4}\b",
    r"\btransformier[a-zäöüß]{1,5}\b",
    r"\boptimier[a-zäöüß]{1,5}\b",
]

SENTENCE_STARTER_CONNECTORS = [
    r"^(darüber hinaus|zudem|des weiteren|ferner|außerdem|ebenso)\b",
    r"^(allerdings|dennoch|jedoch|gleichzeitig|hingegen)\b",
    r"^(insbesondere|vor allem|dabei|hierbei)\b",
    r"^(dies|diese|dieser|dieses|dadurch|somit|folglich|daher|deshalb)\b",
]

COPULA_REPLACEMENT_PATTERNS = [
    r"\bfungiert als\b",
    r"\bdient als\b",
    r"\bsteht als\b",
    r"\brepräsentiert\b",
    r"\bverkörpert\b",
    r"\bbildet das rückgrat\b",
    r"\bmarkiert einen (meilenstein|wendepunkt|übergang)\b",
    r"\bspiegelt .{1,45} wider\b",
    r"\bstellt .{1,35} dar\b",
]

SUPERFICIAL_CLAUSE_PATTERNS = [
    r",\s+was (die bedeutung|die relevanz|das potenzial|die wichtigkeit).{1,50}(unterstreicht|betont|verdeutlicht|hervorhebt)",
    r",\s+wodurch .{1,60}(gefördert|gestärkt|ermöglicht|vorangetrieben|verbessert|optimiert) wird",
    r",\s+und trägt (somit|damit) (dazu bei|maßgeblich)",
    r",\s+was wiederum\b",
]

CLOSER_SENTENCE_PATTERNS = [
    r"^(dies|das) zeigt, wie\b",
    r"^letztlich (zeigt sich|bleibt|ist|hängt)\b",
    r"^insgesamt (zeigt sich|lässt sich|bietet|stellt)\b",
    r"^damit wird deutlich\b",
    r"^somit (wird|zeigt sich|entsteht|lässt sich)\b",
    r"^kurzum[,:]",
    r"^abschließend\b",
]

NEGATIVE_PARALLELISM_PATTERNS = [
    r"\bnicht nur\b.{1,100}\bsondern auch\b",
    r"\bnicht\s+[^,.]{2,50},\s+sondern\s+[^,.]{2,50}",
    r"\bes geht nicht (nur )?darum\b.{1,80}\bsondern\b",
    r"\bweniger\s+[^,.]{2,40},\s+vielmehr\b",
]

RULE_OF_THREE_PATTERN = r"\b[A-ZÄÖÜa-zäöüß]{4,},\s+[a-zäöüß]{4,}\s+und\s+[a-zäöüß]{4,}\b"

LIST_STRUCT_PATTERN = r"(?m)^(?:\s*[-*•]\s+|\s*\d+[.)]\s+)(?:\*\*[^*\n]{2,40}\*\*|[A-ZÄÖÜ][a-zäöüßA-ZÄÖÜ\s-]{2,35}:)"

MODAL_PARTICLES = [
    "halt", "eben", "mal", "eigentlich", "irgendwie", "quasi", "übrigens",
    "nämlich", "sowieso", "eh", "wohl", "schließlich",
]

# Semantik: Deutsche Abschwächungen (Hedges)
GERMAN_HEDGES = [
    r"\bmöglicherweise\b",
    r"\btendenziell\b",
    r"\bgewissermaßen\b",
    r"\bin gewisser weise\b",
    r"\bscheinbar\b",
    r"\banscheinend\b",
    r"\bvermutlich\b",
    r"\bwahrscheinlich\b",
    r"\beventuell\b",
    r"\bwomöglich\b",
    r"\bunter umständen\b",
    r"\bim grunde\b",
    r"\bweitgehend\b",
    r"\bin der regel\b",
    r"\beinigermaßen\b",
    r"\bpotenziell\b",
    r"\bstreng genommen\b",
    r"\bsozusagen\b",
    r"\bteilweise\b",
    r"\bbisweilen\b",
]

# Emotion & Valenz: Geglättete positive KI-Tonalität vs. menschliche negative Ausschläge
GERMAN_POSITIVE_SMOOTH_WORDS = {
    "wertvoll", "wertvolle", "wertvollen", "wertvoller",
    "bereichernd", "bereichernde", "vielversprechend", "vielversprechende",
    "positiv", "positive", "positiven", "harmonisch", "harmonische",
    "konstruktiv", "konstruktive", "förderlich", "gewinnbringend",
    "bemerkenswert", "bemerkenswerte", "zielführend", "zielführende",
    "hilfreich", "hilfreiche", "vorteilhaft", "vorteilhafte",
    "optimistisch", "optimistische", "erfolgreich", "erfolgreiche",
    "beeindruckend", "beeindruckende", "faszinierend", "faszinierende",
    "inspirierend", "inspirierende", "stimmig", "stimmige",
}

GERMAN_NEGATIVE_STRONG_WORDS = {
    "frustrierend", "frustrierende", "ärgerlich", "ärgerliche",
    "katastrophal", "katastrophale", "absurd", "absurde",
    "lächerlich", "lächerliche", "nervig", "nervige", "nervt",
    "mühsam", "mühsame", "enttäuschend", "enttäuschende",
    "chaotisch", "chaotische", "unsinnig", "unsinnige",
    "miserabel", "miserable", "furchtbar", "furchtbare",
    "peinlich", "peinliche", "schrecklich", "schreckliche",
    "fatal", "fatale", "ärgern", "gescheitert", "desaster",
    "quatsch", "blödsinn", "unbrauchbar", "unzumutbar",
}

# Psycholinguistik: Konkrete vs. abstrakte Marker
CONCRETE_OBJECT_WORDS = {
    "haus", "tisch", "stuhl", "buch", "straße", "baum", "wasser", "auto",
    "zug", "bahnhof", "tür", "fenster", "papier", "stift", "bildschirm",
    "tastatur", "kaffee", "hand", "auge", "stadt", "dorf", "schule",
    "jahr", "monat", "tag", "stunde", "minute", "euro", "prozent",
}

CHATBOT_TECH_PATTERNS = [
    ("Markdown-Überschrift", r"(?m)^#{1,6}\s+\S+"),
    ("Markdown-Fettdruck-Reste", r"\*\*[^*\n]{2,60}\*\*"),
    ("Markdown-Codeblock", r"```"),
    ("Datums-Platzhalter", r"\b20\d{2}-[Xx]{2}-[Xx]{2}\b"),
    ("URL-Platzhalter", r"\b(INSERT_SOURCE_URL|INSERT_WEBSITE_URL|PASTE_[A-Z_]+_URL_HERE)\b"),
    ("KI-UTM-Parameter", r"(utm_source=(chatgpt\.com|openai|copilot\.com)|referrer=grok\.com)"),
]

AUSTRIACISMS_HELVETISMS = [
    "weiters", "allfällig", "heuer", "jänner", "feber", "matura",
    "spital", "allenfalls", "innert", "entsprechendes", "vorerst",
    "velo", "trottoir", "natel", "zügel", "obgenannt",
]

# Cache für lokales Perplexitäts-Modell (damit es nur einmal geladen wird)
_PPL_MODEL_CACHE: Dict[str, Tuple[object, object]] = {}


# ---------------------------------------------------------------------------
# Datenstrukturen
# ---------------------------------------------------------------------------

@dataclass
class TextSpan:
    start: int
    end: int
    category: str   # "tech" | "phrase" | "structure" | "buzzword" | "human"
    priority: int   # Höher überschreibt bei Überlappung
    explanation: str


@dataclass
class PassageResult:
    index: int
    text: str
    word_count: int
    sentence_count: int
    score: float
    label: str  # "Unauffällig" | "Grauzone" | "Stark KI-verdächtig"
    hard_tell: bool
    reasons: List[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    spans: List[TextSpan] = field(default_factory=list)
    ollama_note: Optional[str] = None
    perplexity: Optional[float] = None


@dataclass
class DocumentReport:
    source_name: str
    sensitivity: str
    total_words: int
    total_passages: int
    ai_share_percent: float
    strong_ai_percent: float
    gray_zone_percent: float
    human_percent: float
    spacy_model_used: str
    ppl_used: str
    ollama_used: str
    document_notes: List[str]
    passages: List[PassageResult]


# ---------------------------------------------------------------------------
# Hilfsfunktionen: NLP, Syntaxbaum, Informationstheorie, ISBN, PDF
# ---------------------------------------------------------------------------

def load_german_spacy():
    if spacy is None:
        return None, "Kein spaCy (Regex-Fallback aktiv)"
    for model_name in ("de_core_news_lg", "de_core_news_md", "de_core_news_sm"):
        try:
            nlp = spacy.load(model_name)
            return nlp, model_name
        except Exception:
            continue
    return None, "spaCy ohne deutsches Modell (Regex-Fallback aktiv)"


def extract_text_from_pdf(pdf_path: str) -> str:
    if PdfReader is None:
        raise RuntimeError(
            "Das Paket 'pypdf' ist nicht installiert. Bitte installiere es mit: pip install pypdf"
        )
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        pages.append(txt)
    raw = "\n\n".join(pages)
    raw = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)
    raw = re.sub(r"(?<![.!?:;])\n(?!\n)", " ", raw)
    raw = re.sub(r"[ \t]{2,}", " ", raw)
    return raw


def is_valid_isbn10(isbn: str) -> bool:
    digits = re.sub(r"[^0-9Xx]", "", isbn)
    if len(digits) != 10:
        return False
    total = 0
    for i, ch in enumerate(digits):
        val = 10 if (i == 9 and ch in "Xx") else (int(ch) if ch.isdigit() else -1)
        if val < 0:
            return False
        total += (10 - i) * val
    return total % 11 == 0


def is_valid_isbn13(isbn: str) -> bool:
    digits = re.sub(r"[^0-9]", "", isbn)
    if len(digits) != 13:
        return False
    total = sum((1 if i % 2 == 0 else 3) * int(d) for i, d in enumerate(digits))
    return total % 10 == 0


def check_isbns_and_dois(text: str) -> Tuple[List[Tuple[str, int, int]], List[str]]:
    invalid_isbns = []
    for match in re.finditer(r"ISBN(?:-1[03])?:?\s*([0-9Xx\- ]{10,20})", text):
        raw_isbn = match.group(1).strip()
        cleaned = re.sub(r"[^0-9Xx]", "", raw_isbn)
        if len(cleaned) == 10 and not is_valid_isbn10(cleaned):
            invalid_isbns.append((raw_isbn, match.start(), match.end()))
        elif len(cleaned) == 13 and not is_valid_isbn13(cleaned):
            invalid_isbns.append((raw_isbn, match.start(), match.end()))

    dois = re.findall(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", text)
    return invalid_isbns, dois


def _token_depth(token) -> int:
    depth = 0
    cur = token
    while cur.head != cur and depth < 50:
        cur = cur.head
        depth += 1
    return depth


def _shannon_entropy(counts: Counter) -> Optional[float]:
    total = sum(counts.values())
    if total <= 0:
        return None
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def _compute_optional_synsets(alpha_tokens) -> Optional[float]:
    """
    Prüft optional, ob lokal über NLTK (omw-1.4 / wordnet mit Sprache 'deu')
    Synsets verfügbar sind, und berechnet die mittlere Bedeutungsvielfalt
    pro Inhaltswort (NOUN/VERB/ADJ/ADV).
    """
    try:
        from nltk.corpus import wordnet as wn
        # Schneller Check, ob deutsches WordNet lokal vorhanden ist
        if "deu" not in wn.langs():
            return None
        pos_map = {"NOUN": wn.NOUN, "VERB": wn.VERB, "ADJ": wn.ADJ, "ADV": wn.ADV}
        syn_counts = []
        for t in alpha_tokens[:120]:
            wn_pos = pos_map.get(getattr(t, "pos_", ""))
            if wn_pos is None:
                continue
            lemma = getattr(t, "lemma_", None) or t.text
            syns = wn.synsets(lemma.lower(), pos=wn_pos, lang="deu")
            if not syns:
                syns = wn.synsets(t.text.lower(), pos=wn_pos, lang="deu")
            if syns:
                syn_counts.append(len(syns))
        return statistics.mean(syn_counts) if syn_counts else None
    except Exception:
        return None


def analyze_deep_linguistics(text: str, nlp=None) -> dict:
    """
    Liefert Wörter, Sätze sowie (wenn ein deutsches spaCy-Modell aktiv ist)
    erweiterte linguistische Merkmale nach der Stuttgarter/GESIS-Studie:
    - lex_density: Anteil Inhaltswörter (NOUN, VERB, ADJ, ADV)
    - pos_entropy: Shannon-Entropie der Wortarten-Verteilung
    - avg_tree_depth / tree_depth_cv: mittlere Syntaxbaum-Tiefe & Streuung
    - avg_tree_width / tree_width_cv: mittlere Baumbreite (max. Tokens pro Ebene) & Streuung
    - avg_branching_factor: mittlerer Verzweigungsfaktor (Kinder pro Nicht-Blatt-Token)
    - noun_chunk_density: Nominalphrasen pro 100 Wörter
    - dep_entropy: Shannon-Entropie der Universal-Dependencies-/Syntax-Relationen
    - morph_entropy: Shannon-Entropie der morphologischen Merkmalsbündel
    - entity_density: Dichte benannter Entitäten (Personen, Orte, Organisationen)
    - avg_synsets: mittlere WordNet-Synset-Anzahl pro Inhaltswort (falls lokal verfügbar)
    """
    if nlp is not None:
        doc = nlp(text)
        alpha_tokens = [t for t in doc if t.is_alpha]
        words = [t.text.lower() for t in alpha_tokens]
        sentences = []
        tree_depths = []
        tree_widths = []
        branching_counts = []

        for sent in doc.sents:
            s_words = [t.text for t in sent if t.is_alpha]
            if not s_words:
                continue
            sentences.append(s_words)

            level_counts: Counter = Counter()
            sent_depths = []
            for t in sent:
                if t.is_space:
                    continue
                d = _token_depth(t)
                sent_depths.append(d)
                level_counts[d] += 1
                n_children = sum(1 for c in t.children if not c.is_space)
                if n_children > 0:
                    branching_counts.append(n_children)

            if sent_depths:
                tree_depths.append(max(sent_depths))
            if level_counts:
                tree_widths.append(max(level_counts.values()))

        lex_density = None
        pos_entropy = None
        dep_entropy = None
        morph_entropy = None
        if alpha_tokens:
            lexical_pos = {"NOUN", "VERB", "ADJ", "ADV"}
            lex_count = sum(1 for t in alpha_tokens if t.pos_ in lexical_pos)
            lex_density = lex_count / len(alpha_tokens)

            pos_entropy = _shannon_entropy(Counter(t.pos_ for t in alpha_tokens))
            dep_entropy = _shannon_entropy(Counter(t.dep_ for t in alpha_tokens if getattr(t, "dep_", "")))
            morph_strs = [str(t.morph) for t in alpha_tokens if str(getattr(t, "morph", ""))]
            if morph_strs:
                morph_entropy = _shannon_entropy(Counter(morph_strs))

        avg_tree_depth = statistics.mean(tree_depths) if tree_depths else None
        tree_depth_cv = None
        if len(tree_depths) >= 3 and avg_tree_depth and avg_tree_depth > 0:
            tree_depth_cv = statistics.stdev(tree_depths) / avg_tree_depth

        avg_tree_width = statistics.mean(tree_widths) if tree_widths else None
        tree_width_cv = None
        if len(tree_widths) >= 3 and avg_tree_width and avg_tree_width > 0:
            tree_width_cv = statistics.stdev(tree_widths) / avg_tree_width

        avg_branching_factor = statistics.mean(branching_counts) if branching_counts else None

        try:
            n_chunks = sum(1 for _ in doc.noun_chunks)
            noun_chunk_density = (100.0 * n_chunks / len(words)) if words else None
        except Exception:
            noun_chunk_density = None

        entity_density = (len(doc.ents) / len(words)) if words else None
        avg_synsets = _compute_optional_synsets(alpha_tokens)

        return {
            "words": words,
            "sentences": sentences,
            "lex_density": lex_density,
            "pos_entropy": pos_entropy,
            "avg_tree_depth": avg_tree_depth,
            "tree_depth_cv": tree_depth_cv,
            "avg_tree_width": avg_tree_width,
            "tree_width_cv": tree_width_cv,
            "avg_branching_factor": avg_branching_factor,
            "noun_chunk_density": noun_chunk_density,
            "dep_entropy": dep_entropy,
            "morph_entropy": morph_entropy,
            "entity_density": entity_density,
            "avg_synsets": avg_synsets,
        }

    # Regex-Fallback ohne spaCy-Modell
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9„\"])", text.strip())
    sentences = []
    all_words = []
    for s in raw_sentences:
        w = re.findall(r"\b[A-Za-zÄÖÜäöüß]+\b", s)
        if w:
            sentences.append(w)
            all_words.extend(x.lower() for x in w)

    return {
        "words": all_words,
        "sentences": sentences,
        "lex_density": None,
        "pos_entropy": None,
        "avg_tree_depth": None,
        "tree_depth_cv": None,
        "avg_tree_width": None,
        "tree_width_cv": None,
        "avg_branching_factor": None,
        "noun_chunk_density": None,
        "dep_entropy": None,
        "morph_entropy": None,
        "entity_density": None,
        "avg_synsets": None,
    }


def compute_information_metrics(text: str, words: List[str]) -> Tuple[float, float]:
    """
    Berechnet ohne externes Modell:
    - word_entropy: Shannon-Entropie der Wortverteilung
    - compression_ratio: Verhältnis komprimierter Bytes zu Roh-Bytes (zlib)
    """
    word_entropy = 0.0
    if words:
        counts = Counter(words)
        n = len(words)
        word_entropy = -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)

    raw_bytes = text.encode("utf-8")
    comp_ratio = 1.0
    if len(raw_bytes) > 0:
        comp_ratio = len(zlib.compress(raw_bytes, level=6)) / len(raw_bytes)
    return word_entropy, comp_ratio


def compute_valence_and_psycholinguistics(text: str, words: List[str], entity_density: Optional[float]) -> dict:
    """
    Berechnet sprachübergreifend inspirierte (auf Deutsch kalibrierte)
    Valenz-, Emotions- und Psycholinguistik-Kennzahlen:
    - valence_mean / valence_std: Mittelwert & Streuung der Polarität (-1 bis +1)
    - smooth_pos_count / neg_spike_count: geglättete positive vs. stark negative Wörter
    - concreteness_score: Verhältnis konkreter Marker (Zahlen, Entitäten, Konkreta) vs. Abstrakta
    - avg_word_len / long_word_ratio: Näherung für Wortprävalenz / Komposita-Lastigkeit
    """
    if not words:
        return {
            "valence_mean": 0.0,
            "valence_std": 0.0,
            "smooth_pos_count": 0,
            "neg_spike_count": 0,
            "concreteness_score": 0.5,
            "avg_word_len": 0.0,
            "long_word_ratio": 0.0,
        }

    polarities = []
    smooth_pos = 0
    neg_spike = 0
    for w in words:
        if w in GERMAN_POSITIVE_SMOOTH_WORDS:
            polarities.append(0.75)
            smooth_pos += 1
        elif w in GERMAN_NEGATIVE_STRONG_WORDS:
            polarities.append(-1.0)
            neg_spike += 1
        else:
            polarities.append(0.0)

    valence_mean = statistics.mean(polarities)
    valence_std = statistics.stdev(polarities) if len(polarities) >= 2 else 0.0

    # Konkretheit: Zahlen/Jahreszahlen + konkrete Substantive + Named Entities vs. abstrakte Suffixe
    num_hits = len(re.findall(r"\b\d+(?:[.,]\d+)?\b", text))
    concrete_hits = sum(1 for w in words if w in CONCRETE_OBJECT_WORDS)
    abstract_hits = sum(
        1 for w in words
        if len(w) >= 7 and re.search(r"(ung|ungen|heit|heiten|keit|keiten|schaft|schaften|tion|tionen|ität|itäten|ismus|anz|enz)$", w)
    )
    ent_bonus = (entity_density * len(words)) if entity_density is not None else 0.0
    concrete_signal = concrete_hits + 0.6 * num_hits + 1.2 * ent_bonus
    total_ca = concrete_signal + abstract_hits
    concreteness_score = (concrete_signal / total_ca) if total_ca > 0 else 0.5

    avg_word_len = statistics.mean(len(w) for w in words)
    long_word_ratio = sum(1 for w in words if len(w) >= 13) / len(words)

    return {
        "valence_mean": valence_mean,
        "valence_std": valence_std,
        "smooth_pos_count": smooth_pos,
        "neg_spike_count": neg_spike,
        "concreteness_score": concreteness_score,
        "avg_word_len": avg_word_len,
        "long_word_ratio": long_word_ratio,
    }


def get_perplexity_device():
    if torch is None:
        return "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_perplexity_model(model_name: str = "dbmdz/german-gpt2"):
    """Lädt und cached Tokenizer, Modell, Zielgerät (MPS/CPU) und Kontextlimit."""
    if torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        detail = f" ({_TRANSFORMERS_IMPORT_ERROR})" if _TRANSFORMERS_IMPORT_ERROR else ""
        raise RuntimeError(
            "Für die lokale Perplexitäts-Prüfung werden 'torch' und 'transformers' "
            f"in der aktiven Python-Umgebung benötigt{detail}."
        )

    expanded_path = Path(model_name).expanduser()
    is_local_dir = expanded_path.is_dir()
    resolved_name = str(expanded_path.resolve()) if is_local_dir else model_name

    if is_local_dir and not (expanded_path / "config.json").exists():
        raise FileNotFoundError(
            f"Im gewählten Ordner '{resolved_name}' fehlt 'config.json'. "
            "Bitte alle Dateien von dbmdz/german-gpt2 (inkl. config.json, model.safetensors, tokenizer.json) dort ablegen."
        )

    if resolved_name not in _PPL_MODEL_CACHE:
        load_kwargs = {"local_files_only": True} if is_local_dir else {}
        tokenizer = AutoTokenizer.from_pretrained(resolved_name, **load_kwargs)
        model = AutoModelForCausalLM.from_pretrained(resolved_name, **load_kwargs)
        device = get_perplexity_device()
        model.to(device)
        model.eval()
        max_pos = (
            getattr(model.config, "n_positions", None)
            or getattr(model.config, "max_position_embeddings", None)
            or 512
        )
        _PPL_MODEL_CACHE[resolved_name] = (tokenizer, model, device, int(max_pos))
    return _PPL_MODEL_CACHE[resolved_name]


def compute_local_perplexity(text: str, model_name: str = "dbmdz/german-gpt2") -> float:
    """
    Berechnet die Perplexität eines deutschen Textblocks mit einem lokalen
    Causal-LM (Standard: dbmdz/german-gpt2) über PyTorch & Transformers.
    Nutzt ein Sliding Window passend zum Modell-Kontextlimit (z. B. 512 Tokens
    bei dbmdz/german-gpt2) sowie Apple Silicon MPS, falls verfügbar.
    """
    tokenizer, model, device, max_len = load_perplexity_model(model_name)

    # Tokenizer-Längenwarnung beim Vorab-Tokenisieren unterdrücken, da das Sliding Window
    # unten jeden Chunk strikt auf max_len (z. B. 512 Tokens) begrenzt.
    orig_max_len = getattr(tokenizer, "model_max_length", max_len)
    try:
        tokenizer.model_max_length = 10_000_000
        enc = tokenizer(text, return_tensors="pt", truncation=False)
    finally:
        tokenizer.model_max_length = orig_max_len
    input_ids = enc["input_ids"]
    seq_len = int(input_ids.size(1))
    if seq_len < 8:
        raise ValueError("Textblock zu kurz für Perplexitäts-Berechnung (< 8 Tokens).")

    stride = max(64, max_len // 2)
    nll_sum = 0.0
    total_tokens = 0
    prev_end_loc = 0

    with torch.no_grad():
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_len, seq_len)
            trg_len = end_loc - prev_end_loc
            chunk_ids = input_ids[:, begin_loc:end_loc].to(device)
            target_ids = chunk_ids.clone()
            if trg_len < chunk_ids.size(1):
                target_ids[:, :-trg_len] = -100

            outputs = model(chunk_ids, labels=target_ids)
            # outputs.loss ist der Mittelwert über alle nicht-maskierten Ziel-Tokens (verschoben um 1)
            valid_targets = max(1, trg_len - (1 if begin_loc == 0 else 0))
            nll_sum += float(outputs.loss.item()) * valid_targets
            total_tokens += valid_targets

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    avg_nll = nll_sum / max(1, total_tokens)
    return math.exp(min(20.0, avg_nll))


def resolve_spans(spans: List[TextSpan]) -> List[TextSpan]:
    if not spans:
        return []
    ordered = sorted(spans, key=lambda s: (-s.priority, s.start, -(s.end - s.start)))
    accepted: List[TextSpan] = []
    for cand in ordered:
        if cand.start >= cand.end:
            continue
        overlaps = any(not (cand.end <= a.start or cand.start >= a.end) for a in accepted)
        if not overlaps:
            accepted.append(cand)
    return sorted(accepted, key=lambda s: s.start)


# ---------------------------------------------------------------------------
# Stufe 0: Intelligente Block-Bündelung (Segmentierung)
# ---------------------------------------------------------------------------

def _split_long_paragraph(para: str, target_words: int = 190, max_words: int = 250) -> List[str]:
    """Teilt sehr lange Absätze (typisch bei PDFs ohne Leerzeilen) an Satzgrenzen auf."""
    words_total = len(re.findall(r"\b[A-Za-zÄÖÜäöüß]+\b", para))
    if words_total <= max_words:
        return [para]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9„\"(\[])", para) if s.strip()]
    if len(sentences) <= 1:
        # Fallback für Endlos-Text ohne Satzzeichen
        raw_tokens = para.split()
        chunks = []
        for i in range(0, len(raw_tokens), target_words):
            chunks.append(" ".join(raw_tokens[i : i + target_words]))
        return chunks

    chunks: List[str] = []
    cur_sents: List[str] = []
    cur_words = 0

    for sent in sentences:
        swc = len(re.findall(r"\b[A-Za-zÄÖÜäöüß]+\b", sent))
        if cur_sents and (cur_words + swc > max_words or cur_words >= target_words):
            chunks.append(" ".join(cur_sents))
            cur_sents = [sent]
            cur_words = swc
        else:
            cur_sents.append(sent)
            cur_words += swc

    if cur_sents:
        if chunks and cur_words < 45:
            chunks[-1] = chunks[-1] + " " + " ".join(cur_sents)
        else:
            chunks.append(" ".join(cur_sents))

    return chunks


def segment_text(text: str, min_block_words: int = 110, max_block_words: int = 250) -> List[str]:
    raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    if not raw_paragraphs:
        return []

    # Überlange Blöcke (z. B. aus PDFs ohne Leerzeilen) zuerst an Satzgrenzen aufteilen
    normalized_units: List[str] = []
    for para in raw_paragraphs:
        normalized_units.extend(_split_long_paragraph(para, target_words=190, max_words=max_block_words))

    blocks: List[str] = []
    buffer: List[str] = []
    buffer_words = 0

    for unit in normalized_units:
        wc = len(re.findall(r"\b[A-Za-zÄÖÜäöüß]+\b", unit))
        if buffer and (buffer_words + wc > max_block_words):
            blocks.append("\n\n".join(buffer))
            buffer = [unit]
            buffer_words = wc
        else:
            buffer.append(unit)
            buffer_words += wc

        if buffer_words >= min_block_words:
            blocks.append("\n\n".join(buffer))
            buffer = []
            buffer_words = 0

    if buffer:
        if blocks and buffer_words < 45:
            blocks[-1] = blocks[-1] + "\n\n" + "\n\n".join(buffer)
        else:
            blocks.append("\n\n".join(buffer))

    return blocks


# ---------------------------------------------------------------------------
# Stufe 1: Analyse pro Passage inkl. tiefer Linguistik & Inline-Spans
# ---------------------------------------------------------------------------

def analyze_passage(
    index: int,
    text: str,
    nlp=None,
    sensitivity: str = "Ausgewogen",
) -> PassageResult:
    profile = SENSITIVITY_PROFILES.get(sensitivity, SENSITIVITY_PROFILES["Ausgewogen"])
    mult = profile["multiplier"]

    score = 0.0
    hard_tell = False
    reasons: List[str] = []
    raw_spans: List[TextSpan] = []
    lower_text = text.lower()

    # --- Modul A: Technische & Zitations-Checks ---
    for name, pattern in CHATBOT_TECH_PATTERNS:
        matches = list(re.finditer(pattern, text))
        if matches:
            hard_tell = True
            score += 42.0
            reasons.append(f"[Technik] {name} erkannt ({len(matches)}x)")
            for m in matches:
                raw_spans.append(
                    TextSpan(m.start(), m.end(), "tech", 100, f"[Technik] {name}: {m.group(0)[:60]}")
                )

    invalid_isbns, dois = check_isbns_and_dois(text)
    if invalid_isbns:
        hard_tell = True
        score += 45.0
        reasons.append(f"[Zitation] Ungültige ISBN-Prüfsumme: {', '.join(x[0] for x in invalid_isbns)}")
        for raw_isbn, s, e in invalid_isbns:
            raw_spans.append(
                TextSpan(s, e, "tech", 100, f"[Zitation] Ungültige ISBN-Prüfsumme: {raw_isbn}")
            )
    if dois:
        reasons.append(f"[Hinweis] {len(dois)} DOI(s) gefunden — Existenz/Themenbezug manuell prüfen")

    # --- Modul B: Deutsche Stil-, Struktur- & Floskel-Checks ---
    ai_phrase_hits = []
    for pat in GERMAN_AI_PHRASES:
        for m in re.finditer(pat, lower_text):
            ai_phrase_hits.append(m.group(0))
            raw_spans.append(
                TextSpan(m.start(), m.end(), "phrase", 80, f"[KI-Floskel] „{text[m.start():m.end()]}“")
            )
    if ai_phrase_hits:
        score += min(34.0, len(ai_phrase_hits) * 11.0 * mult)
        unique_hits = list(dict.fromkeys(ai_phrase_hits))[:4]
        reasons.append(f"[Stil] Typische KI-Floskeln: {', '.join(repr(h) for h in unique_hits)}")

    stem_hits = []
    for pat in MODERN_AI_STEM_PATTERNS:
        for m in re.finditer(pat, lower_text):
            stem_hits.append(m.group(0))
            raw_spans.append(
                TextSpan(m.start(), m.end(), "buzzword", 45, f"[KI-Modewort] „{text[m.start():m.end()]}“")
            )
    if stem_hits:
        if len(stem_hits) == 1:
            score += 4.5 * mult
        else:
            score += min(24.0, len(stem_hits) * 5.0 * mult)
        unique_buzz = list(dict.fromkeys(stem_hits))[:5]
        reasons.append(f"[Wortschatz] KI-typisches Vokabular ({len(stem_hits)}x): {', '.join(unique_buzz)}")

    copula_hits = []
    for pat in COPULA_REPLACEMENT_PATTERNS:
        for m in re.finditer(pat, lower_text):
            copula_hits.append(m.group(0))
            raw_spans.append(
                TextSpan(m.start(), m.end(), "phrase", 70, f"[Kopula-Ersatzverb] „{text[m.start():m.end()]}“")
            )
    if copula_hits:
        score += min(18.0, len(copula_hits) * 7.5 * mult)
        reasons.append(f"[Stil] Aufwertende Kopula-Ersatzverben: {', '.join(repr(h) for h in copula_hits[:3])}")

    sup_hits = 0
    for pat in SUPERFICIAL_CLAUSE_PATTERNS:
        for m in re.finditer(pat, lower_text):
            sup_hits += 1
            raw_spans.append(
                TextSpan(m.start(), m.end(), "phrase", 75, "[Oberflächlicher Nebensatz-Anhang]")
            )
    if sup_hits > 0:
        score += min(22.0, sup_hits * 11.0 * mult)
        reasons.append(f"[Stil] Oberflächliche Nebensatz-Anhänge („…was die Bedeutung unterstreicht“): {sup_hits}x")

    closer_hits = 0
    for para_m in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text):
        para_text = para_m.group(0)
        para_offset = para_m.start()
        sent_matches = list(re.finditer(r"[^.!?]+[.!?]?", para_text))
        if sent_matches:
            last_m = sent_matches[-1]
            last_sent_lower = last_m.group(0).strip().lower()
            if any(re.search(pat, last_sent_lower) for pat in CLOSER_SENTENCE_PATTERNS):
                closer_hits += 1
                s_abs = para_offset + last_m.start()
                e_abs = para_offset + last_m.end()
                while s_abs < e_abs and text[s_abs].isspace():
                    s_abs += 1
                raw_spans.append(
                    TextSpan(s_abs, e_abs, "structure", 65, "[Closer-Schlusssatz] Schablonenhafter Absatz-Abschluss")
                )
    if closer_hits > 0:
        score += min(18.0, closer_hits * 9.0 * mult)
        reasons.append(f"[Struktur] Schablonenhafter „Closer“-Schlusssatz am Absatzende ({closer_hits}x)")

    neg_par_hits = 0
    for pat in NEGATIVE_PARALLELISM_PATTERNS:
        for m in re.finditer(pat, lower_text):
            neg_par_hits += 1
            raw_spans.append(
                TextSpan(m.start(), m.end(), "structure", 60, "[Struktur] Negativer Parallelismus („Nicht nur…, sondern…“)")
            )
    if neg_par_hits > 0:
        score += min(20.0, neg_par_hits * 9.0 * mult)
        reasons.append(f"[Struktur] Negativer Parallelismus („Nicht nur…, sondern…“): {neg_par_hits}x")

    triplet_matches = list(re.finditer(RULE_OF_THREE_PATTERN, text))
    if triplet_matches:
        score += min(16.0, len(triplet_matches) * 5.5 * mult)
        reasons.append(f"[Rhetorik] Dreiergruppen-Aufzählung (Rule of Three): {len(triplet_matches)}x")
        for m in triplet_matches:
            raw_spans.append(
                TextSpan(m.start(), m.end(), "structure", 55, f"[Rule of Three] Dreier-Aufzählung: „{m.group(0)}“")
            )

    list_matches = list(re.finditer(LIST_STRUCT_PATTERN, text))
    if len(list_matches) >= 2:
        score += min(18.0, len(list_matches) * 5.5 * mult)
        reasons.append(f"[Struktur] Schablonenhafte Listenpunkte mit Begriff-Erklärung: {len(list_matches)}x")
        for m in list_matches:
            raw_spans.append(
                TextSpan(m.start(), m.end(), "structure", 58, "[Listen-Schablone] Begriff-Erklärungs-Liste")
            )

    dash_matches = list(re.finditer(r" — ", text))
    if len(dash_matches) >= 2:
        score += min(15.0, len(dash_matches) * 5.0 * mult)
        reasons.append(f"[Typografie] Gehäufte Geviertstriche mit Leerzeichen (—): {len(dash_matches)}x")
        for m in dash_matches:
            raw_spans.append(
                TextSpan(m.start(), m.end(), "buzzword", 40, "[Typografie] Gehäufter Geviertstrich (—)")
            )

    # --- Modul C: Tiefere Linguistik, Informationstheorie & Satz-Rhythmus ---
    ling = analyze_deep_linguistics(text, nlp=nlp)
    words = ling["words"]
    sentences = ling["sentences"]
    lex_density = ling["lex_density"]
    pos_entropy = ling["pos_entropy"]
    avg_tree_depth = ling["avg_tree_depth"]
    tree_depth_cv = ling["tree_depth_cv"]
    avg_tree_width = ling["avg_tree_width"]
    tree_width_cv = ling["tree_width_cv"]
    avg_branching_factor = ling["avg_branching_factor"]
    noun_chunk_density = ling["noun_chunk_density"]
    dep_entropy = ling["dep_entropy"]
    morph_entropy = ling["morph_entropy"]
    entity_density = ling["entity_density"]
    avg_synsets = ling["avg_synsets"]

    word_count = len(words)
    sentence_count = len(sentences)

    sent_offset_matches = list(re.finditer(r"[^.!?]+[.!?]?", text))
    starter_matches = []
    for sm in sent_offset_matches:
        s_raw = sm.group(0)
        leading_ws = len(s_raw) - len(s_raw.lstrip())
        s_clean = s_raw.lstrip().lower()
        for pat in SENTENCE_STARTER_CONNECTORS:
            cm = re.search(pat, s_clean)
            if cm:
                abs_s = sm.start() + leading_ws + cm.start()
                abs_e = sm.start() + leading_ws + cm.end()
                starter_matches.append((abs_s, abs_e, text[abs_s:abs_e]))
                break

    non_empty_sents = [sm for sm in sent_offset_matches if sm.group(0).strip()]
    if len(non_empty_sents) >= 3 and (len(starter_matches) / len(non_empty_sents)) >= 0.30:
        score += min(16.0, len(starter_matches) * 5.0 * mult)
        reasons.append(
            f"[Rhythmus] Häufige schablonenhafte Satzanfangs-Konnektoren ({len(starter_matches)} von {len(non_empty_sents)} Sätzen)"
        )
        for abs_s, abs_e, word_txt in starter_matches:
            raw_spans.append(
                TextSpan(abs_s, abs_e, "structure", 50, f"[Satzanfangs-Konnektor] „{word_txt}“")
            )

    nominal_ratio = 0.0
    if word_count >= 50:
        nominal_words = [
            w for w in words
            if len(w) >= 7 and re.search(r"(ung|ungen|heit|heiten|keit|keiten|tion|tionen|ität|itäten)$", w)
        ]
        nominal_ratio = len(nominal_words) / word_count
        if nominal_ratio >= 0.08:
            score += 11.0 * mult
            reasons.append(f"[Nominalstil] Hohe Dichte abstrakter Substantivierungen ({nominal_ratio * 100:.1f} % der Wörter)")

    ttr = 0.0
    hapax_ratio = 0.0
    burstiness_cv = 0.0
    avg_sent_len = 0.0

    if word_count > 0:
        counts = Counter(words)
        ttr = len(counts) / word_count
        hapax_count = sum(1 for c in counts.values() if c == 1)
        hapax_ratio = hapax_count / word_count

        if word_count >= 130 and ttr > 0.72:
            score += 11.0 * mult
            reasons.append(f"[Lexik] Hohe Type-Token-Ratio ({ttr:.2f}) trotz Textlänge ({word_count} Wörter)")
        elif word_count >= 100 and hapax_ratio < 0.48:
            score += 10.0 * mult
            reasons.append(f"[Lexik] Geringer Anteil einmaliger Wörter / Hapax Legomena ({hapax_ratio:.2f})")

    if sentence_count >= 3:
        sent_lengths = [len(s) for s in sentences]
        avg_sent_len = statistics.mean(sent_lengths)
        std_sent_len = statistics.stdev(sent_lengths)
        burstiness_cv = (std_sent_len / avg_sent_len) if avg_sent_len > 0 else 0.0

        if burstiness_cv < 0.30:
            score += 20.0 * mult
            reasons.append(
                f"[Burstiness] Sehr gleichförmige Satzlänge (CV {burstiness_cv:.2f}, Ø {avg_sent_len:.1f} Wörter)"
            )
        elif burstiness_cv < 0.45:
            score += 11.0 * mult
            reasons.append(
                f"[Burstiness] Eher gleichförmige Satzlänge (CV {burstiness_cv:.2f}, Ø {avg_sent_len:.1f} Wörter)"
            )

        has_short = any(l <= 6 for l in sent_lengths)
        has_long = any(l >= 28 for l in sent_lengths)
        if sentence_count >= 4 and not has_short and not has_long and 11.0 <= avg_sent_len <= 24.0:
            score += 8.0 * mult
            reasons.append("[Burstiness] Keine Satzlängen-Extreme (alle Sätze im einheitlichen Mittelbereich)")

    # Informationstheorie (Shannon-Entropie & Komprimierbarkeit)
    word_entropy, comp_ratio = compute_information_metrics(text, words)
    if len(text) >= 450 and comp_ratio < 0.46:
        score += 7.0 * mult
        reasons.append(f"[Info-Theorie] Hohe sprachliche Redundanz / Komprimierbarkeit (Ratio {comp_ratio:.2f})")

    # Tiefere spaCy-Merkmale (Syntaxbaum, Baumbreite, Verzweigung, Nominalphrasen, UD/Morphologie, Synsets)
    if lex_density is not None and word_count >= 80 and lex_density >= 0.58:
        score += 7.0 * mult
        reasons.append(f"[spaCy-Lexik] Sehr hohe Dichte an Inhaltswörtern ({lex_density:.2f})")

    if tree_depth_cv is not None and sentence_count >= 4 and tree_depth_cv < 0.22:
        score += 9.0 * mult
        reasons.append(
            f"[spaCy-Syntax] Gleichförmige Satzbaum-Tiefe (CV {tree_depth_cv:.2f}, Ø {avg_tree_depth:.1f})"
        )

    if tree_width_cv is not None and sentence_count >= 4 and tree_width_cv < 0.24:
        score += 6.5 * mult
        reasons.append(
            f"[spaCy-Syntax] Gleichförmige Satzbaum-Breite (CV {tree_width_cv:.2f}, Ø {avg_tree_width:.1f})"
        )

    if avg_branching_factor is not None and sentence_count >= 4 and avg_branching_factor < 1.65:
        score += 5.0 * mult
        reasons.append(f"[spaCy-Syntax] Geringer Verzweigungsfaktor ({avg_branching_factor:.2f})")

    if noun_chunk_density is not None and word_count >= 80 and noun_chunk_density >= 24.0:
        score += 6.0 * mult
        reasons.append(f"[spaCy-Syntax] Hohe Nominalphrasen-Dichte ({noun_chunk_density:.1f} pro 100 Wörter)")

    if dep_entropy is not None and word_count >= 80 and dep_entropy < 3.15:
        score += 5.5 * mult
        reasons.append(f"[spaCy-UD] Geringe Vielfalt der Syntax-Relationen (Entropie {dep_entropy:.2f})")

    if morph_entropy is not None and word_count >= 80 and morph_entropy < 3.40:
        score += 5.0 * mult
        reasons.append(f"[spaCy-Morph] Geringe morphologische Variabilität (Entropie {morph_entropy:.2f})")

    if pos_entropy is not None and word_count >= 80 and pos_entropy < 2.95:
        score += 6.0 * mult
        reasons.append(f"[spaCy-POS] Geringe Wortarten-Variabilität (Entropie {pos_entropy:.2f})")

    if entity_density is not None and word_count >= 100 and entity_density < 0.01:
        score += 5.0 * mult
        reasons.append("[spaCy-NER] Kaum konkrete benannte Entitäten (eher vage/generische Prosa)")

    # Semantik: Hedges (Abschwächungen) & Inline-Markierung
    hedge_matches = []
    for pat in GERMAN_HEDGES:
        for m in re.finditer(pat, lower_text):
            hedge_matches.append(m)
            raw_spans.append(
                TextSpan(m.start(), m.end(), "buzzword", 42, f"[Semantik/Hedge] Abschwächung „{text[m.start():m.end()]}“")
            )
    hedge_density = (100.0 * len(hedge_matches) / word_count) if word_count > 0 else 0.0
    if len(hedge_matches) >= 3 and hedge_density >= 2.2:
        score += min(11.0, len(hedge_matches) * 2.5 * mult)
        reasons.append(f"[Semantik] Häufige Abschwächungen / Hedges ({len(hedge_matches)}x, {hedge_density:.1f}%)")

    # Emotion, Valenz & Psycholinguistik
    psy = compute_valence_and_psycholinguistics(text, words, entity_density)
    if word_count >= 70 and psy["smooth_pos_count"] >= 3 and psy["neg_spike_count"] == 0:
        score += 7.5 * mult
        reasons.append(
            f"[Valenz] Geglättete, leicht positive Tonalität ({psy['smooth_pos_count']}x positiv, keine negativen Ausschläge)"
        )

    if word_count >= 70 and psy["concreteness_score"] < 0.25:
        score += 6.5 * mult
        reasons.append(
            f"[Psycholinguistik] Geringe Konkretheit / stark abstrakter Stil (Index {psy['concreteness_score']:.2f})"
        )

    # Menschliche Gegensignale
    has_personal = bool(re.search(r"\b(ich|wir|mein|meine|unser|unsere)\b", lower_text))
    quote_matches = list(re.finditer(r"[„\"].{10,}[“\"]", text))
    has_quotes = bool(quote_matches)
    modal_matches = []
    for mp in MODAL_PARTICLES:
        for m in re.finditer(rf"\b{mp}\b", lower_text):
            modal_matches.append(m)
    has_modal = bool(modal_matches)

    if has_quotes:
        for m in quote_matches:
            raw_spans.append(
                TextSpan(m.start(), m.end(), "human", 30, "[Mensch-Indiz] Direktes Zitat")
            )
    if has_modal:
        for m in modal_matches:
            raw_spans.append(
                TextSpan(m.start(), m.end(), "human", 25, f"[Mensch-Indiz] Modalpartikel „{text[m.start():m.end()]}“")
            )

    if not has_personal and not has_quotes and not has_modal and word_count >= 80:
        score += 7.0 * mult
        reasons.append("[Stil] Entpersönlichter, glatter Stil ohne direkte Zitate oder Modalpartikeln")
    elif not hard_tell:
        human_bonus = 0.0
        if has_personal:
            human_bonus += 6.0
        if has_modal:
            human_bonus += 5.0
        if has_quotes:
            human_bonus += 5.0
        if entity_density is not None and entity_density >= 0.04:
            human_bonus += 4.0
        if psy["neg_spike_count"] >= 2:
            human_bonus += 5.0
        if human_bonus > 0:
            score -= min(16.0, human_bonus)

    score = max(0.0, min(100.0, round(score, 1)))
    if hard_tell or score >= profile["strong_threshold"]:
        label = "Stark KI-verdächtig"
    elif score >= profile["gray_threshold"]:
        label = "Grauzone"
    else:
        label = "Unauffällig"

    metrics = {
        "ttr": round(ttr, 3),
        "hapax_ratio": round(hapax_ratio, 3),
        "burstiness_cv": round(burstiness_cv, 3),
        "avg_sent_len": round(avg_sent_len, 1),
        "nominal_ratio": round(nominal_ratio, 3),
        "word_entropy": round(word_entropy, 2),
        "comp_ratio": round(comp_ratio, 3),
        "hedge_density": round(hedge_density, 2),
        "valence_mean": round(psy["valence_mean"], 3),
        "concreteness": round(psy["concreteness_score"], 2),
        "avg_word_len": round(psy["avg_word_len"], 2),
        "lex_density": round(lex_density, 3) if lex_density is not None else None,
        "avg_tree_depth": round(avg_tree_depth, 2) if avg_tree_depth is not None else None,
        "tree_depth_cv": round(tree_depth_cv, 3) if tree_depth_cv is not None else None,
        "avg_tree_width": round(avg_tree_width, 2) if avg_tree_width is not None else None,
        "tree_width_cv": round(tree_width_cv, 3) if tree_width_cv is not None else None,
        "avg_branching": round(avg_branching_factor, 2) if avg_branching_factor is not None else None,
        "noun_chunk_density": round(noun_chunk_density, 1) if noun_chunk_density is not None else None,
        "dep_entropy": round(dep_entropy, 2) if dep_entropy is not None else None,
        "morph_entropy": round(morph_entropy, 2) if morph_entropy is not None else None,
        "pos_entropy": round(pos_entropy, 2) if pos_entropy is not None else None,
        "entity_density": round(entity_density, 3) if entity_density is not None else None,
        "avg_synsets": round(avg_synsets, 2) if avg_synsets is not None else None,
    }

    return PassageResult(
        index=index,
        text=text,
        word_count=word_count,
        sentence_count=sentence_count,
        score=score,
        label=label,
        hard_tell=hard_tell,
        reasons=reasons,
        metrics=metrics,
        spans=resolve_spans(raw_spans),
    )


# ---------------------------------------------------------------------------
# Stufe 2: Selektive Prüfungen für die Grauzone (lokale Perplexität & Ollama)
# ---------------------------------------------------------------------------

def apply_perplexity_to_passage(
    passage: PassageResult,
    model_name: str = "dbmdz/german-gpt2",
    sensitivity: str = "Ausgewogen",
) -> PassageResult:
    ppl = compute_local_perplexity(passage.text, model_name=model_name)
    passage.perplexity = round(ppl, 1)
    profile = SENSITIVITY_PROFILES.get(sensitivity, SENSITIVITY_PROFILES["Ausgewogen"])
    mult = profile["multiplier"]

    if ppl < 25.0:
        delta = 20.0 * mult
        verdict = "sehr vorhersagbar / stark KI-typisch"
    elif ppl < 40.0:
        delta = 10.0 * mult
        verdict = "eher vorhersagbar / leicht KI-verdächtig"
    elif ppl > 70.0:
        delta = -10.0
        verdict = "hohe Originalität / eher menschlich"
    else:
        delta = 0.0
        verdict = "neutral"

    passage.score = max(0.0, min(100.0, round(passage.score + delta, 1)))
    passage.reasons.append(f"[Lokale Perplexität] {model_name}: PPL={ppl:.1f} ({verdict})")

    if passage.hard_tell or passage.score >= profile["strong_threshold"]:
        passage.label = "Stark KI-verdächtig"
    elif passage.score >= profile["gray_threshold"]:
        passage.label = "Grauzone"
    else:
        passage.label = "Unauffällig"

    return passage



def clean_ollama_model_name(value: str) -> str:
    """Entfernt den Status-Zusatz ' [aktiv]' aus dem Chooser-Eintrag."""
    return re.sub(r"\s+\[aktiv\]$", "", (value or "").strip())


def fetch_ollama_models(base_url: str = "http://localhost:11434") -> Tuple[List[str], List[str]]:
    """
    Fragt lokal laufendes Ollama ab:
    - /api/ps: aktuell im Speicher geladene / aktive Modelle
    - /api/tags: alle lokal installierten Modelle
    """
    base = base_url.rstrip("/")
    active_models: List[str] = []
    installed_models: List[str] = []

    # 1. Aktive Modelle (/api/ps) — falls ältere Ollama-Version keinen Endpoint hat, ignorieren
    try:
        req_ps = urllib.request.Request(f"{base}/api/ps", method="GET")
        with urllib.request.urlopen(req_ps, timeout=3) as resp:
            ps_data = json.loads(resp.read().decode("utf-8"))
        for item in ps_data.get("models", []):
            name = item.get("name") or item.get("model")
            if name and name not in active_models:
                active_models.append(name)
    except urllib.error.HTTPError:
        pass

    # 2. Installierte Modelle (/api/tags)
    req_tags = urllib.request.Request(f"{base}/api/tags", method="GET")
    with urllib.request.urlopen(req_tags, timeout=4) as resp:
        tags_data = json.loads(resp.read().decode("utf-8"))
    for item in tags_data.get("models", []):
        name = item.get("name") or item.get("model")
        if name and name not in installed_models:
            installed_models.append(name)

    return active_models, installed_models


def format_ollama_chooser_items(active_models: List[str], installed_models: List[str]) -> List[str]:
    """Sortiert aktive Modelle nach oben und markiert sie mit ' [aktiv]'."""
    items: List[str] = [f"{m} [aktiv]" for m in active_models]
    active_set = set(active_models)
    for m in installed_models:
        if m not in active_set:
            items.append(m)
    return items


def ollama_check_gray_zone(
    passage: PassageResult,
    model: str = "llama3.2",
    base_url: str = "http://localhost:11434",
    sensitivity: str = "Ausgewogen",
) -> PassageResult:
    prompt = (
        "Du bist ein forensischer Linguist. Bewerte anhand von Stil, Gleichförmigkeit, "
        "Floskeln und inhaltlicher Konkretheit, wie wahrscheinlich der folgende deutsche Text "
        "von einer KI generiert wurde.\n"
        "Antworte AUSSCHLIESSLICH als JSON mit genau zwei Feldern:\n"
        '{"ki_wahrscheinlichkeit": <Zahl 0-100>, "begruendung": "<1 kurzer deutscher Satz>"}\n\n'
        f"TEXT:\n{passage.text[:2500]}"
    )

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=45) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    raw_reply = data.get("response", "").strip()
    match = re.search(r"\{.*\}", raw_reply, re.DOTALL)
    if not match:
        passage.ollama_note = f"Ollama-Antwort nicht parsbar: {raw_reply[:120]}"
        return passage

    parsed = json.loads(match.group(0))
    ai_prob = float(parsed.get("ki_wahrscheinlichkeit", passage.score))
    reason = str(parsed.get("begruendung", "")).strip()

    combined = round(0.5 * passage.score + 0.5 * max(0.0, min(100.0, ai_prob)), 1)
    profile = SENSITIVITY_PROFILES.get(sensitivity, SENSITIVITY_PROFILES["Ausgewogen"])

    passage.score = combined
    passage.ollama_note = f"Ollama ({model}): {ai_prob:.0f}% — {reason}"
    passage.reasons.append(f"[KI-Zweitprüfung] {passage.ollama_note}")

    if combined >= profile["strong_threshold"]:
        passage.label = "Stark KI-verdächtig"
    elif combined >= profile["gray_threshold"]:
        passage.label = "Grauzone"
    else:
        passage.label = "Unauffällig"

    return passage


# ---------------------------------------------------------------------------
# Stufe 3: Gesamtauswertung & Bericht
# ---------------------------------------------------------------------------

def analyze_document(
    text: str,
    source_name: str = "Direkteingabe",
    sensitivity: str = "Ausgewogen",
    use_perplexity: bool = False,
    ppl_model: str = "dbmdz/german-gpt2",
    ppl_all_passages: bool = False,
    use_ollama: bool = False,
    ollama_model: str = "llama3.2",
    ollama_url: str = "http://localhost:11434",
) -> DocumentReport:
    nlp, spacy_info = load_german_spacy()
    blocks = segment_text(text)

    if not blocks:
        return DocumentReport(
            source_name=source_name,
            sensitivity=sensitivity,
            total_words=0,
            total_passages=0,
            ai_share_percent=0.0,
            strong_ai_percent=0.0,
            gray_zone_percent=0.0,
            human_percent=100.0,
            spacy_model_used=spacy_info,
            ppl_used="Deaktiviert",
            ollama_used="Deaktiviert",
            document_notes=["Kein analysierbarer Text gefunden."],
            passages=[],
        )

    results = [
        analyze_passage(i + 1, block, nlp=nlp, sensitivity=sensitivity)
        for i, block in enumerate(blocks)
    ]

    # Stufe 2a: Lokale Perplexität (Grauzone oder alle Passagen)
    ppl_status = "Deaktiviert"
    if use_perplexity:
        target_passages = results if ppl_all_passages else [r for r in results if r.label == "Grauzone"]
        if not target_passages:
            ppl_status = f"Aktiv ({ppl_model}), aber keine Grauzonen-Passagen angefallen"
        else:
            checked = 0
            errors = 0
            last_err = ""
            for gp in target_passages:
                try:
                    apply_perplexity_to_passage(gp, model_name=ppl_model, sensitivity=sensitivity)
                    checked += 1
                except Exception as exc:
                    errors += 1
                    last_err = str(exc)
                    gp.reasons.append(f"[Perplexität fehlgeschlagen] {exc}")
            ppl_status = f"Aktiv ({ppl_model}): {checked}/{len(target_passages)} Passage(n) geprüft"
            if errors:
                ppl_status += f", {errors} Fehler ({last_err[:80]})"

    # Stufe 2b: Optionaler Ollama-Check (nur für verbleibende Grauzonen-Passagen)
    ollama_status = "Deaktiviert"
    if use_ollama:
        gray_passages = [r for r in results if r.label == "Grauzone"]
        if not gray_passages:
            ollama_status = f"Aktiv ({ollama_model}), keine Grauzonen-Passagen angefallen"
        else:
            checked = 0
            errors = 0
            for gp in gray_passages:
                try:
                    ollama_check_gray_zone(
                        gp, model=ollama_model, base_url=ollama_url, sensitivity=sensitivity
                    )
                    checked += 1
                except (urllib.error.URLError, TimeoutError, ValueError, Exception) as exc:
                    errors += 1
                    gp.reasons.append(f"[KI-Zweitprüfung fehlgeschlagen] {exc}")
            ollama_status = f"Aktiv ({ollama_model}): {checked} Grauzonen-Passage(n) geprüft"
            if errors:
                ollama_status += f", {errors} Fehler"

    total_words = sum(r.word_count for r in results) or 1
    strong_words = sum(r.word_count for r in results if r.label == "Stark KI-verdächtig")
    gray_words = sum(r.word_count for r in results if r.label == "Grauzone")
    human_words = sum(r.word_count for r in results if r.label == "Unauffällig")

    weighted_ai_words = strong_words + sum(
        r.word_count * (r.score / 100.0) for r in results if r.label == "Grauzone"
    )

    doc_notes: List[str] = []
    if total_words < 250:
        doc_notes.append(
            f"Hinweis: Text hat nur {total_words} Wörter. Statistische Kennzahlen sind unter ~250–300 Wörtern weniger stabil."
        )

    lower_all = text.lower()
    found_reg = [w for w in AUSTRIACISMS_HELVETISMS if re.search(rf"\b{w}\b", lower_all)]
    if not found_reg and total_words >= 300:
        doc_notes.append("Keine typischen Regionalismen (Austriazismen/Helvetismen) gefunden (schwaches Zusatzindiz).")

    if len(results) >= 4:
        wcs = [r.word_count for r in results]
        mean_wc = statistics.mean(wcs)
        if mean_wc > 0 and (statistics.stdev(wcs) / mean_wc) < 0.18:
            doc_notes.append("Sehr gleichmäßige Blocklängen über das gesamte Dokument.")

    return DocumentReport(
        source_name=source_name,
        sensitivity=sensitivity,
        total_words=total_words,
        total_passages=len(results),
        ai_share_percent=round(100.0 * weighted_ai_words / total_words, 1),
        strong_ai_percent=round(100.0 * strong_words / total_words, 1),
        gray_zone_percent=round(100.0 * gray_words / total_words, 1),
        human_percent=round(100.0 * human_words / total_words, 1),
        spacy_model_used=spacy_info,
        ppl_used=ppl_status,
        ollama_used=ollama_status,
        document_notes=doc_notes,
        passages=results,
    )


def format_report_markdown(report: DocumentReport) -> str:
    lines = [
        f"# Analysebericht: {report.source_name}",
        "",
        f"- **Profil / Sensitivität:** {report.sensitivity}",
        f"- **Gesamtwörter:** {report.total_words} ({report.total_passages} analysierte Blöcke)",
        f"- **NLP-Backend:** {report.spacy_model_used}",
        f"- **Lokale Perplexität (Stufe 2a):** {report.ppl_used}",
        f"- **Ollama-Check (Stufe 2b):** {report.ollama_used}",
        f"- **Geschätzter KI-Textanteil (Indizienwert):** **{report.ai_share_percent:.1f} %**",
        f"  - Stark KI-verdächtig: {report.strong_ai_percent:.1f} %",
        f"  - Grauzone: {report.gray_zone_percent:.1f} %",
        f"  - Unauffällig: {report.human_percent:.1f} %",
        "",
        "> *Methodischer Hinweis:* Linguistische Merkmale liefern Indizien und keinen juristischen oder wissenschaftlichen Beweis. Stark normierte Fachtexte oder stark überarbeitete Texte können Fehlalarme oder Unterschätzungen auslösen.",
        "",
    ]
    if report.document_notes:
        lines.append("## Dokumentweite Hinweise")
        for note in report.document_notes:
            lines.append(f"- {note}")
        lines.append("")

    lines.append("## Passagen-Analyse")
    for p in report.passages:
        lines.append(
            f"### Passage {p.index} — {p.label} (Score: {p.score}/100 | {p.word_count} Wörter, {p.sentence_count} Sätze)"
        )
        m = p.metrics
        extras = [
            f"Entropie={m['word_entropy']}",
            f"Kompr.={m['comp_ratio']}",
            f"Hedges={m['hedge_density']}%",
            f"Valenz={m['valence_mean']}",
            f"Konkr.={m['concreteness']}",
        ]
        if m.get("lex_density") is not None:
            extras.append(f"Lex.Dichte={m['lex_density']}")
        if m.get("avg_tree_depth") is not None:
            extras.append(f"Syntaxtiefe={m['avg_tree_depth']} (CV={m['tree_depth_cv']})")
        if m.get("avg_tree_width") is not None:
            extras.append(f"Baumbreite={m['avg_tree_width']} (CV={m['tree_width_cv']})")
        if m.get("avg_branching") is not None:
            extras.append(f"Verzw.={m['avg_branching']}")
        if m.get("noun_chunk_density") is not None:
            extras.append(f"NP={m['noun_chunk_density']}/100W")
        if m.get("dep_entropy") is not None:
            extras.append(f"UD-Entr.={m['dep_entropy']}")
        if m.get("morph_entropy") is not None:
            extras.append(f"Morph-Entr.={m['morph_entropy']}")
        if m.get("pos_entropy") is not None:
            extras.append(f"POS-Entr.={m['pos_entropy']}")
        if m.get("avg_synsets") is not None:
            extras.append(f"Synsets={m['avg_synsets']}")
        if p.perplexity is not None:
            extras.append(f"PPL={p.perplexity}")

        lines.append(
            f"*Metriken:* TTR={m['ttr']}, Hapax={m['hapax_ratio']}, Burstiness-CV={m['burstiness_cv']}, "
            f"Ø Satzlänge={m['avg_sent_len']}, Nominalstil={m['nominal_ratio'] * 100:.1f}%, {', '.join(extras)}"
        )
        if p.reasons:
            for r in p.reasons:
                lines.append(f"- {r}")
        else:
            lines.append("- Keine auffälligen KI-Indizien gefunden.")

        ai_spans = [s for s in p.spans if s.category != "human"]
        if ai_spans:
            lines.append("- **Konkrete Fundstellen im Text:**")
            for sp in ai_spans[:8]:
                excerpt = p.text[sp.start:sp.end].replace("\n", " ")
                lines.append(f"  - `{excerpt}` → {sp.explanation}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graphische Oberfläche (Tkinter mit Inline-Highlighting & Hover-Info)
# ---------------------------------------------------------------------------

def launch_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext, ttk

    root = tk.Tk()
    root.title("Lokaler KI-Text-Prüfer v10 (Deutsch — Linguistik, Perplexität & Ollama-Chooser)")
    root.geometry("1280x860")

    current_source = {"name": "Direkteingabe"}
    last_report: dict = {"report": None}
    span_lookup: dict = {}

    # --- Top Toolbar (2 Zeilen für Übersichtlichkeit) ---
    toolbar = ttk.Frame(root, padding=6)
    toolbar.pack(fill=tk.X)

    toolbar2 = ttk.Frame(root, padding=(6, 0, 6, 4))
    toolbar2.pack(fill=tk.X)

    user_cfg = load_user_config()
    saved_ppl_model, has_saved_local_dir = get_saved_ppl_model()
    saved_use_ppl = bool(user_cfg.get("use_perplexity", has_saved_local_dir))
    saved_ppl_all = bool(user_cfg.get("ppl_all_passages", False))
    saved_sensitivity = user_cfg.get("sensitivity", "Ausgewogen")
    if saved_sensitivity not in SENSITIVITY_PROFILES:
        saved_sensitivity = "Ausgewogen"

    sensitivity_var = tk.StringVar(value=saved_sensitivity)
    use_ppl_var = tk.BooleanVar(value=saved_use_ppl)
    ppl_all_var = tk.BooleanVar(value=saved_ppl_all)
    ppl_model_var = tk.StringVar(value=saved_ppl_model)
    use_ollama_var = tk.BooleanVar(value=False)
    ollama_model_var = tk.StringVar(value=str(user_cfg.get("ollama_model", "llama3.2")))
    initial_status = (
        f"Bereit | Gespeicherter Modell-Ordner: {Path(saved_ppl_model).name}"
        if has_saved_local_dir
        else "Bereit (rein lokale Prüfung)"
    )
    status_var = tk.StringVar(value=initial_status)

    def on_load_pdf():
        path = filedialog.askopenfilename(
            title="PDF-Datei auswählen", filetypes=[("PDF-Dateien", "*.pdf")]
        )
        if not path:
            return
        try:
            extracted = extract_text_from_pdf(path)
            input_text.delete("1.0", tk.END)
            input_text.insert("1.0", extracted)
            current_source["name"] = Path(path).name
            status_var.set(f"PDF geladen: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Fehler beim PDF-Import", str(exc))

    def on_load_txt():
        path = filedialog.askopenfilename(
            title="Textdatei auswählen",
            filetypes=[("Text- & Markdown-Dateien", "*.txt *.md"), ("Alle Dateien", "*.*")],
        )
        if not path:
            return
        try:
            content = Path(path).read_text(encoding="utf-8", errors="replace")
            input_text.delete("1.0", tk.END)
            input_text.insert("1.0", content)
            current_source["name"] = Path(path).name
            status_var.set(f"Datei geladen: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Fehler beim Laden", str(exc))

    def render_report(report: DocumentReport):
        last_report["report"] = report
        span_lookup.clear()

        summary_var.set(
            f"Geschätzter KI-Anteil: {report.ai_share_percent:.1f}%  |  "
            f"Stark verdächtig: {report.strong_ai_percent:.1f}%  |  "
            f"Grauzone: {report.gray_zone_percent:.1f}%  |  "
            f"Unauffällig: {report.human_percent:.1f}%  |  "
            f"Wörter: {report.total_words}  ({report.sensitivity})"
        )
        status_var.set(f"Fertig | {report.spacy_model_used} | PPL: {report.ppl_used} | Ollama: {report.ollama_used}")
        hover_var.set("Tipp: Fahre mit der Maus über eine farbig markierte Stelle im Text, um den genauen Grund zu sehen.")

        highlighted_view.configure(state=tk.NORMAL)
        highlighted_view.delete("1.0", tk.END)

        for tag_name in highlighted_view.tag_names():
            if tag_name.startswith("hit_"):
                highlighted_view.tag_delete(tag_name)

        hit_counter = 0
        for p in report.passages:
            badge_tag = (
                "badge_strong" if p.label == "Stark KI-verdächtig"
                else "badge_gray" if p.label == "Grauzone"
                else "badge_human"
            )
            ppl_info = f" | PPL {p.perplexity}" if p.perplexity is not None else ""
            header = f" Passage {p.index} | {p.label} | Score {p.score}/100{ppl_info} \n"
            highlighted_view.insert(tk.END, header, badge_tag)

            cursor = 0
            for sp in p.spans:
                if sp.start > cursor:
                    highlighted_view.insert(tk.END, p.text[cursor:sp.start], "normal_text")

                hit_counter += 1
                hit_tag = f"hit_{hit_counter}"
                span_lookup[hit_tag] = f"Passage {p.index}: {sp.explanation}"

                cat_tag = f"cat_{sp.category}"
                highlighted_view.insert(tk.END, p.text[sp.start:sp.end], (cat_tag, hit_tag))
                highlighted_view.tag_bind(
                    hit_tag, "<Enter>", lambda _e, t=hit_tag: hover_var.set(span_lookup.get(t, ""))
                )
                highlighted_view.tag_bind(
                    hit_tag, "<Button-1>", lambda _e, t=hit_tag: hover_var.set(span_lookup.get(t, ""))
                )
                cursor = sp.end

            if cursor < len(p.text):
                highlighted_view.insert(tk.END, p.text[cursor:], "normal_text")
            highlighted_view.insert(tk.END, "\n\n", "normal_text")

        highlighted_view.configure(state=tk.DISABLED)

        details_view.configure(state=tk.NORMAL)
        details_view.delete("1.0", tk.END)
        details_view.insert("1.0", format_report_markdown(report))
        details_view.configure(state=tk.DISABLED)
        analyze_btn.configure(state=tk.NORMAL)

    def on_analyze():
        raw = input_text.get("1.0", tk.END).strip()
        if not raw:
            messagebox.showwarning("Kein Text", "Bitte zuerst Text einfügen oder eine Datei laden.")
            return

        analyze_btn.configure(state=tk.DISABLED)
        status_var.set("Analysiere (inkl. Linguistik / ggf. Perplexität)...")

        current_ppl = ppl_model_var.get().strip() or "dbmdz/german-gpt2"
        save_user_config({
            "ppl_model": current_ppl,
            "use_perplexity": bool(use_ppl_var.get()),
            "ppl_all_passages": bool(ppl_all_var.get()),
            "sensitivity": sensitivity_var.get(),
            "ollama_model": clean_ollama_model_name(ollama_model_var.get()) or "llama3.2",
        })

        def worker():
            report = analyze_document(
                raw,
                source_name=current_source["name"],
                sensitivity=sensitivity_var.get(),
                use_perplexity=use_ppl_var.get(),
                ppl_model=ppl_model_var.get().strip() or "dbmdz/german-gpt2",
                ppl_all_passages=ppl_all_var.get(),
                use_ollama=use_ollama_var.get(),
                ollama_model=clean_ollama_model_name(ollama_model_var.get()) or "llama3.2",
            )
            root.after(0, lambda: render_report(report))

        threading.Thread(target=worker, daemon=True).start()

    def on_export():
        if last_report["report"] is None:
            messagebox.showinfo("Hinweis", "Bitte zuerst eine Analyse durchführen.")
            return
        path = filedialog.asksaveasfilename(
            title="Bericht speichern",
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("Textdatei", "*.txt")],
        )
        if not path:
            return
        Path(path).write_text(format_report_markdown(last_report["report"]), encoding="utf-8")
        status_var.set(f"Bericht gespeichert: {Path(path).name}")

    # Toolbar Zeile 1
    ttk.Button(toolbar, text="PDF laden...", command=on_load_pdf).pack(side=tk.LEFT, padx=3)
    ttk.Button(toolbar, text="Text/MD laden...", command=on_load_txt).pack(side=tk.LEFT, padx=3)

    ttk.Label(toolbar, text="Sensitivität:").pack(side=tk.LEFT, padx=(12, 3))
    sens_box = ttk.Combobox(
        toolbar,
        textvariable=sensitivity_var,
        values=list(SENSITIVITY_PROFILES.keys()),
        state="readonly",
        width=12,
    )
    sens_box.pack(side=tk.LEFT, padx=3)

    analyze_btn = ttk.Button(toolbar, text="Analyse starten", command=on_analyze)
    analyze_btn.pack(side=tk.LEFT, padx=12)
    ttk.Button(toolbar, text="Export...", command=on_export).pack(side=tk.LEFT, padx=3)

    def on_test_ppl_model():
        m_name = ppl_model_var.get().strip() or "dbmdz/german-gpt2"
        test_ppl_btn.configure(state=tk.DISABLED)
        status_var.set(f"Lade/prüfe Modell '{m_name}' (beim ersten Mal ca. 510 MB Download)...")

        def loader():
            try:
                _tok, _mod, dev, max_pos = load_perplexity_model(m_name)
                save_user_config({"ppl_model": m_name})
                msg = f"Modell '{m_name}' bereit (Gerät: {dev}, max. {max_pos} Tokens)."
                root.after(0, lambda: (
                    status_var.set(msg),
                    test_ppl_btn.configure(state=tk.NORMAL),
                    messagebox.showinfo("Perplexitäts-Modell bereit", msg),
                ))
            except Exception as exc:
                err_msg = f"{type(exc).__name__}: {exc}"
                root.after(0, lambda: (
                    status_var.set(f"Modell-Fehler: {err_msg[:90]}"),
                    test_ppl_btn.configure(state=tk.NORMAL),
                    messagebox.showerror(
                        "Fehler beim Laden von german-gpt2",
                        f"Modell '{m_name}' konnte nicht geladen werden:\n\n{err_msg}\n\n"
                        "Hinweis: Prüfe, ob das Programm in deiner 'ki-pruefer'-Umgebung läuft "
                        "(conda activate ki-pruefer) und Internetzugang für den ersten Download besteht.",
                    ),
                ))

        threading.Thread(target=loader, daemon=True).start()

    # Toolbar Zeile 2 (Stufe-2-Optionen)
    ttk.Checkbutton(
        toolbar2,
        text="Lokale Perplexität (Transformers)",
        variable=use_ppl_var,
    ).pack(side=tk.LEFT, padx=(3, 4))
    ttk.Checkbutton(
        toolbar2,
        text="alle Passagen",
        variable=ppl_all_var,
    ).pack(side=tk.LEFT, padx=(0, 6))
    def on_choose_ppl_folder():
        current_val = ppl_model_var.get().strip()
        init_dir = str(Path(current_val).expanduser()) if current_val and Path(current_val).expanduser().is_dir() else str(Path.home())
        folder = filedialog.askdirectory(
            title="Lokalen Modell-Ordner auswählen (z. B. german-gpt2)",
            initialdir=init_dir,
        )
        if not folder:
            return
        ppl_model_var.set(folder)
        use_ppl_var.set(True)
        save_user_config({"ppl_model": folder, "use_perplexity": True})
        status_var.set(f"Lokaler Modell-Ordner gewählt & gespeichert: {Path(folder).name}")

    ttk.Entry(toolbar2, textvariable=ppl_model_var, width=22).pack(side=tk.LEFT, padx=2)
    ttk.Button(toolbar2, text="Ordner wählen...", command=on_choose_ppl_folder).pack(side=tk.LEFT, padx=2)
    test_ppl_btn = ttk.Button(toolbar2, text="Modell laden/testen", command=on_test_ppl_model)
    test_ppl_btn.pack(side=tk.LEFT, padx=(4, 4))

    ttk.Checkbutton(
        toolbar2,
        text="Grauzone via Ollama prüfen",
        variable=use_ollama_var,
    ).pack(side=tk.LEFT, padx=(16, 4))

    ollama_combo = ttk.Combobox(
        toolbar2,
        textvariable=ollama_model_var,
        values=[],
        width=22,
    )
    ollama_combo.pack(side=tk.LEFT, padx=2)

    def refresh_ollama_models(silent: bool = False):
        refresh_ollama_btn.configure(state=tk.DISABLED)
        if not silent:
            status_var.set("Suche laufende & installierte Ollama-Modelle...")

        def loader():
            try:
                active, installed = fetch_ollama_models()
                items = format_ollama_chooser_items(active, installed)

                def apply_ui():
                    refresh_ollama_btn.configure(state=tk.NORMAL)
                    ollama_combo["values"] = items
                    if items:
                        current_clean = clean_ollama_model_name(ollama_model_var.get())
                        if active and (not current_clean or current_clean == "llama3.2" or current_clean in active):
                            ollama_model_var.set(items[0])
                        elif current_clean not in installed and current_clean not in active:
                            ollama_model_var.set(items[0])
                        status_var.set(
                            f"Ollama bereit: {len(active)} aktiv, {len(installed)} installiert"
                        )
                    else:
                        status_var.set("Ollama erreichbar, aber keine Modelle installiert")
                        if not silent:
                            messagebox.showinfo(
                                "Keine Ollama-Modelle",
                                "Ollama läuft, es wurden aber keine installierten Modelle gefunden.\n"
                                "Installiere z. B. eines mit: ollama pull llama3.2",
                            )

                root.after(0, apply_ui)
            except Exception as exc:
                err_msg = f"{type(exc).__name__}: {exc}"

                def apply_err():
                    refresh_ollama_btn.configure(state=tk.NORMAL)
                    if not silent:
                        status_var.set("Ollama nicht erreichbar (localhost:11434)")
                        messagebox.showwarning(
                            "Ollama nicht erreichbar",
                            f"Konnte keine Verbindung zu Ollama (http://localhost:11434) herstellen:\n\n{err_msg}\n\n"
                            "Starte Ollama zuerst lokal und klicke dann erneut auf '↻'.",
                        )

                root.after(0, apply_err)

        threading.Thread(target=loader, daemon=True).start()

    refresh_ollama_btn = ttk.Button(
        toolbar2,
        text="↻",
        width=3,
        command=lambda: refresh_ollama_models(silent=False),
    )
    refresh_ollama_btn.pack(side=tk.LEFT, padx=(2, 2))

    # --- Status & Summary ---
    status_lbl = ttk.Label(root, textvariable=status_var, padding=(10, 2))
    status_lbl.pack(fill=tk.X)

    summary_var = tk.StringVar(value="Noch keine Analyse durchgeführt.")
    summary_lbl = ttk.Label(
        root, textvariable=summary_var, padding=(10, 4), font=("TkDefaultFont", 10, "bold")
    )
    summary_lbl.pack(fill=tk.X)

    # --- Legende ---
    legend_frame = ttk.Frame(root, padding=(10, 2))
    legend_frame.pack(fill=tk.X)
    legend_items = [
        ("Technik/ISBN", "#f8b4b4"),
        ("KI-Floskel/Kopula", "#f5c2c7"),
        ("Satz-Schablone/Closer", "#ffe0b2"),
        ("Modewort/Gedankenstrich", "#fff0a8"),
        ("Mensch-Indiz", "#c8e6c9"),
    ]
    tk.Label(legend_frame, text="Markierungen im Text:", font=("TkDefaultFont", 9, "bold")).pack(side=tk.LEFT, padx=(0, 6))
    for label_txt, bg_col in legend_items:
        tk.Label(
            legend_frame,
            text=f" {label_txt} ",
            background=bg_col,
            foreground="#111111",
            relief="solid",
            borderwidth=1,
            font=("TkDefaultFont", 8),
        ).pack(side=tk.LEFT, padx=3)

    # --- Paned Window ---
    main_pane = ttk.PanedWindow(root, orient=tk.VERTICAL)
    main_pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

    input_frame = ttk.LabelFrame(main_pane, text="Eingabetext (Reintext einfügen oder PDF/TXT laden)")
    input_text = scrolledtext.ScrolledText(
        input_frame,
        height=8,
        wrap=tk.WORD,
        background="#ffffff",
        foreground="#111111",
        insertbackground="#111111",
        selectbackground="#b3d7ff",
        selectforeground="#000000",
    )
    input_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    main_pane.add(input_frame, weight=1)

    bottom_pane = ttk.PanedWindow(main_pane, orient=tk.HORIZONTAL)
    main_pane.add(bottom_pane, weight=2)

    hl_frame = ttk.LabelFrame(
        bottom_pane,
        text="Inline-Markierung direkt im Text (Maus über Markierung bewegen für Details)",
    )
    highlighted_view = scrolledtext.ScrolledText(
        hl_frame,
        wrap=tk.WORD,
        state=tk.DISABLED,
        background="#ffffff",
        foreground="#111111",
        selectbackground="#b3d7ff",
        selectforeground="#000000",
    )
    highlighted_view.tag_config("normal_text", background="#ffffff", foreground="#111111")
    highlighted_view.tag_config(
        "badge_strong", font=("TkDefaultFont", 9, "bold"), background="#f8d7da", foreground="#58151c"
    )
    highlighted_view.tag_config(
        "badge_gray", font=("TkDefaultFont", 9, "bold"), background="#fff3cd", foreground="#664d03"
    )
    highlighted_view.tag_config(
        "badge_human", font=("TkDefaultFont", 9, "bold"), background="#d1e7dd", foreground="#0f5132"
    )

    highlighted_view.tag_config("cat_tech", background="#f8b4b4", foreground="#111111", underline=True)
    highlighted_view.tag_config("cat_phrase", background="#f5c2c7", foreground="#111111")
    highlighted_view.tag_config("cat_structure", background="#ffe0b2", foreground="#111111")
    highlighted_view.tag_config("cat_buzzword", background="#fff0a8", foreground="#111111")
    highlighted_view.tag_config("cat_human", background="#c8e6c9", foreground="#111111")

    highlighted_view.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    hover_var = tk.StringVar(value="Markierungs-Details erscheinen hier, sobald du über eine hervorgehobene Stelle fährst.")
    hover_lbl = tk.Label(
        hl_frame,
        textvariable=hover_var,
        anchor="w",
        background="#f1f5f9",
        foreground="#111111",
        padx=6,
        pady=4,
        font=("TkDefaultFont", 9),
    )
    hover_lbl.pack(fill=tk.X, padx=4, pady=(0, 4))
    bottom_pane.add(hl_frame, weight=1)

    det_frame = ttk.LabelFrame(bottom_pane, text="Detail-Analyse & Fundstellen")
    details_view = scrolledtext.ScrolledText(
        det_frame,
        wrap=tk.WORD,
        state=tk.DISABLED,
        background="#ffffff",
        foreground="#111111",
        selectbackground="#b3d7ff",
        selectforeground="#000000",
    )
    details_view.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    bottom_pane.add(det_frame, weight=1)

    root.after(200, lambda: refresh_ollama_models(silent=True))
    root.mainloop()


# ---------------------------------------------------------------------------
# Einstiegspunkt
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Lokaler KI-Text-Prüfer v11 (Deutsch)")
    parser.add_argument("--cli", type=str, help="Pfad zu einer PDF- oder Textdatei für den Terminal-Modus")
    parser.add_argument(
        "--sensitivity",
        choices=list(SENSITIVITY_PROFILES.keys()),
        default="Ausgewogen",
        help="Empfindlichkeitsprofil",
    )
    parser.add_argument("--perplexity", action="store_true", help="Lokale Perplexität berechnen")
    parser.add_argument("--ppl-all", action="store_true", help="Perplexität auf alle Passagen anwenden (nicht nur Grauzone)")
    parser.add_argument("--ppl-model", default=None, help="HuggingFace-Modell oder gespeicherter lokaler Ordner für Perplexität")
    parser.add_argument("--ollama", action="store_true", help="Grauzonen-Passagen via lokalem Ollama prüfen")
    parser.add_argument("--ollama-model", default="llama3.2", help="Ollama-Modellname (Standard: llama3.2)")
    args = parser.parse_args()

    if args.cli:
        p = Path(args.cli)
        if not p.exists():
            print(f"Datei nicht gefunden: {p}", file=sys.stderr)
            sys.exit(1)
        text = extract_text_from_pdf(str(p)) if p.suffix.lower() == ".pdf" else p.read_text(encoding="utf-8")
        report = analyze_document(
            text,
            source_name=p.name,
            sensitivity=args.sensitivity,
            use_perplexity=args.perplexity,
            ppl_model=args.ppl_model or get_saved_ppl_model()[0],
            ppl_all_passages=args.ppl_all,
            use_ollama=args.ollama,
            ollama_model=args.ollama_model,
        )
        print(format_report_markdown(report))
    else:
        launch_gui()


if __name__ == "__main__":
    main()
