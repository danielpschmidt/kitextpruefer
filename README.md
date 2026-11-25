# Lokaler KI-Text-Prüfer (Deutsch) — Setup & Kurzanleitung (`ki_text_pruefer-v9.py`)

Dieses Tool prüft deutsche PDF-, Markdown- und Reintexte **komplett lokal** auf typische KI-Schreibmuster, statistische Auffälligkeiten, tiefere linguistische Merkmale (`spaCy`) sowie optional auf **Perplexität** (`dbmdz/german-gpt2`) und eine **Zweitprüfung über Ollama**.

---

## 1. Voraussetzungen & saubere Installation (macOS Apple Silicon / M1 Max)

Um Paketkonflikte mit deiner bestehenden Anaconda-`base`-Umgebung zu vermeiden, empfiehlt sich eine eigene, native ARM64-Conda-Umgebung:

```bash
# 1. Neue native Apple-Silicon-Umgebung erstellen und aktivieren
CONDA_SUBDIR=osx-arm64 conda create -n ki-pruefer python=3.11 -c conda-forge
conda activate ki-pruefer

# 2. Basis-Pakete (PDF, GUI-Unterstützung & deutsches NLP) installieren
pip install pypdf spacy
python -m spacy download de_core_news_sm

# 3. Optional: Für lokale Perplexitäts-Messung (PyTorch mit Apple-Silicon-MPS + Transformers)
pip3 install torch torchvision torchaudio
pip install transformers huggingface_hub certifi
```

### Kurztest für GPU-Beschleunigung (MPS) auf Apple Silicon

```bash
python -c "import torch; print('MPS verfügbar:', torch.backends.mps.is_available())"
```

*(Falls `True` ausgegeben wird, rechnet die Perplexitäts-Prüfung automatisch auf der GPU deines M1-Chips.)*

---

## 2. Vorbereitung der optionalen KI-Module

### A. Lokale Perplexität mit `dbmdz/german-gpt2` (~510 MB)

**Variante 1 — Automatischer Download (im Terminal vorab testen):**

```bash
conda activate ki-pruefer
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export SSL_CERT_FILE=$(python -m certifi)
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('dbmdz/german-gpt2'))"
```

**Variante 2 — Manueller Offline-Ordner (falls Python den Download blockiert):**

1. Öffne im Browser `https://huggingface.co/dbmdz/german-gpt2/tree/main`.
2. Lade mindestens diese **5 Dateien** in einen lokalen Ordner herunter (z. B. `~/Downloads/dbmdz`):
   - `config.json`
   - `model.safetensors` (ca. 509 MB)
   - `special_tokens_map.json`
   - `tokenizer.json`
   - `tokenizer_config.json`
3. Wähle diesen Ordner später direkt in der App über den Button **„Ordner wählen…“** aus.

> **Hinweis zu Terminal-Meldungen:** Beim Laden von `dbmdz/german-gpt2` kann `transformers` einmalig `attn.masked_bias | UNEXPECTED` und `loss_type=None` ausgeben. Beides ist normal und kann ignoriert werden. Das 512-Token-Limit des Modells wird automatisch per Sliding Window eingehalten.

### B. Optional: Zweitprüfung über lokales Ollama

1. Stelle sicher, dass **Ollama** auf deinem Mac läuft und mindestens ein Modell geladen ist (z. B. `ollama pull llama3.2`).
2. In der GUI erkennt der Dropdown-Chooser laufende Modelle automatisch und markiert sie ganz oben mit `[aktiv]`. Mit dem **`↻`**-Button kannst du die Liste jederzeit aktualisieren.

---

## 3. Programm starten

### Grafische Oberfläche (Standard)

```bash
conda activate ki-pruefer
python ki_text_pruefer-v9.py
```

### Terminal-Modus (CLI)

```bash
# Nur lokale Linguistik & Stilprüfung
python ki_text_pruefer-v9.py --cli dokument.pdf

# Mit Sensitivitätsprofil, Perplexität auf allen Passagen und lokalem Modell-Ordner
python ki_text_pruefer-v9.py --cli dokument.pdf --sensitivity Streng --perplexity --ppl-all --ppl-model ~/Downloads/dbmdz

# Mit zusätzlicher Ollama-Zweitprüfung für Grauzonen-Passagen
python ki_text_pruefer-v9.py --cli dokument.pdf --ollama --ollama-model llama3.2
```

---

## 4. Bedienung der Oberfläche Schritt für Schritt

1. **Text laden:** Über **„PDF laden…“**, **„Text/MD laden…“** oder direktes Einfügen ins obere Textfeld.
2. **Sensitivität wählen:**
   - `Konservativ`: Weniger Fehlalarme bei stark normierten Fachtexten.
   - `Ausgewogen` *(Standard)*: Für allgemeine, journalistische und akademische Texte.
   - `Streng`: Höchste Empfindlichkeit für subtile KI-Muster.
3. **Optionale Stufe 2 zuschalten:**
   - **Lokale Perplexität (Transformers):** Haken setzen; bei Bedarf **„alle Passagen“** aktivieren (sonst nur Grauzone). Mit **„Ordner wählen…“** den lokalen Modell-Ordner angeben und mit **„Modell laden/testen“** vorab prüfen.
   - **Grauzone via Ollama prüfen:** Haken setzen und gewünschtes Modell aus dem Dropdown wählen.
4. **Analyse starten & auswerten:**
   - Im linken Ergebnisfenster bleiben unauffällige Passagen weiß; nur **konkrete Treffer** werden farbig markiert (**Rot** = Technik/ISBN, **Korallenrot** = Floskeln/Kopula, **Orange** = Satzschablonen/Closer, **Gelb** = Modewörter/Geviertstriche, **Grün** = menschliche Gegensignale wie Zitate oder Modalpartikeln).
   - Fahre mit der **Maus über eine markierte Stelle** (oder klicke sie an), um unten sofort die Begründung zu sehen.
   - Rechts findest du die Einzelmetriken (TTR, Hapax, Burstiness-CV, Nominalstil, Entropie, Syntaxtiefe, Perplexität) und kannst den Bericht über **„Export…“** als `.md` oder `.txt` speichern.

---

## 5. Wichtige Hinweise zur Einordnung der Ergebnisse

- **Indizien, kein Beweis:** Linguistische Kennzahlen und Perplexitätswerte liefern Anhaltspunkte. Stark redigierte Texte, Übersetzungen oder stark standardisierte Fachtexte (z. B. Methodenteile) können Fehlalarme auslösen.
- **Textlänge:** Statistische Kennzahlen sind unter ~250–300 Wörtern weniger stabil. Lange Fließtexte ohne Leerzeilen (z. B. aus PDFs) werden in v9 automatisch an Satzgrenzen in Blöcke von ca. 150–250 Wörtern unterteilt.
- **Verlässlichstes Warnsignal:** Technische Chatbot-Reste (z. B. `utm_source=chatgpt.com`) und **ungültige ISBN-Prüfsummen** oder erfundene Literaturangaben sind deutlich aussagekräftiger als einzelne Stilmerkmale.

---

## 6. Häufige Probleme & schnelle Lösungen

- **`No module named 'torch'` / `'pypdf'` / `'spacy'`:** Du befindest dich wahrscheinlich noch in `base` statt in `ki-pruefer`. Führe `conda activate ki-pruefer` aus und starte das Skript erneut.
- **`is not a local folder and is not a valid model identifier list`:** Nutze **„Ordner wählen…“** mit den manuell heruntergeladenen Dateien von `dbmdz/german-gpt2` oder setze im Terminal `export SSL_CERT_FILE=$(python -m certifi)`.
- **Ollama-Liste ist leer:** Prüfe, ob die Ollama-App bzw. `ollama serve` lokal auf Port `11434` läuft, und klicke dann auf **`↻`**.
