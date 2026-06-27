# Oscar 2026 films in HdRezka comments

Automated quantitative content analysis of user comments to 28 feature films
connected with the 2026 Academy Awards season.

## Dataset scope

- full collected corpus: **20,660 comments**;
- films: **28**;
- stratified analytical sample: **2,555 comments**;
- sampling rule: all comments for films with at most 120 comments, otherwise
  a reproducible random sample of 120 comments (`seed = 20260626`);
- annotation model: local `qwen3:8b` through Ollama.

The full raw corpus and browser session files are deliberately excluded from
the public repository. The repository can contain the reproducible sample and
its model labels.

## Repository structure

```text
config/                    final annotation codebook
data/                      film list and optional public analytical datasets
docs/                      methodology and file provenance
prompts/                   final compact Ollama coding prompt
scripts/collection/        session capture, URL resolution and comment collection
scripts/sampling/          stratified sampling
scripts/annotation/        local Qwen3 annotation with checkpoints
scripts/analysis/          descriptive analysis of final v4 labels
scripts/utilities/         poster downloader
results/                   generated analysis outputs, ignored by Git
posters/                   downloaded posters, ignored by Git
```

## Environment

Recommended: Python 3.11–3.13.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install -r requirements.txt
python -m playwright install chromium
```

Install Ollama separately and download the model:

```bash
ollama pull qwen3:8b
```

## Pipeline

### 1. Capture an authenticated browser session

```bash
python scripts/collection/00_capture_session.py
```

The resulting `session/storage_state.json` is private and ignored by Git.

### 2. Resolve and validate film pages

```bash
python scripts/collection/01_resolve_urls.py
python scripts/collection/02_probe_comments.py
python scripts/collection/03_validate_urls.py
```

### 3. Collect comments

```bash
python scripts/collection/04_collect_comments.py
```

Raw data are written under `data/` and `artifacts/`; they are ignored by Git.

### 4. Create the stratified sample

```bash
python scripts/sampling/04_make_sample_safe.py
```

### 5. Annotate locally with Qwen3

```bash
python scripts/annotation/06_ollama_coder_v4_fast.py   --input data/comments_sample_2555.csv   --job-name sample_2555_v4   --batch-size 12
```

The same command resumes from the checkpoint after interruption. Do not add
`--restart` when resuming.

### 6. Analyze final labels

```bash
python scripts/analysis/08_analyze_v4.py   --input data/sample_2555_v4_labeled_ollama.csv   --output-dir results/analysis_v4   --min-film-n 30
```

### 7. Download film posters

```bash
python scripts/utilities/09_download_posters.py
```

## Method

The main method is automated quantitative content analysis. The unit of
analysis is an individual comment. The codebook covers relevance, evaluative
valence, explicitly expressed emotions, praise and criticism targets,
comparisons, Oscar stance, rhetorical modes and exploratory frames.

## Reproducibility and limitations

- Comments are public platform texts, but the full raw corpus is not included.
- The sample represents HdRezka comments, not all film viewers.
- One author may contribute several comments.
- Automated labels are more reliable for valence than for detailed emotions
  and frames.
- Browser session data, cookies, author salts, checkpoints and local paths
  must never be committed.
