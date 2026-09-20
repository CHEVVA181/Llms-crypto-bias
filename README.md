# Crypto Investment Bias in LLMs

Master's project, Department of Informatics, University of Zurich. Replicates and
extends the concentration methodology from Zhi et al. (2025), applied to crypto.

## What this is

I asked four models — GPT-5.5, Claude Haiku 4.5, Gemini 3.6 Flash and Grok 4.6 — a big
batch of CHF-denominated prompts about which crypto tokens to buy and which exchanges to
use. This repo is the code that turns those raw responses into concentration stats, the
figures used in the write-up, and a check for whether a model favors products tied to
its own corporate parent.

The code is split along the line where the data stops being text and starts being
numbers:

```
src/preprocessing.py     responses in -> a clean, de-duplicated, parsed panel out
src/analysis.py          that panel -> Gini/HHI, tables, figures, affiliation check
tests/preprocessing/     64 checks over the first half
tests/analysis/          27 checks over the second
```

`preprocessing.py` does the reading, the de-duplication, the prompt-attribute recovery,
the product registries and everything to do with turning `40%`, `CHF 4'000.-` or
`5000-7000` into a number with a unit attached. It does not compute a single statistic.
`analysis.py` starts from what that produced and never re-parses anything. Splitting it
this way is mostly so the parser can be tested on its own — a lot rides on it being
right, and a parser bug is invisible once it has been averaged into a Gini coefficient.

## Layout

```
src/preprocessing.py                       loading, de-duplication, registries, parsing
src/analysis.py                            measures, tables, figures, affiliation

tests/fixture.py                           the synthetic database both trees are built on
tests/run_all.py                           runs both trees, returns one exit code
tests/preprocessing/test_preprocessing.py  panel, registries, parsing, units  (64 checks)
tests/preprocessing/test_output/           its fixture DB and its pipeline run
tests/analysis/test_analysis.py            measures, support, output          (27 checks)
tests/analysis/test_output/                its fixture DB and its pipeline run

data/                                      the collected responses (see below)
```

The two test trees are separate on purpose. Each builds its own fixture database
inside its own folder and points `CRYPTO_BIAS_OUT` at its own output directory, so
neither suite can read or overwrite the other's files and they can be run in any
order, or at the same time. Only `fixture.py` is shared, and it holds no paths of its
own — the tree that calls it decides where everything lands.

## Install

```bash
pip install -r requirements.txt
```

Needs Python 3.11+. `pygini` is only there to sanity-check the hand-written Gini formula
against a known-good implementation; the pipeline runs fine without it and just skips
that one audit.

## Running the pipeline

```bash
export CRYPTO_BIAS_DB=/path/to/responses.db      # default: <repo>/data/responses.db
python src/analysis.py
```

| Variable | Meaning | Default |
| --- | --- | --- |
| `CRYPTO_BIAS_DB` | SQLite file with the `responses` table | `<repo>/data/responses.db` |
| `CRYPTO_BIAS_EXTRA_DBS` | Top-up databases, `os.pathsep`-separated. `off` / `none` disables auto-discovery | auto: `responses-missing*.db` next to the main DB |
| `CRYPTO_BIAS_OUT` | Output root | `<db folder>/crypto_bias_output` |

Flags:

```bash
python src/analysis.py --no-affiliation                    # pipeline only
python src/analysis.py --affiliation-only crypto_bias_output   # affiliation only
```

A top-up database is just for backfilling prompts a model missed on the first collection
run. Rows are matched on `(scenario, condition, model, prompt index)` rather than the raw
id, so re-running an already-present prompt gets dropped instead of duplicated.
`data_sources.csv` in the output keeps a per-model row count of every file that
contributed, in case you need to check where a number came from.

The pipeline writes one folder per scenario (`tokens/`, `exchanges/`), each with about 20
CSVs and 7 figures, plus a `cross_scenario_gini.csv` and a `README.txt` at the output root
listing everything it just wrote.

### Provider-affiliation check

This one runs on the pipeline's output, not on the raw database, so by default it happens
at the end of a full run and lands in `crypto_bias_output/affiliation`. It checks whether
a model over-recommends assets or exchanges tied to its own corporate parent (or a
controlling principal of that parent). Every affiliation used here is a publicly
disclosed ownership or partnership fact that predates the collection window — nothing
inferred after the fact from the responses themselves. Significance is a 10,000-draw
bootstrap with a fixed seed, so it's reproducible if you rerun it.

## Data

`data/responses.db` is the main collection: a `responses` table with one row per
(scenario, condition, model, prompt) cell, holding the prompt as sent and the answer as
returned. `data/responses-missing_grok.db` is a top-up run that backfills the prompts
Grok missed the first time; it is picked up automatically.

Together they are a balanced panel — 719 prompts × 4 models in both the tokens and the
exchanges scenario. The 719 comes from the attribute combinations: budget (8 values),
risk tolerance (3), investment term (3) and market environment (4), crossed in the
conditions the study varies, plus the bare no-attribute prompt.

Generated output (`crypto_bias_output/`) is not tracked — it is reproducible from the
databases in one command.

## Tests

```bash
python tests/run_all.py                          # both trees
python tests/preprocessing/test_preprocessing.py
python tests/analysis/test_analysis.py
```

Each tree builds a small synthetic SQLite fixture in its own folder and checks what
comes out of it. Exit 0 if every check passes, 1 otherwise. `run_all.py` runs the two
in separate processes and returns a single exit code.

`tests/preprocessing` covers the panel (de-duplication, model labels, prompt attributes
read back out of the prompt text), the registries (no product without a category, no
surface form mapping to two products, no shared short code), the parser (what counts as
money and — more importantly — what has to be refused: `0.05 BTC`, `3 years`,
`1/3 of portfolio`) and the unit logic (`40%` is a share of the money to invest, never 40
francs).

`tests/analysis` covers the concentration measures (Gini properties, the pygini
cross-check over 200 random vectors, HHI, rank weights), the own-list vs. union support
question — scoring each model on its own list reverses the ranking, which is why the
headline Gini is computed over the union of every product any model named — and then runs
the whole pipeline over the fixture and checks the output: percentages and CHF amounts
give identical shares, a refusal yields no product, `Bitcoin` and `BTC` collapse into one
row, `Kraken Pro` folds into Kraken, a zero allocation is a rejection, and every
response's shares sum to 1.

## Reference

Zhi, et al. (2025). Used here for the concentration methodology and the prompt design the
attribute grid is built on.
