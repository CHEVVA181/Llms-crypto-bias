# Crypto Investment Bias in LLMs

Master's project, Department of Informatics, University of Zurich. Replicates and
extends the concentration methodology from Zhi et al. (2025), applied to crypto.

## What this is

I asked four models — GPT-5.5, Claude Haiku 4.5, Gemini 3.6 Flash and Grok 4.6 — a big
batch of CHF-denominated prompts about which crypto tokens to buy and which exchanges to
use. The question is how concentrated those recommendations are: whether a handful of
products absorb most of the money and most of the mentions, whether that changes with
the budget, the risk tolerance, the term or the market environment, and whether a model
favors products tied to its own corporate parent.

## Methodology

### Collection

Every model saw the same prompt set, in two scenarios: which tokens to invest in, and
which exchanges to use. The prompts vary four attributes — budget, risk tolerance,
investment term and market environment — over every non-empty subset of the four, which
is 719 prompts per scenario. Four models across both scenarios makes a balanced panel of
719 × 4 × 2 responses, one row per (scenario, condition, model, prompt) cell holding the
prompt as sent and the answer as returned. A top-up run backfills the prompts Grok
missed on the first pass; rows are matched on (scenario, condition, model, prompt index)
rather than the raw id, so a prompt that is already present gets dropped instead of
duplicated.

### Parsing

Nothing is measured until the text is a number with a unit attached. De-duplication, the
prompt-attribute recovery, the product registries and the amount parsing all happen
first, and the parser refuses more than it accepts: `40%` is a share of the money to
invest and never 40 francs, while `0.05 BTC`, `3 years` and `1/3 of portfolio` are
rejected rather than guessed at. Surface forms collapse onto one product — `Bitcoin` and
`BTC` are one row, `Kraken Pro` folds into Kraken — and a zero allocation counts as a
rejection, not a recommendation. Parsing is kept strictly apart from measurement,
because a parser bug is invisible once it has been averaged into a Gini coefficient.

### Measures

Concentration is Gini and HHI, with rank weights for where in the list a product was
named. The headline Gini is computed over the union of every product any model named,
not over each model's own list: scoring each model only on the products it happened to
name reverses the ranking, which makes the union the honest denominator. The hand-written
Gini is cross-checked against a known-good implementation over random vectors.

### Breakdowns

The pooled index hides what the prompt actually moves, so four splits are computed
alongside it. Response status separates valid, hedged, refused and unparseable answers,
and the refusal rate is broken out by attribute value — a model that declines the crisis
prompts drops exactly the responses whose mix would have been most unusual, so the
missingness is informative rather than random. The tier mix is recomputed for every
attribute value, which is where the conditioning shows up: the head of the distribution
barely moves, the tail moves a lot. Exposure to venues inside the user's own
jurisdiction is counted separately, since every prompt is denominated in Swiss francs
and so states that jurisdiction unambiguously. And budget utilisation — allocated francs
over the budget named in the prompt — is reported per model, because the allocation
shares only mean something if the answers add up to what was asked.

### Provider-affiliation check

The last step asks whether a model over-recommends assets or exchanges tied to its own
corporate parent, or to a controlling principal of that parent. Every affiliation used
is a publicly disclosed ownership or partnership fact that predates the collection
window — nothing inferred after the fact from the responses themselves. Significance is
a 10,000-draw bootstrap with a fixed seed, so the result is reproducible.

## Prompts

`prompts/` holds the prompt texts the responses were collected with, one file per
scenario and attribute combination, one prompt per line. The file name lists the
attributes that vary inside it and nothing else is stated in those prompts, so
`tokens-general-budget_risk.txt` is every budget crossed with every risk tolerance.

Four attributes are varied:

| Attribute | Values |
| --- | --- |
| Budget | 100, 1'000, 10'000, 20'000, 30'000, 40'000, 50'000, 100'000 CHF |
| Risk tolerance | risk-averse, risk-neutral, risk-seeking |
| Investment term | less than one year, one to three years, three to ten years |
| Market environment | expansion, crisis, recession, recovery |

Every non-empty subset of the four gets a file, 15 per scenario, and the tokens and the
exchanges side hold the same set:

```
budget                           8
risk                             3
term                             3
environment                      4

budget_risk                     24
budget_term                     24
budget_environment              32
risk_term                        9
risk_environment                12
term_environment                12

budget_risk_term                72
budget_risk_environment         96
budget_term_environment         96
risk_term_environment           36

budget_risk_term_environment   288
                              ----
                               719
```

## Reference

Zhi, et al. (2025). Used here for the concentration methodology and the prompt design the
attribute grid is built on.
