"""
===========================================================================
 analysis/analysis.py

 Everything that turns the parsed panel into a result: the concentration
 measures (Gini, HHI), the per-model and per-attribute tables, the figures
 used in the write-up, and the provider-affiliation check.

 Reading responses, normalising product names and reading amounts all happen
 in preprocessing/preprocessing.py - this module starts from what that one
 produced.  It is runnable from any working directory: it puts the sibling
 preprocessing folder on sys.path itself.

     python analysis/analysis.py                 # pipeline + affiliation check
     python analysis/analysis.py --no-affiliation
     python analysis/analysis.py --affiliation-only <pipeline_output> [<dest>]
===========================================================================
"""
import re
import sys
import warnings
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

# preprocessing lives in its own folder next to this one.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "preprocessing"))

from preprocessing import (
    ATTRIBUTES, BLUE_CHIPS, BUDGET_ORDER, DB_PATH, ENV_ORDER,
    FLAT_BUDGET_AS_ALTERNATIVES, GINI_SUPPORT, NON_PRODUCT_RE, OUT_ROOT,
    PCT_SUM_TOL, RANGE_POLICY, RANK_WEIGHTING, REFUSAL_RE, RISK_ORDER,
    SAVE_PDF, SCENARIOS, SCENARIO_CFG, TERM_ORDER, ZERO_AMOUNT_AS_REJECTION,
    clean_name, load_panel, parse_amount, resolve_amounts, split_line,
    summarise_panel,
)

try:                                     # Windows consoles are not utf-8
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)


# ===========================================================================
#  1.  CONCENTRATION MEASURES
#
#  gini_paper is the formula as written in Zhi et al. (2025); gini_pygini is
#  an independent implementation kept only to cross-check it.  The assert
#  below is a silent guard - if the two ever disagree the module refuses to
#  load rather than quietly reporting a wrong number.
# ===========================================================================
def rank_weights(n: int) -> np.ndarray:
    idx = np.arange(1, n + 1)
    if RANK_WEIGHTING == "uniform":
        w = np.ones(n)
    elif RANK_WEIGHTING == "zipf":
        w = 1.0 / idx
    else:
        w = (n - idx + 1).astype(float)
    return w / w.sum()

try:
    from pygini import gini as pygini_gini
    HAVE_PYGINI = True
except ImportError:
    print("\n!! pygini not installed - run:  pip install pygini")
    HAVE_PYGINI = False

def gini_paper(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return np.nan
    total = x.sum()
    if total <= 0:
        return 0.0
    x = np.sort(x)
    i = np.arange(1, n + 1)
    return float(np.sum((2 * i - n - 1) * x) / (n * total))

def gini_pygini(x) -> float:
    if not HAVE_PYGINI:
        return np.nan
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(pygini_gini(np.ascontiguousarray(x))) if x.size else np.nan

_rng = np.random.default_rng(0)
_maxdiff = 0.0
for _ in range(500):
    v = _rng.random(int(_rng.integers(2, 80))) * int(_rng.integers(1, 1000))
    if HAVE_PYGINI:
        _maxdiff = max(_maxdiff, abs(gini_paper(v) - gini_pygini(v)))
assert (not HAVE_PYGINI) or _maxdiff < 1e-6, _maxdiff   # silent guard

def hhi(shares) -> float:
    s = np.asarray(shares, float)
    s = s[~np.isnan(s)]
    return float(np.sum(s ** 2)) if s.size else np.nan

# ===========================================================================
#  2.  FIGURE STYLING
# ===========================================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "figure.dpi": 130, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "legend.frameon": False,
})

CAT_COLORS = {

    "Store of Value (PoW)": "orange", "Smart-Contract L1": "blue",
    "Layer 2 & Scaling": "cyan", "DeFi": "green",
    "AI & Infrastructure": "purple", "Payments": "pink",
    "Exchange & Platform": "gold", "RWA & Tokenized": "olive",
    "Stablecoin": "gray", "Meme": "red",

    "US-listed CEX": "blue", "EU-regulated CEX": "cyan",
    "Swiss bank/broker": "green", "Global CEX": "orange",
    "Derivatives venue": "purple", "CeFi lending": "olive",
    "Non-custodial/brokerage": "teal", "On-chain / DEX": "pink",
    "Defunct": "red",
}
PALETTE = ["blue", "red", "green", "orange", "purple", "cyan",
           "pink", "brown", "olive", "teal", "magenta", "navy"]
OTHERS_COLOR = "silver"
LABEL_COLOR = "black"

TIER_COLORS = {
    "Stable": "gray", "Blue chip": "blue", "Large-cap alt": "cyan",
    "Small-cap alt": "green", "Meme": "red",
    "Swiss regulated": "green", "EU regulated": "cyan", "US regulated": "blue",
    "Global / offshore": "orange", "Non-custodial": "purple",
    "Defunct / failed": "red",
}

SEQ_BLUE = mcolors.LinearSegmentedColormap.from_list("seq_blue", [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SEQ_BLUE = SEQ_BLUE.copy()
SEQ_BLUE.set_bad("#f0efec")

def _text_on(color):
    r, g, b = mcolors.to_rgb(color)
    return "black" if (0.299 * r + 0.587 * g + 0.114 * b) > 0.6 else "white"

def _short(model):
    return model.replace(" Flash", "").replace("Claude ", "")

# ===========================================================================
#  3.  THE PER-SCENARIO PIPELINE
#
#  parse -> normalise -> concentration -> tables -> figures, once for the
#  tokens scenario and once for the exchanges scenario.
# ===========================================================================
def run_scenario(scen: str, db: pd.DataFrame, model_order: list) -> dict:
    MODEL_ORDER = model_order
    cfg = SCENARIO_CFG[scen]
    registry = cfg["registry"]
    CATEGORY = cfg["category"]
    CATEGORY_ORDER = [c for c in cfg["category_order"]]
    TIER_BY_CAT = cfg["tier_by_category"]
    TIER_ORDER = cfg["tier_order"]
    PROD = cfg["product_word"]

    out_dir = OUT_ROOT / scen
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def save(df, name, index=False):
        df.to_csv(out_dir / name, index=index, encoding="utf-8-sig")

    def savefig(fig, name):
        fig.savefig(fig_dir / f"{name}.png", dpi=220)
        if SAVE_PDF:
            fig.savefig(fig_dir / f"{name}.pdf")
        plt.close(fig)
        print(f"  figures/{name}.png")

    alias = {}
    for canon, (code, aliases) in registry.items():
        for k in [canon.lower(), code.lower(), *[a.lower() for a in aliases]]:
            if k in alias and alias[k] != canon:
                raise KeyError(f"[{scen}] alias collision: {k!r} claimed by "
                               f"{alias[k]!r} and {canon!r}")
            alias[k] = canon
    CODE = {c: code for c, (code, _) in registry.items()}

    print("\n" + "=" * 74)
    print(f"SCENARIO: {scen.upper()}   "
          f"({len(registry)} canonical {PROD}s, {len(alias)} surface forms)")
    print("=" * 74)

    raw = db[db["scenario"] == scen].copy().reset_index(drop=True)
    print(raw.groupby("model").size().reindex(MODEL_ORDER).rename("responses").to_string())

    n_models = raw["model"].nunique()
    key_cov = raw.groupby("prompt_key")["model"].nunique()
    common_keys = set(key_cov[key_cov == n_models].index)
    print(f"prompt keys: {len(key_cov)} total, {len(common_keys)} answered by all {n_models} models")

    n_all = len(raw)
    dropped = raw[~raw["prompt_key"].isin(common_keys)]
    if len(dropped):
        save(dropped.groupby(["model", "condition"]).size()
             .rename("n_dropped").reset_index(), "paired_mode_dropped.csv")
    raw = raw[raw["prompt_key"].isin(common_keys)].reset_index(drop=True)
    raw["response_id"] = np.arange(len(raw))
    print(f"balanced panel: {n_all} -> {len(raw)} responses "
          f"({len(common_keys)} prompts x {n_models} models)")

    def parse_frame(frame: pd.DataFrame):
        records, status_rows = [], []
        unmapped, unmapped_example = Counter(), {}
        unparsed_amounts = Counter()

        for row in frame.itertuples(index=False):
            text = str(row.response)
            is_refusal = bool(REFUSAL_RE.search(text))

            items = []
            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                name_part, amt_part = split_line(line)
                name = clean_name(name_part)
                if not name or len(name) > 45 or NON_PRODUCT_RE.search(name):
                    continue

                key = name.lower().strip(" .")
                canon = alias.get(key)
                if canon is None:
                    canon = alias.get(re.sub(r"\s*\(.*?\)\s*", " ", key).strip())
                if canon is None:
                    if re.search(r"[a-z]", key) and len(key.split()) <= 4:
                        unmapped[key] += 1
                        unmapped_example.setdefault(key, name)
                    continue

                amount, kind, note, cur = parse_amount(amt_part, with_currency=True)
                if amount is None and amt_part.strip():
                    unparsed_amounts[amt_part.strip()[:60]] += 1
                items.append((canon, name, amount, kind, note, len(items) + 1, cur))

            if items and not is_refusal:
                stat = "valid"
            elif items and is_refusal:
                stat = "hedged"
            elif not items and is_refusal:
                stat = "refusal"
            else:
                stat = "unparseable"

            budget = (float(row.budget_chf)
                      if row.budget_chf is not None and pd.notna(row.budget_chf)
                      else np.nan)
            raw_vals = [i[2] for i in items]
            raw_units = [i[3] for i in items]
            money, weight, units, unit_basis, pct_sum, chf_sum = resolve_amounts(
                raw_vals, raw_units, budget)

            kinds_seen = {u for u, v in zip(units, raw_vals)
                          if v is not None and u in ("percent", "currency")}
            alloc_chf = float(np.nansum(money)) if np.isfinite(money).any() else np.nan

            curs = {c for c, v in zip([i[6] for i in items], raw_vals)
                    if v is not None and c}
            written_cur = (sorted(curs)[0] if len(curs) == 1
                           else ("mixed" if curs else None))
            non_chf = bool(curs - {"CHF"})

            flat_budget = False
            if (FLAT_BUDGET_AS_ALTERNATIVES and len(items) > 1
                    and np.isfinite(budget) and budget > 0
                    and np.isfinite(money).all()
                    and np.allclose(money, budget, rtol=1e-6, atol=1e-9)):
                flat_budget = True

            usable_util = (np.isfinite(budget) and budget > 0
                           and np.isfinite(alloc_chf)
                           and not non_chf and not flat_budget)
            status_rows.append({
                "response_id": row.response_id, "response_uid": row.id,
                "model": row.model,
                "condition": row.condition, "status": stat,
                "n_products": len(items), "n_chars": len(text),
                "n_unparsed_amounts": sum(1 for v in raw_vals if v is None),
                "n_ranges": sum(1 for i in items if i[4] == "range"),
                "n_loose_amounts": sum(1 for i in items if i[4] == "loose"),
                "mixed_units": len(kinds_seen) > 1,
                "amount_kind": (sorted(kinds_seen)[0] if len(kinds_seen) == 1
                                else ("mixed" if kinds_seen else "none")),
                "unit_basis": unit_basis,
                "pct_sum": pct_sum,
                "budget_chf": budget,
                "amount_sum_chf": alloc_chf,
                "written_currency": written_cur,
                "non_chf_currency": non_chf,
                "flat_full_budget": flat_budget,
                "budget_utilisation": (alloc_chf / budget if usable_util else np.nan),
            })

            if not items:
                continue

            ok = np.isfinite(weight)
            w = np.where(ok, weight, 0.0)
            if flat_budget:

                shares, basis = rank_weights(len(items)), "rank_flat_budget"
            elif ok.any() and w.sum() > 0:
                shares = w / w.sum()
                basis = "reported" if ok.all() else "reported_partial"
            elif ok.any():
                shares, basis = rank_weights(len(items)), "rank_all_zero"
            else:
                shares, basis = rank_weights(len(items)), "rank_imputed"

            for idx, (canon, rawname, amt, kind, note, rank, cur) in enumerate(items):
                if (ZERO_AMOUNT_AS_REJECTION and basis.startswith("reported")
                        and ok[idx] and weight[idx] == 0):
                    continue
                records.append({
                    "response_id": row.response_id, "model": row.model, "rank": rank,
                    "product_raw": rawname, "product": canon,
                    "code": CODE.get(canon, canon),

                    "amount_reported": (float(amt) if amt is not None else np.nan),
                    "amount_unit": (units[idx] if amt is not None else "unparsed"),
                    "amount_currency": cur,

                    "amount_chf": money[idx],
                    "unit_basis": unit_basis,
                    "amount_note": note, "share": shares[idx], "alloc_basis": basis,
                })

        p = pd.DataFrame(records)
        s = pd.DataFrame(status_rows)
        if p.empty:
            raise RuntimeError(f"[{scen}] nothing parsed")

        n_before = len(p)
        p = (p.sort_values(["response_id", "rank"])
             .groupby(["response_id", "product"], as_index=False)
             .agg(model=("model", "first"), rank=("rank", "min"),
                  product_raw=("product_raw", "first"), code=("code", "first"),
                  amount_reported=("amount_reported", lambda v: v.sum(min_count=1)),
                  amount_unit=("amount_unit", "first"),
                  amount_currency=("amount_currency", "first"),
                  amount_chf=("amount_chf", lambda v: v.sum(min_count=1)),
                  unit_basis=("unit_basis", "first"),
                  amount_note=("amount_note", "first"),
                  share=("share", "sum"), alloc_basis=("alloc_basis", "first"),
                  n_lines=("rank", "size")))
        if n_before != len(p):
            print(f"  collapsed {n_before - len(p)} duplicate lines "
                  f"(same canonical {PROD} named twice in one response)")

        p["share"] = p.groupby("response_id")["share"].transform(lambda v: v / v.sum())
        meta = (["response_id", "id", "condition", "prompt_key", "attrs", "budget_chf",
                 "risk_value", "term_value", "env_value"]
                + [f"has_{a}" for a in ATTRIBUTES])
        p = p.merge(frame[meta], on="response_id", how="left")

        p = p.rename(columns={"id": "response_uid"})
        p["category"] = p["product"].map(CATEGORY)
        p["tier"] = p["category"].map(TIER_BY_CAT)
        if scen == "tokens":
            p["tier"] = np.where(p["product"].isin(BLUE_CHIPS), "Blue chip", p["tier"])
        return p, s, unmapped, unmapped_example, unparsed_amounts

    parsed, status, unmapped, unmapped_example, unparsed_amounts = parse_frame(raw)

    save(status, "response_status.csv")
    save(parsed, "parsed_recommendations.csv")

    print(f"\nparsed rows (response x {PROD}): {len(parsed):,}")
    print(f"responses with >=1 {PROD}: {parsed['response_id'].nunique():,} / {len(raw):,} "
          f"(status=='valid': {(status['status'] == 'valid').sum():,})")
    print("\nresponse status by model:")
    print(pd.crosstab(status["model"], status["status"]).reindex(MODEL_ORDER).to_string())
    print("\nallocation basis:")
    print(parsed["alloc_basis"].value_counts().to_string())
    print("\namount unit by model (how the model expressed the allocation):")
    print(pd.crosstab(status["model"], status["amount_kind"]).reindex(MODEL_ORDER).to_string())
    print(f"responses with mixed %/currency units: {int(status['mixed_units'].sum())}")
    print(f"amounts given as a range (-> {RANGE_POLICY}): {int(status['n_ranges'].sum())}")
    print(f"amounts read with the loose fallback pattern: {int(status['n_loose_amounts'].sum())}")

    cur_counts = status["written_currency"].value_counts(dropna=True)
    if len(cur_counts):
        print("\ncurrency written by the model (prompts are all in CHF):")
        print(cur_counts.to_string())
    n_nonchf = int(status["non_chf_currency"].sum())
    if n_nonchf:
        with_budget = int((status["non_chf_currency"]
                           & status["budget_chf"].notna()).sum())
        print(f"responses answering in a non-CHF currency: {n_nonchf} "
              f"({with_budget} of them against a CHF budget in the prompt)")
        print("  -> figures kept at face value (no FX rate); shares are "
              "unaffected, budget utilisation is not computed for them")
        if with_budget:
            print("  !! WARNING: a non-CHF answer to a CHF budget - check "
                  "these before quoting any CHF figure")

    n_flat = int(status["flat_full_budget"].sum())
    if n_flat:
        print(f"\nresponses repeating the FULL budget on every {PROD} "
              f"({n_flat}): read as a ranked list of alternatives, not an "
              f"allocation" if FLAT_BUDGET_AS_ALTERNATIVES else
              f"\nresponses repeating the full budget on every {PROD}: {n_flat} "
              f"(scored as written - FLAT_BUDGET_AS_ALTERNATIVES is off)")
        print(status[status["flat_full_budget"]]
              .groupby("model").size().rename("n").to_string())

    print("\nunit basis (how a percentage was turned into money):")
    print(status["unit_basis"].value_counts().to_string())
    pct_resp = status[status["pct_sum"].notna()]
    if len(pct_resp):
        print(f"responses answering in %: {len(pct_resp)}  "
              f"(sum of the percentages: median {pct_resp['pct_sum'].median():.1f}, "
              f"within 95-105: {int(pct_resp['pct_sum'].between(95, 105).sum())})")

    if unmapped:
        save(pd.DataFrame([{"surface_form": k, "example_as_written": unmapped_example[k], "n": n}
                           for k, n in unmapped.most_common()]), "unmapped_names.csv")
        print(f"{len(unmapped)} unmapped surface forms -> unmapped_names.csv")
    if unparsed_amounts:
        save(pd.DataFrame(unparsed_amounts.most_common(), columns=["amount_text", "n"]),
             "unparsed_amounts.csv")

    UNION = sorted(parsed["product"].unique())
    print(f"\ngini support = {GINI_SUPPORT}: union = {len(UNION)} {PROD}s "
          f"recommended by at least one model")

    def _vec(series, support):
        return series.reindex(support, fill_value=0).to_numpy(float)

    def gini_on(series, support):
        return gini_paper(_vec(series, support))

    def gini_rows(df):
        rows = []
        for model, sub in df.groupby("model"):
            own = sorted(sub["product"].unique())
            amt = sub.groupby("product")["share"].sum()
            frq = sub["product"].value_counts()
            a_uni, f_uni = _vec(amt, UNION), _vec(frq, UNION)
            a_own, f_own = _vec(amt, own), _vec(frq, own)
            prim_a, prim_f = ((a_uni, f_uni) if GINI_SUPPORT == "union"
                              else (a_own, f_own))
            tot_a, tot_f = float(amt.sum()), float(frq.sum())
            rows.append({
                "model": model,
                "n_responses": sub["response_id"].nunique(),
                "n_products": len(own),
                "n_products_union": len(UNION),
                "gini_support": GINI_SUPPORT,

                "GI_amount_paper": gini_paper(prim_a),
                "GI_amount_pygini": gini_pygini(prim_a),
                "GI_freq_paper": gini_paper(prim_f),
                "GI_freq_pygini": gini_pygini(prim_f),

                "GI_amount_union": gini_paper(a_uni),
                "GI_freq_union": gini_paper(f_uni),

                "GI_amount_own": gini_paper(a_own),
                "GI_freq_own": gini_paper(f_own),
                "top1_amount_share": float(amt.max() / tot_a) if tot_a > 0 else np.nan,
                "top3_amount_share": float(amt.nlargest(3).sum() / tot_a) if tot_a > 0 else np.nan,
                "top1_freq_share": float(frq.max() / tot_f) if tot_f > 0 else np.nan,
                "top3_freq_share": float(frq.nlargest(3).sum() / tot_f) if tot_f > 0 else np.nan,
                "HHI_amount": hhi(amt / tot_a) if tot_a > 0 else np.nan,
                "effective_n_amount": (1.0 / hhi(amt / tot_a)) if tot_a > 0 else np.nan,
                "mean_products_per_response": len(sub) / sub["response_id"].nunique(),
            })
        return pd.DataFrame(rows).set_index("model").reindex(MODEL_ORDER).reset_index()

    gini_by_model = gini_rows(parsed)
    #  Rank 1 = most concentrated.  The two support sets can disagree on the
    #  ordering - a model with a narrow vocabulary looks equal under its own
    #  support and concentrated under the union - so both ranks are recorded.
    gini_by_model["rank_own"] = (gini_by_model["GI_amount_own"]
                                 .rank(ascending=False, method="min").astype(int))
    gini_by_model["rank_union"] = (gini_by_model["GI_amount_union"]
                                   .rank(ascending=False, method="min").astype(int))
    save(gini_by_model, "gini_by_model.csv")
    print(f"\n=== Gini by model (balanced panel, {GINI_SUPPORT} support) ===")
    show = ["model", "n_products", "n_products_union",
            "GI_amount_paper", "GI_amount_pygini", "GI_freq_paper", "GI_freq_pygini",
            "GI_amount_own", "GI_freq_own"]
    print(gini_by_model[show].round(4).to_string(index=False))

    print(f"\n=== Support set: own {PROD}s vs. the union "
          "(rank 1 = most concentrated) ===")
    print(gini_by_model[["model", "n_products", "GI_amount_own", "rank_own",
                         "GI_amount_union", "rank_union"]]
          .round(3).to_string(index=False))
    if not gini_by_model["rank_own"].equals(gini_by_model["rank_union"]):
        print("  !! the two supports imply different orderings - the union "
              "column is the one to quote")

    ver = gini_by_model[["model"]].copy()
    ver["abs_diff_amount"] = (gini_by_model["GI_amount_paper"]
                              - gini_by_model["GI_amount_pygini"]).abs()
    ver["abs_diff_freq"] = (gini_by_model["GI_freq_paper"]
                            - gini_by_model["GI_freq_pygini"]).abs()
    ver["agree_1e-9"] = (ver[["abs_diff_amount", "abs_diff_freq"]].max(axis=1) < 1e-9)
    save(ver, "gini_verification.csv")   # evidence on file, not on screen

    save(gini_by_model[["model", "n_responses", "n_products", "n_products_union",
                        "top1_amount_share", "top3_amount_share", "top1_freq_share",
                        "top3_freq_share", "HHI_amount", "effective_n_amount",
                        "mean_products_per_response"]], "concentration_summary.csv")

    st = status.merge(raw[["response_id", "prompt_key"]], on="response_id", how="left")
    st["usable"] = st["status"].isin(["valid", "hedged"])
    cov_u = (st.pivot_table(index="prompt_key", columns="model", values="usable",
                            aggfunc="max")
             .reindex(columns=MODEL_ORDER).fillna(False).astype(bool))
    paired_keys = set(cov_u.index[cov_u.all(axis=1)])

    print("\n=== robustness: differential refusal ===")
    print(f"prompts answered by every model: {len(paired_keys)} of {len(cov_u)}")
    if len(paired_keys) >= 30 and len(paired_keys) < len(cov_u):
        gini_paired = gini_rows(parsed[parsed["prompt_key"].isin(paired_keys)])
        cmp_ = (gini_by_model[["model", "GI_amount_paper", "GI_freq_paper", "n_products"]]
                .merge(gini_paired[["model", "GI_amount_paper", "GI_freq_paper",
                                    "n_products", "n_responses"]],
                       on="model", suffixes=("", "_paired")))
        cmp_ = cmp_.set_index("model").reindex(MODEL_ORDER).reset_index()
        save(cmp_, "gini_paired_usable.csv")
        print(cmp_[["model", "GI_amount_paper", "GI_amount_paper_paired",
                    "GI_freq_paper", "GI_freq_paper_paired"]]
              .round(4).to_string(index=False))
        same_rank = (list(cmp_.sort_values("GI_amount_paper", ascending=False)["model"])
                     == list(cmp_.sort_values("GI_amount_paper_paired",
                                              ascending=False)["model"]))
        if not same_rank:
            print("  !! the model ranking DEPENDS on who refused - report the "
                  "paired-usable numbers, not the headline")
    else:
        print("  every model answered (almost) every prompt - no confound to check")

    resp_with = (parsed.groupby("model")["response_id"].nunique()
                 .reindex(MODEL_ORDER, fill_value=0))
    resp_total = raw.groupby("model").size().reindex(MODEL_ORDER, fill_value=0)

    freq = (parsed.groupby(["model", "product", "code", "category", "tier"])
            .agg(n_mentions=("response_id", "size"),
                 n_responses=("response_id", "nunique"),
                 amount_share=("share", "sum"))
            .reset_index())
    freq["freq_share"] = freq["n_mentions"] / freq.groupby("model")["n_mentions"].transform("sum")
    freq["amount_share"] = freq["amount_share"] / freq.groupby("model")["amount_share"].transform("sum")
    freq = freq.sort_values(["model", "freq_share"], ascending=[True, False])
    save(freq, "recommendation_frequency.csv")

    print(f"\n=== Top-10 {PROD}s by recommendation frequency ===")
    for model in MODEL_ORDER:
        sub = freq[freq["model"] == model]
        print(f"\n--- {model} ({resp_with[model]} responses with {PROD}s "
              f"/ {resp_total[model]} total) ---")
        print(sub.head(10)[["product", "code", "category", "n_mentions", "freq_share",
                            "amount_share"]]
              .to_string(index=False, formatters={
                  "freq_share": "{:.2%}".format,
                  "amount_share": "{:.2%}".format}))

    def share_table(df, group, value, order):
        if value == "amount":
            m = df.pivot_table(index="model", columns=group, values="share", aggfunc="sum")
        else:
            m = df.pivot_table(index="model", columns=group, values="response_id", aggfunc="size")
        m = m.fillna(0.0)
        m = m.div(m.sum(axis=1), axis=0)
        cols = [c for c in order if c in m.columns]
        return m.reindex(MODEL_ORDER)[cols]

    cat_amount = share_table(parsed, "category", "amount", CATEGORY_ORDER)
    cat_freq = share_table(parsed, "category", "freq", CATEGORY_ORDER)[cat_amount.columns]
    tier_amount = share_table(parsed, "tier", "amount", TIER_ORDER)
    tier_freq = share_table(parsed, "tier", "freq", TIER_ORDER)[tier_amount.columns]

    save(cat_amount.round(4), "category_share_amount.csv", index=True)
    save(cat_freq.round(4), "category_share_frequency.csv", index=True)
    save(tier_amount.round(4), "risk_tier_share_amount.csv", index=True)
    save(tier_freq.round(4), "risk_tier_share_frequency.csv", index=True)

    print(f"\n=== {cfg['cat_word'].title()} share of investment amount (%) ===")
    print((cat_amount * 100).round(1).to_string())
    print(f"\n=== {cfg['tier_word'].title()} share of investment amount (%) ===")
    print((tier_amount * 100).round(1).to_string())

    CAT_SUPPORT = [c for c in CATEGORY_ORDER if c in set(parsed["category"])]
    rows = []
    for model, sub in parsed.groupby("model"):
        amt = sub.groupby("category")["share"].sum()
        frq = sub["category"].value_counts()
        rows.append({"model": model,
                     "GI_category_amount": gini_on(amt, CAT_SUPPORT),
                     "GI_category_freq": gini_on(frq, CAT_SUPPORT),
                     "n_categories_used": int((amt > 0).sum()),
                     "n_categories_support": len(CAT_SUPPORT)})
    gini_category = (pd.DataFrame(rows).set_index("model").reindex(MODEL_ORDER).reset_index()
                     .merge(gini_by_model[["model", "GI_amount_paper", "GI_freq_paper"]], on="model"))
    save(gini_category, "gini_by_category_level.csv")
    print(f"\n=== Gini: {PROD} level vs {cfg['cat_word']} level ({GINI_SUPPORT} support) ===")
    print(gini_category.round(4).to_string(index=False))

    VALUE_COLS = {"budget": ("budget_chf", BUDGET_ORDER),
                  "risk": ("risk_value", RISK_ORDER),
                  "term": ("term_value", TERM_ORDER),
                  "environment": ("env_value", ENV_ORDER)}
    rows = []
    for model, sub in parsed.groupby("model"):
        for a, (col, order) in VALUE_COLS.items():
            for val in order:
                s = sub[sub[col] == val]
                if s.empty:
                    continue
                amt = s.groupby("product")["share"].sum()
                frq = s["product"].value_counts()
                top = amt.idxmax() if float(amt.sum()) > 0 else None
                rows.append({"model": model, "attribute": a, "value": str(val),
                             "n_responses": s["response_id"].nunique(),
                             "n_products": int(amt.size),
                             "GI_amount": gini_on(amt, UNION),
                             "GI_freq": gini_on(frq, UNION),
                             "GI_amount_own": gini_paper(amt.to_numpy(float)),
                             f"top_{PROD}": top,
                             "top_share": float(amt.max() / amt.sum()) if float(amt.sum()) > 0 else np.nan})
    gini_val = pd.DataFrame(rows)
    save(gini_val, "gini_by_attribute_value.csv")
    print(f"\n=== GI(amount) by attribute VALUE ({GINI_SUPPORT} support) ===")
    print(gini_val.groupby(["attribute", "value"])["GI_amount"].mean()
          .round(3).to_string())

    ORDERED_VALUES = [(a, str(v)) for a, (_, order) in VALUE_COLS.items()
                      for v in order]

    #  Response status by model.  A refusal is an answer with no extractable
    #  recommendation; a hedged answer carries caveats but still names
    #  products and is kept.  Usable = valid + hedged, and is the denominator
    #  of every allocation statistic below.
    STATUS_ORDER = ["valid", "hedged", "refusal", "unparseable"]
    status_summary = (status.groupby(["model", "status"]).size().unstack(fill_value=0)
                      .reindex(index=MODEL_ORDER, columns=STATUS_ORDER, fill_value=0))
    status_summary["submitted"] = status_summary[STATUS_ORDER].sum(axis=1)
    status_summary["usable"] = status_summary["valid"] + status_summary["hedged"]
    status_summary["refusal_rate"] = (status_summary["refusal"]
                                      / status_summary["submitted"])
    status_summary = status_summary.reset_index()
    save(status_summary, "response_status_summary.csv")
    print("\n=== Response status by model (share of prompts submitted) ===")
    print(status_summary.assign(
        refusal_rate=(status_summary["refusal_rate"] * 100).round(2))
        .to_string(index=False))

    #  Refusal is not uniform over the grid: if a model declines the crisis
    #  prompts it also drops the responses whose tier mix would have been
    #  most unusual, so the missingness is informative, not random.
    st_attr = status.merge(
        raw[["response_id", "risk_value", "term_value", "env_value"]],
        on="response_id", how="left")
    st_attr["is_refusal"] = st_attr["status"].eq("refusal")
    rows = []
    for model, sub in st_attr.groupby("model"):
        for a, (col, order) in VALUE_COLS.items():
            for val in order:
                s = sub[sub[col] == val]
                if s.empty:
                    continue
                rows.append({"model": model, "attribute": a, "value": str(val),
                             "n_prompts": int(len(s)),
                             "n_refusals": int(s["is_refusal"].sum()),
                             "refusal_rate": float(s["is_refusal"].mean())})
    refusal_attr = pd.DataFrame(rows)
    save(refusal_attr, "refusal_by_attribute.csv")
    decliners = [m for m in MODEL_ORDER
                 if refusal_attr.loc[refusal_attr["model"] == m, "n_refusals"].sum() > 0]
    if decliners:
        print("\n=== Refusal rate by attribute value, % (models that decline) ===")
        piv = (refusal_attr[refusal_attr["model"].isin(decliners)]
               .pivot_table(index=["attribute", "value"], columns="model",
                            values="refusal_rate")
               .reindex(index=pd.MultiIndex.from_tuples(ORDERED_VALUES),
                        columns=decliners))
        print((piv * 100).round(1).to_string())
    else:
        print("\nno model refused a single prompt - refusal breakdown skipped")

    #  Allocation discipline: where the prompt names a budget, does the answer
    #  add up to it?  Not computed for answers written in another currency or
    #  for prompts with no budget, which is why n is below the usable count.
    #  The band a stated budget counts as met within.  EPS keeps a response
    #  that lands exactly on the edge (u = 0.98) on the inside of it, where
    #  binary floating point would otherwise push it out.
    UTIL_TOL, UTIL_EPS = 0.02, 1e-9
    util = status[status["budget_utilisation"].notna()]
    rows = []
    for model in MODEL_ORDER:
        u = util.loc[util["model"] == model, "budget_utilisation"].astype(float)
        if u.empty:
            continue
        rows.append({"model": model, "n": int(u.size),
                     "median": float(u.median()), "mean": float(u.mean()),
                     "within_tol": float((u.sub(1.0).abs()
                                          <= UTIL_TOL + UTIL_EPS).mean()),
                     "over_budget": float((u > 1.0 + UTIL_TOL + UTIL_EPS).mean()),
                     "n_over_1": int((u > 1.0 + UTIL_EPS).sum()),
                     "min": float(u.min()), "max": float(u.max())})
    budget_util = pd.DataFrame(rows)
    save(budget_util, "budget_utilisation.csv")
    print("\n=== Budget utilisation on the responses that state a budget "
          f"(within_tol = |1 - u| <= {UTIL_TOL:.0%}) ===")
    print(budget_util.round(4).to_string(index=False) if len(budget_util)
          else "  no response carries a usable budget")

    #  The pooled tier split hides the conditioning: the head of the
    #  distribution barely moves with the prompt, the tail moves a lot, and
    #  the tail is where the tier mix lives.
    rows = []
    for model, sub in parsed.groupby("model"):
        for a, (col, order) in VALUE_COLS.items():
            for val in order:
                s = sub[sub[col] == val]
                if s.empty:
                    continue
                amt = s.groupby("tier")["share"].sum()
                frq = s["tier"].value_counts()
                amt = amt / amt.sum() if float(amt.sum()) else amt
                frq = frq / frq.sum() if float(frq.sum()) else frq
                rec = {"model": model, "attribute": a, "value": str(val),
                       "n_responses": int(s["response_id"].nunique())}
                for t in TIER_ORDER:
                    rec[f"amount_{t}"] = float(amt.get(t, 0.0))
                    rec[f"freq_{t}"] = float(frq.get(t, 0.0))
                rows.append(rec)
    tier_attr = pd.DataFrame(rows)
    save(tier_attr, "tier_share_by_attribute.csv")
    rsk = tier_attr[tier_attr["attribute"] == "risk"]
    if len(rsk):
        print(f"\n=== {cfg['tier_word'].title()} share of amount by stated "
              "risk tolerance (%) ===")
        show_rsk = (rsk.set_index(["model", "value"])[
            [f"amount_{t}" for t in TIER_ORDER]]
            .rename(columns=lambda c: c[len("amount_"):])
            .reindex(pd.MultiIndex.from_product([MODEL_ORDER, RISK_ORDER]))
            .dropna(how="all"))
        print((show_rsk * 100).round(1).to_string())

    #  Every prompt is in Swiss francs, which states the user's jurisdiction.
    #  Does the model ever name a venue inside it?
    HOME_TIER = "Swiss regulated"
    if HOME_TIER in TIER_ORDER:
        rows = []
        for model in MODEL_ORDER:
            sub = parsed[parsed["model"] == model]
            if sub.empty:
                continue
            home = sub[sub["tier"] == HOME_TIER]
            usable = int(sub["response_id"].nunique())
            tot_amt = float(sub["share"].sum())
            rows.append({
                "model": model,
                "usable_responses": usable,
                "responses_naming_home_venue": int(home["response_id"].nunique()),
                "share_of_responses": (home["response_id"].nunique() / usable
                                       if usable else np.nan),
                "share_of_mentions": len(home) / len(sub) if len(sub) else np.nan,
                "share_of_amount": (float(home["share"].sum()) / tot_amt
                                    if tot_amt else np.nan),
                "n_home_venues_named": int(home["product"].nunique()),
            })
        home_exposure = pd.DataFrame(rows)
        save(home_exposure, "swiss_venue_exposure.csv")
        print(f"\n=== Exposure to {HOME_TIER} venues (the currency of every "
              "prompt names the jurisdiction) ===")
        print(home_exposure.assign(
            share_of_responses=(home_exposure["share_of_responses"] * 100).round(1),
            share_of_mentions=(home_exposure["share_of_mentions"] * 100).round(2),
            share_of_amount=(home_exposure["share_of_amount"] * 100).round(2))
            .to_string(index=False))

    #  Top-N by allocated amount with the rest collapsed into one row, so the
    #  table adds to 100% and the length of the tail stays visible.
    TOP_N = 10
    rows = []
    for model in MODEL_ORDER:
        sub = (freq[freq["model"] == model]
               .sort_values("amount_share", ascending=False))
        if sub.empty:
            continue
        for i, r in enumerate(sub.head(TOP_N).itertuples(index=False), start=1):
            rows.append({"model": model, "rank": i, "product": r.product,
                         "code": r.code, "category": r.category, "tier": r.tier,
                         "amount_share": float(r.amount_share),
                         "freq_share": float(r.freq_share)})
        tail = sub.iloc[TOP_N:]
        if len(tail):
            rows.append({"model": model, "rank": TOP_N + 1,
                         "product": f"Other ({len(tail)} {PROD}s)",
                         "code": None, "category": None, "tier": None,
                         "amount_share": float(tail["amount_share"].sum()),
                         "freq_share": float(tail["freq_share"].sum())})
    top_products = pd.DataFrame(rows)
    save(top_products, "top_products.csv")
    print(f"\n=== Top-{TOP_N} {PROD}s by allocated amount, tail collapsed ===")
    for model in MODEL_ORDER:
        sub = top_products[top_products["model"] == model]
        if sub.empty:
            continue
        print(f"\n--- {model} ---")
        print(sub[["rank", "product", "code", "tier", "amount_share", "freq_share"]]
              .to_string(index=False, formatters={
                  "amount_share": "{:.2%}".format,
                  "freq_share": "{:.2%}".format}))

    print("\nwriting figures...")

    sets = {m: set(freq[freq["model"] == m]["product"]) for m in MODEL_ORDER}
    rows = []
    for a, b in combinations(MODEL_ORDER, 2):
        inter = len(sets[a] & sets[b])
        smaller = min(len(sets[a]), len(sets[b]))
        rows.append({"model_a": a, "model_b": b,
                     "n_a": len(sets[a]), "n_b": len(sets[b]),
                     "n_shared": inter, "n_smaller": smaller,
                     "overlap_coverage": inter / smaller if smaller else np.nan})
    overlap = pd.DataFrame(rows)
    save(overlap, "overlap_recommendations.csv")
    print(f"\n=== Pairwise {PROD} recommendation overlap between models ===")
    print("overlap_coverage = n_shared / the smaller of the two supports")
    print(overlap.round(3).to_string(index=False))

    def top_k_colors(metrics, k=3):
        seen = []
        for metric in metrics:
            for model in MODEL_ORDER:
                for t in freq[freq["model"] == model].nlargest(k, metric)["code"]:
                    if t not in seen:
                        seen.append(t)
        return {t: PALETTE[i % len(PALETTE)] for i, t in enumerate(seen)}

    def plot_top_k(ax, metric, colors, k, title):
        x = np.arange(len(MODEL_ORDER))
        for xi, model in enumerate(MODEL_ORDER):
            sub = freq[freq["model"] == model].nlargest(k, metric)
            bottom = 0.0
            for _, r in sub.iloc[::-1].iterrows():
                c = colors[r["code"]]
                ax.bar(xi, r[metric], bottom=bottom, color=c, width=0.62,
                       edgecolor="white", linewidth=0.7)
                if r[metric] > 0.03:
                    ax.text(xi, bottom + r[metric] / 2, f"{r[metric]:.1%}",
                            ha="center", va="center", fontsize=7.5,
                            color=_text_on(c), fontweight="bold")
                bottom += r[metric]
            rest = max(0.0, 1 - bottom)
            ax.bar(xi, rest, bottom=bottom, color=OTHERS_COLOR, width=0.62,
                   edgecolor="white", linewidth=0.7)
            if rest > 0.03:
                ax.text(xi, bottom + rest / 2, f"{rest:.1%}", ha="center",
                        va="center", fontsize=7.5, color=LABEL_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels([_short(m) for m in MODEL_ORDER], rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(title)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)

    C3 = top_k_colors(["amount_share", "freq_share"], 3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    plot_top_k(axes[0], "amount_share", C3, 3, "Top-3 Investment Amount")
    plot_top_k(axes[1], "freq_share", C3, 3, "Top-3 Recommendation Frequency")
    axes[0].set_ylabel("Proportion of amount")
    axes[1].set_ylabel("Proportion of frequency")
    handles = [Patch(facecolor=c, label=t) for t, c in C3.items()]
    handles.append(Patch(facecolor=OTHERS_COLOR, label="Others"))
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.suptitle(f"Distribution of preferred {PROD}s in crypto investment recommendations",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, f"fig1_top3_{PROD}s")

    def plot_stack(ax, mat, colors, title, ylabel):
        bottoms = np.zeros(len(mat))
        x = np.arange(len(mat))
        for col in mat.columns:
            v = mat[col].to_numpy()
            ax.bar(x, v, bottom=bottoms, width=0.62, color=colors[col],
                   edgecolor="white", linewidth=0.7, label=col)
            for xi, (b, h) in enumerate(zip(bottoms, v)):
                if h > 0.045:
                    ax.text(xi, b + h / 2, f"{h:.0%}", ha="center", va="center",
                            fontsize=7, color=_text_on(colors[col]), fontweight="bold")
            bottoms += v
        ax.set_xticks(x)
        ax.set_xticklabels([_short(m) for m in mat.index], rotation=20, ha="right")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    plot_stack(axes[0], cat_amount, CAT_COLORS,
               f"{cfg['cat_word'].title()} share of investment amount", "Proportion of amount")
    plot_stack(axes[1], cat_freq, CAT_COLORS,
               f"{cfg['cat_word'].title()} share of recommendation frequency", "Proportion of frequency")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8,
               title=cfg["cat_word"].title(), title_fontsize=8.5)
    fig.suptitle(f"Distribution of preferred {PROD} {cfg['cat_word']}s",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    savefig(fig, "fig2_category_distribution")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    plot_stack(axes[0], tier_amount, TIER_COLORS,
               f"{cfg['tier_word'].title()} by investment amount", "Proportion of amount")
    plot_stack(axes[1], tier_freq, TIER_COLORS,
               f"{cfg['tier_word'].title()} by recommendation frequency", "Proportion of frequency")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8,
               title=cfg["tier_word"].title(), title_fontsize=8.5)
    fig.tight_layout()
    savefig(fig, "fig3_risk_tier")

    n = len(MODEL_ORDER)
    nrow = int(np.ceil(n / 2))
    fig, axes = plt.subplots(nrow, 2, figsize=(11, 3.1 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, model in zip(axes, MODEL_ORDER):
        sub = freq[freq["model"] == model].nlargest(10, "amount_share").iloc[::-1]
        y = np.arange(len(sub))
        ax.barh(y, sub["amount_share"], color=[CAT_COLORS[c] for c in sub["category"]],
                height=0.68)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["code"], fontsize=8)
        for yi, v in zip(y, sub["amount_share"]):
            ax.text(v + 0.004, yi, f"{v:.1%}", va="center", fontsize=7.5, color=LABEL_COLOR)
        ax.set_xlim(0, max(0.05, sub["amount_share"].max() * 1.22))
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_title(model, fontsize=9.5)
        ax.grid(axis="y", visible=False)
    for ax in axes[n:]:
        ax.axis("off")
    handles = [Patch(facecolor=CAT_COLORS[c], label=c) for c in CATEGORY_ORDER
               if c in set(freq["category"])]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8,
               title=cfg["cat_word"].title(), title_fontsize=8.5)
    fig.suptitle(f"Top-10 {PROD}s by investment amount, coloured by {cfg['cat_word']}",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    savefig(fig, "fig4_top10_by_category")

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for model in MODEL_ORDER:
        amt = parsed[parsed["model"] == model].groupby("product")["share"].sum()
        sup = UNION if GINI_SUPPORT == "union" else sorted(amt.index)
        v = np.sort(_vec(amt, sup))
        if v.sum() <= 0:
            continue
        cum = np.concatenate([[0], np.cumsum(v) / v.sum()])
        xs = np.linspace(0, 1, len(cum))
        ax.plot(xs, cum, label=f"{_short(model)}  GI={gini_paper(v):.3f}",
                color=dict(zip(MODEL_ORDER, PALETTE))[model], lw=1.8)
    ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", label="perfect equality")
    ax.set_xlabel(f"Cumulative share of {PROD}s (least to most funded, "
                  f"{GINI_SUPPORT} support)")
    ax.set_ylabel("Cumulative share of investment amount")
    ax.set_title(f"Lorenz curves - concentration of {PROD} recommendations")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    savefig(fig, "fig5_lorenz")

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, (a, (col, order)) in zip(axes, VALUE_COLS.items()):
        sub = gini_val[gini_val["attribute"] == a]
        if sub.empty:
            ax.axis("off")
            continue
        labels = [str(v) for v in order if str(v) in set(sub["value"])]
        x = np.arange(len(labels))
        w = 0.8 / max(1, len(MODEL_ORDER))
        for j, model in enumerate(MODEL_ORDER):
            s = sub[sub["model"] == model].set_index("value").reindex(labels)
            ax.bar(x + j * w - 0.4 + w / 2, s["GI_amount"], width=w,
                   color=dict(zip(MODEL_ORDER, PALETTE))[model],
                   label=_short(model) if a == "budget" else None)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v).replace(" ", "\n") if len(str(v)) > 12 else str(v)
                            for v in labels], fontsize=7.5, rotation=0)
        ax.set_title(a)
        ax.set_ylim(0, 1)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("GI (investment amount)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    fig.suptitle(f"Does the scenario move the bias?  GI of {PROD} recommendations "
                 "by attribute value", fontsize=11, fontweight="bold", y=1.04)
    fig.tight_layout()
    savefig(fig, "fig6_gini_by_attribute_value")

    ov = pd.DataFrame(np.nan, index=MODEL_ORDER, columns=MODEL_ORDER, dtype=float)
    for r in overlap.itertuples(index=False):
        ov.loc[r.model_a, r.model_b] = ov.loc[r.model_b, r.model_a] = r.overlap_coverage

    k = len(MODEL_ORDER)
    labels = [_short(m) for m in MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(1.15 * k + 1.5, 1.0 * k + 1.7))
    A = np.ma.masked_invalid(ov.to_numpy(float))
    ax.imshow(A, cmap=SEQ_BLUE, vmin=0, vmax=1)
    for i in range(k):
        for j in range(k):
            if A.mask[i, j]:
                continue
            ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if A[i, j] > 0.5 else "#222222")
    ax.set_xticks(range(k), labels, rotation=20, ha="right", fontsize=8.5)
    ax.set_yticks(range(k), labels, fontsize=8.5)

    ax.set_xticks(np.arange(k + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(k + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.grid(which="major", visible=False)
    ax.set_axisbelow(False)
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.set_title(f"{PROD.title()} recommendation overlap",
                 fontsize=11.5, fontweight="bold", pad=12)
    fig.tight_layout()
    savefig(fig, "fig7_pairwise_overlap")

    return dict(scen=scen, parsed=parsed, freq=freq, gini=gini_by_model,
                gini_val=gini_val, overlap=overlap, status=status,
                cat_amount=cat_amount, tier_amount=tier_amount, out_dir=out_dir)

# ===========================================================================
#  4.  PROVIDER-AFFILIATION CHECK
#
#  Do the four models over-weight assets and venues in which the corporate
#  parent, or a controlling principal of that parent, holds a disclosed
#  economic or reputational interest?  Runs on the pipeline's own output.
# ===========================================================================
RNG = np.random.default_rng(20260911)
N_BOOT = 10_000

MODELS = ["GPT-5.5", "Claude Haiku 4.5", "Gemini 3.6 Flash", "Grok 4.6"]
SHORT  = {"GPT-5.5": "GPT-5.5", "Claude Haiku 4.5": "Claude Haiku",
          "Gemini 3.6 Flash": "Gemini Flash", "Grok 4.6": "Grok 4.6"}

# ---------------------------------------------------------------------------
#  The affiliation map.  Every entry is an ownership or partnership fact that
#  was publicly disclosed BEFORE the collection window, not an inference from
#  the data.  `focal` is the model whose provider holds the interest.
# ---------------------------------------------------------------------------
AFFILIATIONS = [
    dict(key="grok_btc",      scenario="tokens",    code="BTC",
         focal="Grok 4.6",
         tie="Tesla and SpaceX (Musk-controlled) hold disclosed bitcoin treasuries"),
    dict(key="grok_doge",     scenario="tokens",    code="DOGE",
         focal="Grok 4.6",
         tie="Musk's sustained public association with Dogecoin; DOGE accepted by Tesla"),
    dict(key="gemini_sol",    scenario="tokens",    code="SOL",
         focal="Gemini 3.6 Flash",
         tie="Google Cloud runs Solana validators / block-data and dev partnerships"),
    dict(key="gemini_cb",     scenario="exchanges", code="COINBASE",
         focal="Gemini 3.6 Flash",
         tie="Coinbase-Google Cloud commercial partnership"),
    dict(key="gemini_gemini", scenario="exchanges", code="GEMINI",
         focal="Gemini 3.6 Flash",
         tie="Name collision only: the Gemini exchange is unrelated to Google (placebo)"),
    dict(key="claude_ftx",    scenario="exchanges", code="FTX",
         focal="Claude Haiku 4.5",
         tie="FTX/Alameda was a major early outside shareholder in Anthropic"),
    dict(key="gpt_wld",       scenario="tokens",    code="WLD",
         focal="GPT-5.5",
         tie="OpenAI's CEO co-founded World / Worldcoin (WLD)"),
]

def load(base, scenario):
    df = pd.read_csv(base / scenario / "parsed_recommendations.csv")
    df.columns = [c.lstrip("﻿") for c in df.columns]
    return df

def response_panel(df, code):
    """One row per parsed response: did it name `code`, and what share of the
    budget did it give it?  Absences are true zeros, not missing data."""
    resp = df[["response_uid", "model"]].drop_duplicates()
    hit = (df[df["code"] == code]
             .groupby(["response_uid", "model"], as_index=False)["share"].sum())
    out = resp.merge(hit, on=["response_uid", "model"], how="left")
    out["share"] = out["share"].fillna(0.0)
    out["named"] = (out["share"] > 0).astype(int)
    # a product can be named without a parseable amount -> recover the flag
    named_any = set(df.loc[df["code"] == code, "response_uid"])
    out.loc[out["response_uid"].isin(named_any), "named"] = 1
    return out

def cluster_boot(focal_v, other_v, n=N_BOOT):
    """Non-parametric bootstrap over responses; returns the 95% CI of the
    difference in means and a two-sided bootstrap p-value."""
    d0 = focal_v.mean() - other_v.mean()
    fi = RNG.integers(0, len(focal_v), size=(n, len(focal_v)))
    oi = RNG.integers(0, len(other_v), size=(n, len(other_v)))
    d = focal_v[fi].mean(axis=1) - other_v[oi].mean(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return d0, lo, hi, max(p, 1.0 / n)

def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        run = max(run, val)
        adj[idx] = min(run, 1.0)
    return adj


def run_affiliation(base: Path, out: Path) -> None:
    """`base` is the pipeline output root (the folder holding tokens/ and
    exchanges/); `out` is where the affiliation tables and figures go."""
    base, out = Path(base), Path(out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    # ---------------------------------------------------------------------------
    print("=" * 78)
    print(" PROVIDER-AFFILIATION ANALYSIS".center(78))
    print("=" * 78)

    data = {s: load(base, s) for s in ("tokens", "exchanges")}
    for s, df in data.items():
        n_resp = df["response_uid"].nunique()
        print(f"  loaded {s:10s}: {len(df):6,d} parsed rows   "
              f"{n_resp:5,d} responses   {df['code'].nunique():3d} products")
    print()

    rows = []
    for a in AFFILIATIONS:
        df = data[a["scenario"]]
        panel = response_panel(df, a["code"])
        f = panel[panel["model"] == a["focal"]]
        o = panel[panel["model"] != a["focal"]]
        fs, os_ = f["share"].to_numpy(), o["share"].to_numpy()

        if fs.sum() == 0 and os_.sum() == 0 and f["named"].sum() == 0 and o["named"].sum() == 0:
            rows.append(dict(key=a["key"], scenario=a["scenario"], code=a["code"],
                             focal=a["focal"], n_focal=len(f), n_other=len(o),
                             focal_share=0.0, other_share=0.0, lift=np.nan,
                             diff=0.0, ci_lo=0.0, ci_hi=0.0, p=1.0,
                             focal_named=0, other_named=0, tie=a["tie"],
                             status="ABSENT FROM CORPUS"))
            continue

        diff, lo, hi, p = cluster_boot(fs, os_)
        lift = (fs.mean() / os_.mean()) if os_.mean() > 0 else np.inf
        rows.append(dict(key=a["key"], scenario=a["scenario"], code=a["code"],
                         focal=a["focal"], n_focal=len(f), n_other=len(o),
                         focal_share=fs.mean(), other_share=os_.mean(), lift=lift,
                         diff=diff, ci_lo=lo, ci_hi=hi, p=p,
                         focal_named=int(f["named"].sum()),
                         other_named=int(o["named"].sum()),
                         tie=a["tie"], status="tested"))

    res = pd.DataFrame(rows)
    live = res["status"] == "tested"
    res.loc[live, "p_holm"] = holm(res.loc[live, "p"].to_numpy())

    print("-" * 78)
    print(" PRE-SPECIFIED AFFILIATION TESTS")
    print(" mean allocated share of budget, focal model vs. the other three pooled")
    print("-" * 78)
    hdr = f"{'asset':>9} {'focal model':>14} {'focal':>8} {'others':>8} {'lift':>6} {'diff':>8} {'95% CI':>18} {'p_holm':>9}"
    print(hdr); print("-" * 78)
    for _, r in res.iterrows():
        if r["status"] != "tested":
            print(f"{r['code']:>9} {SHORT[r['focal']]:>14}   -- {r['status']} --")
            continue
        ci = f"[{r['ci_lo']*100:+.2f},{r['ci_hi']*100:+.2f}]"
        print(f"{r['code']:>9} {SHORT[r['focal']]:>14} "
              f"{r['focal_share']*100:7.2f}% {r['other_share']*100:7.2f}% "
              f"{r['lift']:5.2f}x {r['diff']*100:+7.2f}pp {ci:>18} {r['p_holm']:9.4f}")
    print("-" * 78)
    print()

    # ---------------------------------------------------------------------------
    #  Placebo ranking: where does the affiliated asset sit among ALL assets when
    #  each model is ranked by its own-vs-others over-weight?
    # ---------------------------------------------------------------------------
    print("-" * 78)
    print(" PLACEBO RANKING  (every product, every model, ranked by over-weight)")
    print("-" * 78)
    rank_rows = []
    for scenario, df in data.items():
        codes = sorted(df["code"].dropna().unique())
        for m in MODELS:
            recs = []
            for c in codes:
                p = response_panel(df, c)
                fm = p.loc[p["model"] == m, "share"].mean()
                om = p.loc[p["model"] != m, "share"].mean()
                recs.append((c, fm, om, fm - om))
            d = pd.DataFrame(recs, columns=["code", "focal", "other", "diff"])
            d = d.sort_values("diff", ascending=False).reset_index(drop=True)
            d["rank"] = d.index + 1
            d["model"], d["scenario"], d["n_products"] = m, scenario, len(codes)
            rank_rows.append(d)
    ranks = pd.concat(rank_rows, ignore_index=True)

    for _, r in res[res["status"] == "tested"].iterrows():
        q = ranks[(ranks["scenario"] == r["scenario"]) & (ranks["model"] == r["focal"])
                  & (ranks["code"] == r["code"])]
        if len(q):
            q = q.iloc[0]
            print(f"  {SHORT[r['focal']]:>14} / {r['code']:<9} "
                  f"rank {int(q['rank']):3d} of {int(q['n_products']):3d} "
                  f"({r['scenario']} scenario)")
    print("-" * 78)
    print()

    print(" Largest single over-weight per model (tokens):")
    for m in MODELS:
        q = ranks[(ranks["scenario"] == "tokens") & (ranks["model"] == m)].iloc[0]
        print(f"   {SHORT[m]:>14}  {q['code']:<8} {q['diff']*100:+6.2f}pp "
              f"({q['focal']*100:5.2f}% vs {q['other']*100:5.2f}%)")
    print()
    print(" Largest single over-weight per model (exchanges):")
    for m in MODELS:
        q = ranks[(ranks["scenario"] == "exchanges") & (ranks["model"] == m)].iloc[0]
        print(f"   {SHORT[m]:>14}  {q['code']:<10} {q['diff']*100:+6.2f}pp "
              f"({q['focal']*100:5.2f}% vs {q['other']*100:5.2f}%)")
    print()

    res.to_csv(out / "affiliation_tests.csv", index=False)
    ranks.to_csv(out / "affiliation_lift_ranks.csv", index=False)
    print(f"  wrote {out/'affiliation_tests.csv'}")
    print(f"  wrote {out/'affiliation_lift_ranks.csv'}")
    print("=" * 78)

    # ===========================================================================
    #  CONFOUND CONTROL -- is the effect just list length?
    #  Grok names 2.90 products per response against 4.5-5.6 for the others, and a
    #  shorter list mechanically raises the share of whatever sits at the top.  The
    #  test below holds the number of products named in the response fixed and
    #  compares models only on the strata where all four are observed.
    # ===========================================================================
    MIN_CELL = 10

    def length_strata(scenario, code):
        df = data[scenario]
        lens = (df.groupby(["response_uid", "model"], as_index=False)["code"]
                  .nunique().rename(columns={"code": "k"}))
        pan = response_panel(df, code).merge(lens, on=["response_uid", "model"])
        tab, common = [], []
        for k in sorted(pan["k"].unique()):
            sub = pan[pan["k"] == k]
            cell = {"k": int(k), "n": int(len(sub))}
            full = True
            for m in MODELS:
                v = sub.loc[sub["model"] == m, "share"]
                cell[m] = v.mean() if len(v) >= MIN_CELL else np.nan
                cell[f"n_{m}"] = int(len(v))
                full &= len(v) >= MIN_CELL
            tab.append(cell)
            if full:
                common.append(int(k))
        tab = pd.DataFrame(tab)
        sup = tab[tab["k"].isin(common)]
        w = sup["n"] / sup["n"].sum()
        std = {m: float((sup[m] * w).sum()) for m in MODELS}
        raw = {m: float(pan.loc[pan["model"] == m, "share"].mean()) for m in MODELS}
        return tab, sup, std, raw, common

    print("=" * 78)
    print(" CONFOUND CONTROL: LIST LENGTH".center(78))
    print("=" * 78)

    strat_store = {}
    for code, scen in (("BTC", "tokens"), ("DOGE", "tokens"), ("SOL", "tokens")):
        tab, sup, std, raw, common = length_strata(scen, code)
        strat_store[code] = (tab, std, raw, common)
        print(f"\n  {code} -- allocated share of budget within list-length strata")
        print("  " + f"{'k':>3} " + "".join(f"{SHORT[m]:>16}" for m in MODELS) + f"{'n':>7}")
        for _, r in tab.iterrows():
            if r["n"] < 40:
                continue
            line = f"  {int(r['k']):>3} "
            for m in MODELS:
                line += (f"{r[m]*100:9.2f}% (n={int(r['n_'+m]):<3d})"
                         if not np.isnan(r[m]) else f"{'--':>16}")
            print(line + f"{int(r['n']):>7}")
        print(f"  common support: list lengths {common}")
        print(f"  {'':>18}{'raw':>10}{'length-std':>13}{'shift':>9}")
        for m in MODELS:
            print(f"  {SHORT[m]:>18}{raw[m]*100:9.2f}%{std[m]*100:12.2f}%"
                  f"{(std[m]-raw[m])*100:+8.2f}pp")
        focal = {"BTC": "Grok 4.6", "DOGE": "Grok 4.6", "SOL": "Gemini 3.6 Flash"}[code]
        others = [m for m in MODELS if m != focal]
        om = float(np.mean([std[m] for m in others]))
        verdict = "SURVIVES" if std[focal] > om else "DOES NOT SURVIVE"
        print(f"  --> length-standardised lift for {SHORT[focal]}: "
              f"{(std[focal]/om if om else np.inf):.2f}x   [{verdict}]")
    print()
    print("=" * 78)

    pd.concat([t.assign(code=c) for c, (t, *_ ) in strat_store.items()]) \
      .to_csv(out / "affiliation_length_strata.csv", index=False)
    strat = strat_store["BTC"][0]
    strat = strat[strat["n"] >= 40].reset_index(drop=True)

    # ===========================================================================
    #  TWO FURTHER PROBES
    #   (a) sector self-interest: all four vendors are AI companies -- do their
    #       models over-weight the AI & infrastructure token category?
    #   (b) XRP / Ripple, an early Google Ventures portfolio company.
    # ===========================================================================
    print("=" * 78)
    print(" SECTOR SELF-INTEREST: 'AI & Infrastructure' token category".center(78))
    print("=" * 78)
    tokdf = data["tokens"]
    ai_codes = sorted(tokdf.loc[tokdf["category"] == "AI & Infrastructure", "code"].unique())
    print(f"  category members: {', '.join(ai_codes)}")
    resp = tokdf[["response_uid", "model"]].drop_duplicates()
    aihit = (tokdf[tokdf["category"] == "AI & Infrastructure"]
               .groupby(["response_uid", "model"], as_index=False)["share"].sum())
    aip = resp.merge(aihit, on=["response_uid", "model"], how="left").fillna({"share": 0.0})
    lens = (tokdf.groupby(["response_uid", "model"], as_index=False)["code"]
              .nunique().rename(columns={"code": "k"}))
    aip = aip.merge(lens, on=["response_uid", "model"])
    common = [int(k) for k in sorted(aip["k"].unique())
              if all((aip[(aip["k"] == k) & (aip["model"] == m)].shape[0] >= MIN_CELL)
                     for m in MODELS)]
    sup = aip[aip["k"].isin(common)]
    wt = sup.groupby("k").size() / len(sup)
    print(f"  {'':>18}{'raw':>10}{'length-std':>13}{'responses':>11}")
    for m in MODELS:
        raw = aip.loc[aip["model"] == m, "share"].mean()
        std = sum(wt[k] * sup[(sup["k"] == k) & (sup["model"] == m)]["share"].mean()
                  for k in common)
        n = int((aip[aip["model"] == m]["share"] > 0).sum())
        print(f"  {SHORT[m]:>18}{raw*100:9.2f}%{std*100:12.2f}%{n:11d}")
    print(f"  common support: list lengths {common}")
    print()

    print("-" * 78)
    print(" XRP / Ripple  (Google Ventures led a 2015 Ripple Labs round)")
    print("-" * 78)
    xp = response_panel(tokdf, "XRP")
    for m in MODELS:
        v = xp[xp["model"] == m]
        print(f"  {SHORT[m]:>18}  named in {int(v['named'].sum()):3d} / {len(v):4d} responses"
              f"   mean allocated share {v['share'].mean()*100:5.3f}%")
    print("-" * 78)
    print()

    # --- length-controlled bootstrap for the one surviving signal ---------------
    print("-" * 78)
    print(" LENGTH-CONTROLLED BOOTSTRAP: DOGE, Grok 4.6 vs. the other three")
    print("-" * 78)
    dg = response_panel(tokdf, "DOGE").merge(lens, on=["response_uid", "model"])
    dgc = dg[dg["k"].isin(strat_store["DOGE"][3])]
    f = dgc[dgc["model"] == "Grok 4.6"]["share"].to_numpy()
    o = dgc[dgc["model"] != "Grok 4.6"]["share"].to_numpy()
    d0, lo, hi, pv = cluster_boot(f, o)
    print(f"  Grok      mean share {f.mean()*100:6.3f}%   (n = {len(f)} responses)")
    print(f"  others    mean share {o.mean()*100:6.3f}%   (n = {len(o)} responses)")
    print(f"  difference {d0*100:+.3f}pp   95% CI [{lo*100:+.3f}, {hi*100:+.3f}]   p = {pv:.4f}")
    dgn = response_panel(tokdf, "DOGE")
    print()
    print("  responses naming DOGE at all:")
    for m in MODELS:
        v = dgn[dgn["model"] == m]
        print(f"    {SHORT[m]:>18}  {int(v['named'].sum()):3d} / {len(v):4d}"
              f"  ({v['named'].mean()*100:5.2f}% of responses)")
    print("-" * 78)
    print()

    # ===========================================================================
    #  FIGURE 8
    # ===========================================================================
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "figure.dpi": 130, "savefig.bbox": "tight", "savefig.facecolor": "white",
        "legend.frameon": False,
    })
    MODEL_COLORS = {"GPT-5.5": "green", "Claude Haiku 4.5": "orange",
                    "Gemini 3.6 Flash": "blue", "Grok 4.6": "red"}
    SUPPORT, AGAINST, NULLC = "#1b7f3b", "#b02318", "#8a8a8a"

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.7),
                             gridspec_kw={"width_ratios": [1.12, 1.0]})
    fig2, axes2 = plt.subplots(1, 2, figsize=(12.6, 4.7),
                               gridspec_kw={"width_ratios": [1.0, 1.15]})

    # --- panel A: affiliation lift, log scale -----------------------------------
    ax = axes[0]
    tested = res[res["status"] == "tested"].copy()
    tested["lab"] = [f"{r['code']} — {SHORT[r['focal']]}" for _, r in tested.iterrows()]
    tested = tested.iloc[::-1]
    y = np.arange(len(tested))
    lifts = tested["lift"].replace(np.inf, 60.0).to_numpy()
    cols = [SUPPORT if (l > 1 and p < .05) else AGAINST if (l < 1 and p < .05) else NULLC
            for l, p in zip(tested["lift"], tested["p_holm"])]
    ax.barh(y, lifts, color=cols, height=0.6)
    ax.axvline(1.0, color="black", lw=1.0)
    ax.set_xscale("log"); ax.set_xlim(0.1, 400)
    ax.set_yticks(y, tested["lab"], fontsize=8.5)
    for yi, r in zip(y, tested.itertuples()):
        txt = "only model to\nallocate to it" if np.isinf(r.lift) else f"{r.lift:.2f}×"
        ax.text(lifts[yi] * 1.2, yi, txt, va="center", fontsize=7.4)
    ax.set_xlabel("allocated share, focal model ÷ other three  (log scale)")
    ax.set_title("A. Affiliation lift, before confound control")
    ax.legend(handles=[Patch(facecolor=SUPPORT, label="over-weighted (Holm $p<.05$)"),
                       Patch(facecolor=AGAINST, label="under-weighted (Holm $p<.05$)"),
                       Patch(facecolor=NULLC,  label="not distinguishable")],
              fontsize=7.4, loc="upper right")

    # --- panel B: BTC share within list-length strata ---------------------------
    ax = axes[1]
    ks = strat["k"].to_numpy(); width = 0.2
    for i, m in enumerate(MODELS):
        ax.bar(np.arange(len(ks)) + (i - 1.5) * width, strat[m] * 100, width,
               color=MODEL_COLORS[m], label=SHORT[m])
    ax.set_xticks(np.arange(len(ks)), [int(k) for k in ks])
    ax.set_xlabel("number of tokens named in the response")
    ax.set_ylabel("allocated share of budget to BTC (%)")
    ax.set_title("B. Bitcoin: the over-weight is list length")
    ax.legend(fontsize=7.5, ncol=2, loc="upper right")

    # --- panel C: DOGE ----------------------------------------------------------
    ax = axes2[0]
    doge_raw = [strat_store["DOGE"][2][m] * 100 for m in MODELS]
    doge_std = [strat_store["DOGE"][1][m] * 100 for m in MODELS]
    xx = np.arange(4)
    ax.bar(xx - 0.19, doge_raw, 0.36, color=[MODEL_COLORS[m] for m in MODELS],
           alpha=0.45, label="raw")
    ax.bar(xx + 0.19, doge_std, 0.36, color=[MODEL_COLORS[m] for m in MODELS],
           label="length-standardised")
    for i, (a, b) in enumerate(zip(doge_raw, doge_std)):
        ax.text(i - 0.19, a + 0.015, f"{a:.2f}", ha="center", fontsize=7.2)
        ax.text(i + 0.19, b + 0.015, f"{b:.2f}", ha="center", fontsize=7.2,
                fontweight="bold")
    ax.set_xticks(xx, [SHORT[m] for m in MODELS], fontsize=8.5)
    ax.set_ylabel("allocated share of budget to DOGE (%)")
    ax.set_title("A. Dogecoin: the one signal that survives")
    ax.legend(handles=[Patch(facecolor="grey", alpha=0.45, label="raw"),
                       Patch(facecolor="grey", label="length-standardised")],
              fontsize=7.5, loc="upper left")

    # --- panel D: placebo distribution ------------------------------------------
    ax = axes2[1]
    hl = {"Grok 4.6": ["BTC", "DOGE"], "Gemini 3.6 Flash": ["SOL"]}
    for i, m in enumerate(MODELS):
        d = ranks[(ranks["scenario"] == "tokens") & (ranks["model"] == m)]
        jit = RNG.normal(0, 0.055, len(d))
        ax.scatter(np.full(len(d), i) + jit, d["diff"] * 100, s=13,
                   color="lightgrey", edgecolor="none", zorder=2)
        for code in hl.get(m, []):
            q = d[d["code"] == code]
            ax.scatter([i], q["diff"] * 100, s=70, color=MODEL_COLORS[m],
                       edgecolor="black", linewidth=0.7, zorder=4)
            ax.annotate(f"{code} (rank {int(q['rank'].iloc[0])}/47)",
                        (i, q["diff"].iloc[0] * 100), textcoords="offset points",
                        xytext=(9, -1), fontsize=7.5, fontweight="bold")
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(range(4), [SHORT[m] for m in MODELS], fontsize=8.5)
    ax.set_xlim(-0.5, 4.55)
    ax.set_ylabel("over-weight vs. the other three (pp)")
    ax.set_title("B. All 47 tokens, raw; affiliated assets marked")

    fig.suptitle("Provider affiliation: the bitcoin result",
                 fontsize=11.5, fontweight="bold", y=1.02)
    fig.savefig(out / "figures" / "fig8_affiliation.png", dpi=220)
    fig.savefig(out / "figures" / "fig8_affiliation.pdf")
    fig2.suptitle("Dogecoin, and where the affiliated assets rank",
                  fontsize=11.5, fontweight="bold", y=1.02)
    fig2.savefig(out / "figures" / "fig9_affiliation_doge.png", dpi=220)
    fig2.savefig(out / "figures" / "fig9_affiliation_doge.pdf")
    print(f"  wrote {out/'figures'/'fig8_affiliation.pdf'}")
    print(f"  wrote {out/'figures'/'fig9_affiliation_doge.pdf'}")
    print("=" * 78)


# ===========================================================================
#  5.  ENTRY POINT
# ===========================================================================
def main(argv=None) -> dict:
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--affiliation-only" in argv:
        i = argv.index("--affiliation-only")
        rest = [a for a in argv[i + 1:] if not a.startswith("-")]
        base = Path(rest[0]) if rest else OUT_ROOT
        dest = Path(rest[1]) if len(rest) > 1 else base / "affiliation"
        run_affiliation(base, dest)
        return {}

    panel = load_panel()
    summarise_panel(panel)
    db, model_order = panel.frame, panel.model_order

    results = {s: run_scenario(s, db, model_order)
               for s in SCENARIOS if s in set(db["scenario"])}

    if len(results) > 1:
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        fig_dir = OUT_ROOT / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for scen, r in results.items():
            g = r["gini"].copy()
            g["scenario"] = scen
            rows.append(g[["scenario", "model", "n_responses", "n_products",
                           "n_products_union", "gini_support",
                           "GI_amount_paper", "GI_amount_pygini", "GI_freq_paper",
                           "GI_freq_pygini", "GI_amount_own", "GI_freq_own",
                           "top1_amount_share", "top3_amount_share", "HHI_amount",
                           "effective_n_amount"]])
        cross = pd.concat(rows, ignore_index=True)
        cross.to_csv(OUT_ROOT / "cross_scenario_gini.csv", index=False, encoding="utf-8-sig")

        print("\n" + "=" * 74)
        print("CROSS-SCENARIO SUMMARY")
        print("=" * 74)
        print(cross.round(4).to_string(index=False))
        print("\nGI(amount) mean by scenario:")
        print(cross.groupby("scenario")[["GI_amount_paper", "GI_freq_paper",
                                         "n_products", "top1_amount_share"]]
              .mean().round(3).to_string())

        sub = cross.dropna(subset=["GI_amount_paper", "GI_freq_paper"])
        if len(sub) > 2:
            r = float(np.corrcoef(sub["GI_amount_paper"], sub["GI_freq_paper"])[0, 1])
            print(f"\nPearson r between GI(amount) and GI(frequency) across "
                  f"model x scenario cells: {r:.3f}  (paper reports 0.85)")

        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))
        scen_list = list(results)
        x = np.arange(len(model_order))
        w = 0.8 / len(scen_list)
        for j, scen in enumerate(scen_list):
            s = cross[cross["scenario"] == scen].set_index("model").reindex(model_order)
            axes[0].bar(x + j * w - 0.4 + w / 2, s["GI_amount_paper"], width=w,
                        color=PALETTE[j], label=scen)
            axes[1].bar(x + j * w - 0.4 + w / 2, s["GI_freq_paper"], width=w,
                        color=PALETTE[j], label=scen)
            axes[2].bar(x + j * w - 0.4 + w / 2, s["n_products"], width=w,
                        color=PALETTE[j], label=scen)
        for ax, t, yl in zip(axes,
                             ["GI - investment amount", "GI - recommendation frequency",
                              "Distinct products recommended"],
                             ["Gini index", "Gini index", "count"]):
            ax.set_xticks(x)
            ax.set_xticklabels([_short(m) for m in model_order], rotation=20, ha="right")
            ax.set_title(t)
            ax.set_ylabel(yl)
            ax.grid(axis="x", visible=False)
        axes[0].set_ylim(0, 1)
        axes[1].set_ylim(0, 1)
        axes[0].legend(fontsize=8)
        fig.suptitle("Tokens vs exchanges: product bias compared",
                     fontsize=11, fontweight="bold", y=1.03)
        fig.tight_layout()
        fig.savefig(fig_dir / "fig7_scenario_comparison.png", dpi=220)
        if SAVE_PDF:
            fig.savefig(fig_dir / "fig7_scenario_comparison.pdf")
        plt.close(fig)
        print(f"  figures/fig7_scenario_comparison.png")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    panel.merge_log.to_csv(OUT_ROOT / "data_sources.csv", index=False, encoding="utf-8-sig")
    print(f"\n  data_sources.csv  ({len(panel.sources)} source db(s))")

    print(f"\ndone -> {OUT_ROOT}")

    if "--no-affiliation" not in argv:
        run_affiliation(OUT_ROOT, OUT_ROOT / "affiliation")

    return results


if __name__ == "__main__":
    main()
