"""
====================================================================
 TEST SUITE FOR analysis.py  -  the checks that catch a silent wrong result
====================================================================
    python tests/analysis/test_analysis.py

 1 measures   Gini properties, the pygini cross-check, HHI, rank weights
 2 support    own-list vs. union support, and why the union is the headline
 3 run        the pipeline runs end to end on a small synthetic DB
 4 output     what the pipeline made of the fixture

Exit code 0 when every check passes, 1 otherwise.
"""
import io
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))            # tests/ - the shared fixture

from fixture import A, B, ANALYSIS_DIR, Report, build_fixture, section

sys.path.insert(0, str(ANALYSIS_DIR))

# This tree keeps its fixture and its output to itself.
OUT = HERE / "test_output"
FIXTURE_OUT = OUT / "run"
FIXTURE_DB = build_fixture(OUT / "fixture.db")
os.environ["CRYPTO_BIAS_DB"] = str(FIXTURE_DB)
os.environ["CRYPTO_BIAS_OUT"] = str(FIXTURE_OUT)
os.environ["CRYPTO_BIAS_EXTRA_DBS"] = "off"

import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402
import analysis as an                                               # noqa: E402

R = Report()


# ====================================================================
def s_measures():
    section("[1/4]  MEASURES   Gini properties, the package cross-check, HHI")
    R.near(an.gini_paper([1, 1, 1, 1]), 0.0, "a perfectly equal split scores 0")
    R.near(an.gini_paper([1, 0, 0, 0]), 0.75, "one winner out of 4 scores (n-1)/n")
    R.near(an.gini_paper([3, 1, 7]), an.gini_paper([30, 10, 70]),
           "scale and order do not matter")
    R.check(np.isnan(an.gini_paper([])), "an empty vector has no Gini")
    R.near(an.gini_paper([0, 0, 0]), 0.0, "an all-zero vector scores 0, not NaN")

    if an.HAVE_PYGINI:
        rng = np.random.default_rng(7)
        worst = max(abs(an.gini_paper(v) - an.gini_pygini(v)) for v in
                    (rng.random(int(rng.integers(2, 60))) * 1000 for _ in range(200)))
        R.check(worst < 1e-6, "paper formula == pygini over 200 random vectors",
                f"{worst:.2e}", "< 1e-6")
    else:
        print("  SKIP  pygini is not installed - cross-check skipped")

    R.near(an.hhi([1.0]), 1.0, "one product taking everything has HHI 1")
    R.near(an.hhi([0.25] * 4), 0.25, "four equal products have HHI 1/n")
    R.check(an.hhi([0.5, 0.5]) > an.hhi([0.25] * 4),
            "HHI rises as the allocation concentrates")

    w = an.rank_weights(4)
    R.near(float(w.sum()), 1.0, "rank weights sum to 1")
    R.check(bool(np.all(np.diff(w) < 0)),
            "an earlier rank carries more weight than a later one")


def s_support():
    section("[2/4]  SUPPORT   scoring a model on its own list flips the ranking")
    # this is the reason the headline Gini is computed over the union of all
    # products any model named, zero-filled per model
    narrow, union = pd.Series({"BTC": .6, "ETH": .4}), ["BTC", "ETH", "SOL", "LINK"]
    broad = pd.Series({"BTC": .4, "ETH": .3, "SOL": .2, "LINK": .1})
    n_own, b_own = an.gini_paper(narrow.to_numpy()), an.gini_paper(broad.to_numpy())
    n_uni = an.gini_paper(narrow.reindex(union, fill_value=0).to_numpy(float))
    b_uni = an.gini_paper(broad.reindex(union, fill_value=0).to_numpy(float))
    R.check(n_own < b_own and n_uni > b_uni,
            "own support calls the narrow model less concentrated, union more",
            f"own {n_own:.3f}<{b_own:.3f}, union {n_uni:.3f}>{b_uni:.3f}", "flip")
    R.check(an.GINI_SUPPORT == "union", "the headline support is the union",
            an.GINI_SUPPORT, "union")


def s_run():
    section("[3/4]  FIXTURE RUN   analysis.py runs end to end")
    buf, t0 = io.StringIO(), time.time()
    try:
        with redirect_stdout(buf):
            results = an.main(["--no-affiliation"])
    except Exception:
        R.check(False, "the pipeline runs start to finish without raising")
        print(traceback.format_exc(), buf.getvalue()[-2000:])
        return None
    R.check(True, "the pipeline runs start to finish without raising",
            f"{time.time() - t0:.1f}s", "no exception")
    R.check(set(results) == {"tokens", "exchanges"}, "both scenarios were analysed")
    miss = [f"{s}/{n}" for s in ("tokens", "exchanges")
            for n in ("parsed_recommendations.csv", "response_status.csv",
                      "gini_by_model.csv", "gini_verification.csv")
            if not (FIXTURE_OUT / s / n).exists()]
    R.check(not miss, "the core CSVs were written", ", ".join(miss) or "all there", "all")
    return results


def s_output(results):
    section("[4/4]  OUTPUT   what the pipeline made of the fixture")
    parsed = results["tokens"]["parsed"].copy()
    status = results["tokens"]["status"].copy()
    for df in (parsed, status):
        df["case"] = df["response_uid"].astype(str).str.rsplit("|", n=1).str[-1]
    mA, mB = A[2], B[2]

    def case(idx, model):
        return parsed[(parsed["case"] == str(idx)) & (parsed["model"] == model)]

    a = case("0", mA).set_index("product")["share"]
    b = case("0", mB).set_index("product")["share"]
    R.check(len(a) == 3 and (a - b.reindex(a.index)).abs().max() < 1e-12,
            "40/35/25 % == 4000/3500/2500 CHF, share for share")
    pct = parsed[(parsed["amount_unit"] == "percent") & parsed["budget_chf"].notna()]
    err = (pct["amount_chf"] - pct["amount_reported"] / 100 * pct["budget_chf"]).abs()
    R.check(len(pct) > 0 and err.max() < 1e-6,
            "every percentage became its share of the budget in CHF",
            f"{err.max():.1e}", "< 1e-6")
    c1 = case("1", mA)
    R.check(set(c1["unit_basis"]) == {"proportional"} and c1["amount_chf"].isna().all(),
            "no budget -> proportional basis, no CHF value invented")
    R.check(list(status[(status["case"] == "5") & (status["model"] == mA)]["status"])
            == ["refusal"] and len(case("5", mA)) == 0,
            "a refusal is labelled a refusal and yields no product")
    c6 = case("6", mA).set_index("product")
    R.check(len(c6) == 2 and int(c6.loc["Bitcoin", "n_lines"]) == 2
            and abs(c6.loc["Bitcoin", "amount_chf"] - 5000) < 1e-6,
            "Bitcoin and BTC collapsed into one row and their amounts added")
    R.near(case("7", mA).set_index("product").loc["Bitcoin", "amount_chf"], 6000,
           f"5000-7000 CHF -> {an.RANGE_POLICY} = 6000")
    R.check("Solana" not in set(case("7", mB)["product"]),
            "a zero allocation is dropped as a rejection")

    sums = parsed.groupby("response_id")["share"].sum()
    R.check(np.allclose(sums, 1.0) and parsed["share"].between(0, 1).all(),
            "every response's shares sum to 1 and sit inside [0, 1]",
            f"{sums.min():.6f}..{sums.max():.6f}", "1.0")
    cur = parsed[parsed["amount_unit"] == "currency"]
    R.check(((cur["amount_chf"] - cur["amount_reported"]).abs() < 1e-9).all(),
            "a CHF amount is carried through unchanged")

    # the exchange scenario folds sub-brands into the parent venue
    xparsed = results["exchanges"]["parsed"]
    kraken = xparsed[(xparsed["model"] == A[2]) & (xparsed["product"] == "Kraken")]
    R.check(len(kraken) == 1 and abs(float(kraken["amount_chf"].iloc[0]) - 7000) < 1e-6,
            "Kraken and Kraken Pro are one venue worth 7000 CHF",
            float(kraken["amount_chf"].iloc[0]) if len(kraken) else None, 7000)

    g = results["tokens"]["gini"]
    R.check((g["GI_amount_union"] >= g["GI_amount_own"] - 1e-12).all()
            and (g["n_products"] <= g["n_products_union"]).all(),
            "union GI >= own GI, and every model's list fits inside the union")


def main():
    print("=" * 78)
    print(f" analysis.py  -  TEST SUITE     {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)
    s_measures()
    s_support()
    results = s_run()
    if results is None:
        print("\nthe pipeline could not be run - the output section was skipped")
    else:
        s_output(results)
    return R.summary("analysis.py")


if __name__ == "__main__":
    sys.exit(main())
