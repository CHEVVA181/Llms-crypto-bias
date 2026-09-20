"""
====================================================================
 TEST SUITE FOR preprocessing.py  -  everything before the statistics
====================================================================
    python tests/test_preprocessing.py

 1 panel      the responses load, de-duplicate and label correctly
 2 registry   every product has a code, a category and no alias collision
 3 parsing    what parse_amount reads, and what it must REFUSE to read
 4 units      "40%" is a share of the money, never 40 CHF

Exit code 0 when every check passes, 1 otherwise.
"""
import os
import sys
import time

from fixture import N_CELLS, N_DUPLICATES, OUT, SRC, Report, build_fixture, section

sys.path.insert(0, str(SRC))

FIXTURE_DB = build_fixture(OUT / "fixture_preprocessing.db", with_duplicates=True)
os.environ["CRYPTO_BIAS_DB"] = str(FIXTURE_DB)
os.environ["CRYPTO_BIAS_OUT"] = str(OUT / "preprocessing")
os.environ["CRYPTO_BIAS_EXTRA_DBS"] = "off"

import numpy as np                                                  # noqa: E402
import preprocessing as pp                                          # noqa: E402

R = Report()


# ====================================================================
def s_panel():
    section("[1/4]  PANEL   reading, de-duplicating and labelling the responses")
    panel = pp.load_panel()
    db = panel.frame

    R.check(len(db) == N_CELLS, "one row per collected cell", len(db), N_CELLS)
    R.check(panel.n_duplicates == N_DUPLICATES,
            "a cell collected twice is kept once", panel.n_duplicates, N_DUPLICATES)
    R.check("Dogecoin" not in " ".join(db["response"]),
            "the dropped duplicate is the later one, not the original")
    R.check(panel.model_order == ["GPT-5.5", "Grok 4.6"],
            "models are labelled and ordered", panel.model_order,
            ["GPT-5.5", "Grok 4.6"])
    R.check(set(db["scenario"]) == {"tokens", "exchanges"},
            "both scenarios survive the load")

    # the prompt attributes are read back out of the prompt text
    budget = db.loc[db["prompt_idx"] == "0", "budget_chf"]
    R.near(budget.iloc[0], 10000, "10'000 CHF in the prompt -> 10000.0")
    R.check(db.loc[db["prompt_idx"] == "1", "budget_chf"].isna().all(),
            "no budget in the prompt -> no budget invented")
    R.check(set(db["risk_value"].dropna()) == {"risk-neutral"},
            "the risk attribute is recovered from the prompt")
    n_budget = int(db["attrs"].str.contains("budget").sum())
    R.check(int(db["has_budget"].sum()) == n_budget and db["has_risk"].all(),
            "the attribute flags follow the `variables` column",
            int(db["has_budget"].sum()), n_budget)
    R.check(db["prompt_key"].str.contains("#").all(),
            "every row carries a condition#index prompt key")
    return panel


def s_registry():
    section("[2/4]  REGISTRY   codes, categories and surface forms")
    for scen in ("tokens", "exchanges"):
        cfg = pp.SCENARIO_CFG[scen]
        reg, cat = cfg["registry"], cfg["category"]
        aliases = [a for _, (_c, al) in reg.items() for a in al]
        codes = [c for _, (c, _al) in reg.items()]

        R.check(not set(reg) - set(cat), f"[{scen}] every product has a category",
                sorted(set(reg) - set(cat)) or "none", "none")
        R.check(set(cat.values()) <= set(cfg["category_order"]),
                f"[{scen}] every category appears in the plotting order")
        R.check(len(set(codes)) == len(codes),
                f"[{scen}] no two products share a short code",
                len(codes) - len(set(codes)), 0)
        R.check(len(set(aliases)) == len(aliases),
                f"[{scen}] no surface form maps to two products",
                len(aliases) - len(set(aliases)), 0)
        R.check(all(a == a.lower().strip() for a in aliases),
                f"[{scen}] surface forms are stored lowercase and trimmed")
        R.check(set(cfg["tier_by_category"]) >= set(cat.values()),
                f"[{scen}] every category maps to a tier")

    R.check(pp.TOKEN_REGISTRY["Bitcoin"][0] == "BTC"
            and "btc" in pp.TOKEN_REGISTRY["Bitcoin"][1],
            "Bitcoin carries the BTC code and the bare ticker")
    R.check("kraken pro" in pp.EXCHANGE_REGISTRY["Kraken"][1],
            "a sub-brand is a surface form of its parent venue")
    R.check(pp.BLUE_CHIPS == {"Bitcoin", "Ethereum"},
            "the blue-chip set is the two majors", sorted(pp.BLUE_CHIPS),
            ["Bitcoin", "Ethereum"])


def s_parsing():
    section("[3/4]  PARSING   what is a number, and what is NOT money")
    for text, want in [("1'500", 1500), ("1,500", 1500), ("12,5", 12.5),
                       ("1.000,50", 1000.5), ("1,000.50", 1000.5)]:
        R.near(pp._to_float(text), want, f"_to_float({text!r})")
    for text in ("abc", "40 60"):        # gluing "40 60" into 4060 invents data
        R.check(pp._to_float(text) is None, f"_to_float({text!r}) is unreadable")

    for text, v, k, cur in [("40%", 40, "percent", None),
                            ("40% of the portfolio", 40, "percent", None),
                            ("CHF 4'000.-", 4000, "currency", "CHF"),
                            ("$6,000", 6000, "currency", "USD"),
                            ("€500", 500, "currency", "EUR"),
                            ("4000", 4000, "unitless", None),
                            ("5000-7000 CHF", 6000, "currency", "CHF")]:
        gv, gk, _n, gc = pp.parse_amount(text, with_currency=True)
        R.check(gv is not None and abs(gv - v) < 1e-6 and gk == k and gc == cur,
                f"parse_amount({text!r})", f"{gv} {gk} {gc}", f"{v} {k} {cur}")

    for text in ("N/A", "-", "", "0.05 BTC", "3 years", "1/3 of portfolio",
                 "crypto investments under 1 year carry extreme volatility"):
        R.check(pp.parse_amount(text)[0] is None, f"NOT money: {text[:40]!r}")

    # line splitting and name cleaning feed the registry lookup
    R.check(pp.clean_name("**1. Bitcoin (BTC):**") == "Bitcoin (BTC)",
            "clean_name strips markdown, numbering and trailing punctuation",
            pp.clean_name("**1. Bitcoin (BTC):**"), "Bitcoin (BTC)")
    for line, head in [("Bitcoin: 40%", "Bitcoin"),
                       ("| Bitcoin | 40% |", "Bitcoin"),
                       ("Bitcoin - 40%", "Bitcoin")]:
        got = pp.clean_name(pp.split_line(line)[0])
        R.check(got == head, f"split_line({line!r}) finds the product", got, head)

    R.check(bool(pp.REFUSAL_RE.search("I cannot provide personalized advice")),
            "a refusal is recognised as a refusal")
    R.check(bool(pp.NON_PRODUCT_RE.search("Your risk tolerance")),
            "boilerplate is recognised as a non-product line")
    R.check(not pp.NON_PRODUCT_RE.search("Bitcoin"),
            "a real product is not mistaken for boilerplate")


def s_units():
    section('[4/4]  UNITS   "40%" is 40% of the money to invest, never 40 CHF')
    money, weight, _u, basis, pct, _c = pp.resolve_amounts([40., 35., 25.],
                                                           ["percent"] * 3, 10000.)
    R.near(money[0], 4000, "40% of a 10'000 CHF budget -> 4000 CHF")
    R.near(pct, 100, "the percentages add up to 100")
    R.near(weight.sum(), 10000, "the allocation adds up to the budget")
    R.check(basis == "prompt", "basis is the budget from the prompt", basis, "prompt")

    money, weight, _u, basis, _p, _c = pp.resolve_amounts([60., 40.], ["percent"] * 2,
                                                          float("nan"))
    R.check(basis == "proportional" and bool(np.isnan(money).all()),
            "no budget -> proportional basis, no CHF value invented", basis,
            "proportional")
    R.near(weight[0] / weight.sum(), 0.6, "the 60/40 split survives intact")

    _m, weight, _u, basis, _p, _c = pp.resolve_amounts([50., 5000.],
                                                       ["percent", "currency"],
                                                       float("nan"))
    R.check(basis == "implied", "% next to CHF, no budget -> implied pot", basis,
            "implied")
    R.near(weight[0] / weight.sum(), 0.5, "the mixed answer is a 50/50 split")

    for vals, budget, want in [([60., 40.], float("nan"), "percent"),
                               ([6000., 4000.], 10000., "currency")]:
        got = pp.resolve_amounts(vals, ["unitless"] * 2, budget)[2][0]
        R.check(got == want, f"bare {vals[0]:g} -> {want}", got, want)
    R.check(pp.resolve_amounts([None, None], [None, None], 10000.)[3] == "none",
            "no numbers at all -> basis 'none'")


def main():
    print("=" * 78)
    print(f" preprocessing.py  -  TEST SUITE     {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 78)
    s_panel()
    s_registry()
    s_parsing()
    s_units()
    return R.summary("preprocessing.py")


if __name__ == "__main__":
    sys.exit(main())
