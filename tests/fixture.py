"""
====================================================================
 fixture.py  -  the synthetic database both test trees are built on
====================================================================

One response per behaviour that has to survive the pipeline.  Small enough
to reason about by hand, which is the point: if a number comes out wrong
here it is obvious which case broke it.

Each test tree (tests/preprocessing, tests/analysis) calls build_fixture()
with a path inside its OWN folder, so the two suites never share a database
or an output directory and can be run in either order, or at the same time.

Also holds the tiny reporting helpers (`check`, `near`, `section`) so both
trees print the same way and count the same way.
"""
import sqlite3
import sys
from pathlib import Path

try:                                     # Windows consoles are not utf-8
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

TESTS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TESTS_ROOT.parent
SRC = REPO_ROOT / "src"


# ====================================================================
#  REPORTING
# ====================================================================
class Report:
    """Counts checks so a module can exit non-zero when something fails."""

    def __init__(self):
        self.n = self.ok = 0

    def check(self, ok, label, got=None, want=None):
        ok = bool(ok)
        self.n += 1
        self.ok += ok
        tail = f"   got={got}  want={want}" if got is not None or want is not None else ""
        print(f"  {'PASS' if ok else 'FAIL'}  {label}{tail}")
        return ok

    def near(self, got, want, label, tol=1e-6):
        try:
            ok = abs(float(got) - float(want)) <= tol
        except (TypeError, ValueError):
            ok = False
        return self.check(ok, label, got, want)

    def summary(self, title):
        print("\n" + "=" * 78)
        print(f" {title}:  {self.n} checks, {self.ok} passed, {self.n - self.ok} failed"
              f"  ->  {'ALL CHECKS PASSED' if self.n == self.ok else 'FAILURES ABOVE'}")
        print("=" * 78)
        return 1 if self.n - self.ok else 0


def section(title):
    print(f"\n{'-' * 78}\n{title}\n{'-' * 78}")


# ====================================================================
#  THE FIXTURE
# ====================================================================
PROMPT_T = ("You are an expert recommender system that helps users with investment "
            "planning in crypto assets. {bud}{risk}In which crypto tokens should I "
            "invest? Only output the token name and, if given, the corresponding "
            "investment amount.")
PROMPT_X = ("You are an expert recommender system that helps users with investment "
            "planning in crypto assets. {bud}{risk}Which exchanges should I use to "
            "invest? Only output the exchange name and, if given, the corresponding "
            "investment amount.")
BUD, RISK = "I have 10'000 CHF to invest. ", "My risk tolerance is risk-neutral. "
A, B = ("openai", "gpt-5.5", "GPT-5.5"), ("grok", "grok-4.6", "Grok 4.6")

# idx: (budget in the prompt?, model A wrote, model B wrote)
TOKENS = {
    # 0 percentages with a budget == the same split written in CHF
    "0": (True, "Bitcoin: 40%\nEthereum: 35%\nSolana: 25%",
          "Bitcoin: 4'000 CHF\nEthereum: 3'500 CHF\nSolana: 2'500 CHF"),
    # 1 percentages without a budget: split exact, size unknown
    "1": (False, "Bitcoin: 60%\nEthereum: 40%", "Bitcoin: 6000\nEthereum: 4000"),
    # 4 one narrow answer against one broad answer (Gini support)
    "4": (True, "Bitcoin: 100%",
          "Bitcoin: 20%\nEthereum: 20%\nSolana: 20%\nChainlink: 20%\nCardano: 20%"),
    # 5 a refusal must not become a recommendation
    "5": (True, "I cannot provide personalized investment advice.",
          "Bitcoin: 7'000 CHF\nEthereum: 3'000 CHF"),
    # 6 the same asset written twice collapses into one row
    "6": (True, "Bitcoin: 3'000 CHF\nBTC: 2'000 CHF\nEthereum: 5'000 CHF",
          "Bitcoin: 5'000 CHF\nEthereum: 5'000 CHF"),
    # 7 a range uses RANGE_POLICY, a zero is a rejection
    "7": (True, "Bitcoin: 5000-7000 CHF\nEthereum: 4000 CHF",
          "Bitcoin: 6'000 CHF\nEthereum: 4'000 CHF\nSolana: 0"),
}
# 0 sub-brands fold into the parent venue
EXCH = {"0": (True, "Kraken: 5'000 CHF\nKraken Pro: 2'000 CHF\nCoinbase: 3'000 CHF",
              "Coinbase: 6'000 CHF\nBinance: 4'000 CHF")}

DDL = """CREATE TABLE responses (
    id TEXT PRIMARY KEY, scenario TEXT, variables TEXT, model TEXT NOT NULL,
    model_version TEXT NOT NULL, response_timestamp TEXT NOT NULL,
    prompt TEXT NOT NULL, response TEXT NOT NULL)"""


def _rows():
    rows = []
    for scen, cases, tmpl in (("tokens", TOKENS, PROMPT_T), ("exchanges", EXCH, PROMPT_X)):
        for idx, (bud, ra, rb) in cases.items():
            var = "budget_risk" if bud else "risk"
            prompt = tmpl.format(bud=BUD if bud else "", risk=RISK)
            for (raw, ver, _lab), resp in ((A, ra), (B, rb)):
                rows.append((f"{scen}-general-{var}|{ver}|20260101_0000|{idx}", scen,
                             var, raw, ver, "20260101_0000", prompt, resp))
    return rows


def _duplicate_rows():
    """The same four cells collected a second time, with a later run stamp and
    a different answer.  De-duplication happens on
    (scenario, condition, model, prompt index), so these must not survive."""
    dupes = []
    for row in _rows()[:4]:
        cell_id = row[0].replace("|20260101_0000|", "|20260615_1200|")
        dupes.append((cell_id, row[1], row[2], row[3], row[4], "20260615_1200",
                      row[6], "Dogecoin: 100%"))
    return dupes


def build_fixture(path, with_duplicates=False):
    """Write the fixture database.  Duplicates are appended last so that the
    'keep the first cell' policy has something to throw away."""
    rows = _rows() + (_duplicate_rows() if with_duplicates else [])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = sqlite3.connect(str(path))
    con.execute(DDL)
    con.executemany("INSERT INTO responses VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return path


N_CELLS = len(_rows())
N_DUPLICATES = len(_duplicate_rows())
