"""
===========================================================================
 preprocessing.py

 Everything that happens to a raw model response before any statistic is
 computed: reading the response databases, de-duplicating the panel,
 recovering the prompt attributes, mapping free-text product names onto a
 canonical registry, and turning "40%", "CHF 4'000.-" or "5000-7000" into a
 number with a unit attached.

 Nothing in this module measures concentration or draws anything - that is
 analysis.py.  Import it:

     from preprocessing import load_panel, parse_amount, SCENARIO_CFG
===========================================================================
"""
import os
import re
import sys
import sqlite3
import warnings
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:                                     # Windows consoles are not utf-8
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 60)

# Response database. Override with the CRYPTO_BIAS_DB environment variable;
# the default is <repo>/data/responses.db.
REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(os.environ.get("CRYPTO_BIAS_DB",
                              str(REPO_ROOT / "data" / "responses.db")))

_extra_env = os.environ.get("CRYPTO_BIAS_EXTRA_DBS")
if _extra_env is None:
    EXTRA_DB_PATHS = sorted(DB_PATH.parent.glob("responses-missing*.db"))
elif _extra_env.strip().lower() in {"", "none", "0", "off"}:
    EXTRA_DB_PATHS = []
else:
    EXTRA_DB_PATHS = [Path(x) for x in _extra_env.split(os.pathsep) if x.strip()]
EXTRA_DB_PATHS = [p for p in EXTRA_DB_PATHS if p.resolve() != DB_PATH.resolve()]

DUPLICATE_POLICY = "primary"

OUT_ROOT = Path(os.environ.get("CRYPTO_BIAS_OUT",
                               str(DB_PATH.parent / "crypto_bias_output")))

ZERO_AMOUNT_AS_REJECTION = True
RANK_WEIGHTING = "linear"
RANGE_POLICY = "midpoint"

GINI_SUPPORT = "union"

UNITLESS_AS_PERCENT_MAX = 100.0
PCT_SUM_TOL = 15.0

LOOSE_MAX_WORDS_WITHOUT_UNIT = 6

FLAT_BUDGET_AS_ALTERNATIVES = True

SAVE_PDF = True
SCENARIOS = ("tokens", "exchanges")

# ===========================================================================
#  1.  READING THE RESPONSE DATABASES
# ===========================================================================
def read_responses(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"response DB not found: {path}")
    con = sqlite3.connect(str(path))
    try:
        frame = pd.read_sql("SELECT * FROM responses", con)
    finally:
        con.close()
    frame["source_db"] = Path(path).name
    return frame

def split_ids(frame: pd.DataFrame) -> pd.DataFrame:
    parts = frame["id"].astype(str).str.split("|", expand=True)
    frame["condition"] = parts[0]
    frame["model_raw"] = parts[1]
    frame["run_stamp"] = parts[2]
    frame["prompt_idx"] = parts[3]
    frame["prompt_key"] = frame["condition"] + "#" + frame["prompt_idx"]
    return frame

CELL_KEY = ["scenario", "condition", "model_raw", "prompt_idx"]

MODEL_LABELS = {
    "openai:gpt-5.5": "GPT-5.5",
    "gpt-5.5": "GPT-5.5",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "gemini-3.6-flash": "Gemini 3.6 Flash",
    "grok-4.6": "Grok 4.6",
}

ATTRIBUTES = ["budget", "risk", "term", "environment"]

# The four prompt attributes are written back out of the prompt text rather
# than trusted from the `variables` column, so a malformed id cannot silently
# mislabel a cell.
BUDGET_RE = re.compile(r"I have ([\d'’,\.]+)\s*CHF to invest", re.I)
RISK_RE = re.compile(r"My risk tolerance is ([a-z\- ]+?)\.", re.I)
TERM_RE = re.compile(r"My investment term is ([a-z\- ]+?)\.", re.I)
ENV_RE = re.compile(r"The market environment is ([a-z\- ]+?)\.", re.I)

def _budget_num(s):
    if s is None:
        return np.nan
    return float(re.sub(r"[^\d]", "", s))

BUDGET_ORDER = [100, 1000, 10000, 20000, 30000, 40000, 50000, 100000]
RISK_ORDER = ["risk-averse", "risk-neutral", "risk-seeking"]
TERM_ORDER = ["less than one year", "one to three years", "three to ten years"]
ENV_ORDER = ["crisis", "recession", "recovery", "expansion"]


@dataclass
class Panel:
    """The de-duplicated response panel plus the bookkeeping needed to say
    where every row came from."""
    frame: pd.DataFrame
    model_order: list
    merge_log: pd.DataFrame
    sources: list = field(default_factory=list)
    n_rows_read: int = 0
    n_duplicates: int = 0


def load_panel(db_path: Path = None, extra_db_paths=None) -> Panel:
    """Read the primary database plus any top-up databases, drop duplicate
    cells, label the models, and recover the prompt attributes.

    A cell is `(scenario, condition, model, prompt index)` rather than the raw
    id, so re-running a prompt that is already present gets dropped instead of
    duplicated.
    """
    db_path = Path(db_path) if db_path is not None else DB_PATH
    extras = list(EXTRA_DB_PATHS if extra_db_paths is None else extra_db_paths)
    extras = [Path(p) for p in extras if Path(p).resolve() != db_path.resolve()]

    sources = [db_path] + extras
    frames = [split_ids(read_responses(p)) for p in sources]
    db = pd.concat(frames, ignore_index=True)

    n_before = len(db)
    if DUPLICATE_POLICY == "latest":
        db = db.sort_values("run_stamp", kind="mergesort")
        db = db.drop_duplicates(subset=CELL_KEY, keep="last")
    else:
        db = db.drop_duplicates(subset=CELL_KEY, keep="first")
    db = db.sort_index().reset_index(drop=True)
    n_dupes = n_before - len(db)

    merge_log = (db.groupby(["source_db", "model_raw", "scenario"])
                   .size().rename("responses").reset_index())

    db["model"] = db["model_raw"].map(MODEL_LABELS).fillna(db["model_raw"])
    db["response"] = db["response"].fillna("")

    db["attrs"] = db["variables"].fillna("")
    for a in ATTRIBUTES:
        db[f"has_{a}"] = db["attrs"].str.contains(a, regex=False)

    db["budget_chf"] = db["prompt"].str.extract(BUDGET_RE)[0].map(
        lambda s: _budget_num(s) if isinstance(s, str) else np.nan)
    db["risk_value"] = db["prompt"].str.extract(RISK_RE)[0].str.strip().str.lower()
    db["term_value"] = db["prompt"].str.extract(TERM_RE)[0].str.strip().str.lower()
    db["env_value"] = db["prompt"].str.extract(ENV_RE)[0].str.strip().str.lower()

    model_order = [m for m in ["GPT-5.5", "Claude Haiku 4.5", "Gemini 3.6 Flash", "Grok 4.6"]
                   if m in set(db["model"])]
    model_order += [m for m in sorted(db["model"].unique().tolist())
                    if m not in model_order]

    return Panel(frame=db, model_order=model_order, merge_log=merge_log,
                 sources=sources, n_rows_read=n_before, n_duplicates=n_dupes)


def summarise_panel(panel: Panel, db_path: Path = None) -> None:
    """Print what was read, how the models are covered, and which prompt keys
    are still missing.  A missing key is dropped for every model, so it is
    worth seeing before any number is trusted."""
    db = panel.frame
    db_path = Path(db_path) if db_path is not None else DB_PATH
    extras = [p for p in panel.sources if Path(p) != db_path]

    print("=" * 74)
    print(f"DB: {db_path}")
    if extras:
        for p in extras:
            added = int((db["source_db"] == Path(p).name).sum())
            print(f"  + top-up: {Path(p).name}  ({added:,} new responses kept)")
        print(f"  duplicate policy: {DUPLICATE_POLICY}  "
              f"({panel.n_duplicates:,} duplicate cells dropped of "
              f"{panel.n_rows_read:,} rows read)")
    else:
        print("  + top-up: none (CRYPTO_BIAS_EXTRA_DBS disabled or no files found)")
    print(f"{len(db):,} responses | scenarios: "
          f"{sorted(db['scenario'].dropna().unique().tolist())}")
    print(pd.crosstab(db["model"], db["scenario"]).reindex(panel.model_order).to_string())
    print("attribute value coverage:")
    print(f"  budget      {int(db['budget_chf'].notna().sum()):>5,}  values "
          f"{sorted(db['budget_chf'].dropna().astype(int).unique().tolist())}")
    print(f"  risk        {int(db['risk_value'].notna().sum()):>5,}  values "
          f"{sorted(db['risk_value'].dropna().unique().tolist())}")
    print(f"  term        {int(db['term_value'].notna().sum()):>5,}  values "
          f"{sorted(db['term_value'].dropna().unique().tolist())}")
    print(f"  environment {int(db['env_value'].notna().sum()):>5,}  values "
          f"{sorted(db['env_value'].dropna().unique().tolist())}")

    cov = (db.pivot_table(index="prompt_key", columns="model", values="id",
                          aggfunc="size").notna())
    missing = (~cov).sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    if len(missing):
        print("prompt keys still missing per model "
              "(these prompts are dropped for all models):")
        for m, n in missing.items():
            print(f"  {m:<20} {int(n):>5,}")
    else:
        print("prompt coverage: complete - every model answered every prompt key")


# ===========================================================================
#  2.  THE PRODUCT REGISTRIES
#
#  Canonical name -> (short code, every surface form seen in a response).
#  51 tokens / 174 surface forms, 32 exchanges.  Sub-brands ("Kraken Pro")
#  fold into the parent venue so a model is not credited twice for one pick.
# ===========================================================================
TOKEN_REGISTRY = {
    "Bitcoin":            ("BTC",    ["bitcoin", "btc", "bitcoin (btc)", "xbt"]),
    "Ethereum":           ("ETH",    ["ethereum", "eth", "ether", "ethereum (eth)", "ether (eth)"]),
    "Solana":             ("SOL",    ["solana", "sol", "solana (sol)"]),
    "Chainlink":          ("LINK",   ["chainlink", "link", "chainlink (link)"]),
    "USD Coin":           ("USDC",   ["usd coin", "usdc", "usd coin (usdc)", "circle usdc"]),
    "Tether":             ("USDT",   ["tether", "usdt", "tether (usdt)"]),
    "Dai":                ("DAI",    ["dai", "dai (dai)", "makerdao dai"]),
    "Render":             ("RENDER", ["render", "render token", "rndr", "render (render)", "render (rndr)"]),
    "Avalanche":          ("AVAX",   ["avalanche", "avax", "avalanche (avax)"]),
    "Sui":                ("SUI",    ["sui", "sui (sui)"]),
    "Polkadot":           ("DOT",    ["polkadot", "dot", "polkadot (dot)"]),
    "NEAR Protocol":      ("NEAR",   ["near protocol", "near", "near protocol (near)", "near (near)"]),
    "Polygon":            ("POL",    ["polygon", "matic", "pol", "polygon (pol)", "polygon (matic)"]),
    "Bittensor":          ("TAO",    ["bittensor", "tao", "bittensor (tao)"]),
    "Cardano":            ("ADA",    ["cardano", "ada", "cardano (ada)"]),
    "Aave":               ("AAVE",   ["aave", "aave (aave)"]),
    "PAX Gold":           ("PAXG",   ["pax gold", "paxg", "pax gold (paxg)", "paxos gold", "paxos gold (paxg)"]),
    "Uniswap":            ("UNI",    ["uniswap", "uni", "uniswap (uni)"]),
    "Arbitrum":           ("ARB",    ["arbitrum", "arb", "arbitrum (arb)"]),
    "Pepe":               ("PEPE",   ["pepe", "pepe (pepe)", "pepe coin"]),
    "dogwifhat":          ("WIF",    ["dogwifhat", "wif", "dogwifhat (wif)", "dog wif hat"]),
    "Dogecoin":           ("DOGE",   ["dogecoin", "doge", "dogecoin (doge)"]),
    "Injective":          ("INJ",    ["injective", "inj", "injective (inj)"]),
    "Fetch.ai":           ("FET",    ["fetch.ai", "fet", "fetch ai", "fetch.ai (fet)",
                                      "artificial superintelligence alliance", "asi"]),
    "Pendle":             ("PENDLE", ["pendle", "pendle (pendle)"]),
    "Celestia":           ("TIA",    ["celestia", "tia", "celestia (tia)"]),
    "BNB":                ("BNB",    ["bnb", "binance coin", "bnb (bnb)"]),
    "Ondo Finance":       ("ONDO",   ["ondo", "ondo finance", "ondo (ondo)"]),
    "Optimism":           ("OP",     ["optimism", "op", "optimism (op)"]),
    "Bonk":               ("BONK",   ["bonk", "bonk (bonk)"]),
    "XRP":                ("XRP",    ["xrp", "ripple", "ripple (xrp)"]),
    "Monero":             ("XMR",    ["monero", "xmr", "monero (xmr)"]),
    "Kaspa":              ("KAS",    ["kaspa", "kas", "kaspa (kas)"]),
    "Aptos":              ("APT",    ["aptos", "apt", "aptos (apt)"]),
    "Shiba Inu":          ("SHIB",   ["shiba inu", "shib", "shiba inu (shib)"]),
    "Floki":              ("FLOKI",  ["floki", "floki inu", "floki (floki)"]),
    "Tether Gold":        ("XAUT",   ["tether gold", "xaut", "tether gold (xaut)"]),
    "CryptoFranc":        ("XCHF",   ["cryptofranc", "xchf", "cryptofranc (xchf)", "crypto franc"]),
    "VNX Swiss Franc":    ("VCHF",   ["vnx swiss franc", "vchf", "vnx chf", "vnx swiss franc (vchf)"]),
    "Litecoin":           ("LTC",    ["litecoin", "ltc", "litecoin (ltc)"]),
    "Cosmos":             ("ATOM",   ["cosmos", "atom", "cosmos (atom)"]),
    "Toncoin":            ("TON",    ["toncoin", "ton", "toncoin (ton)"]),
    "Hyperliquid":        ("HYPE",   ["hyperliquid", "hype", "hyperliquid (hype)"]),
    "Lido DAO":           ("LDO",    ["lido dao", "ldo", "lido", "lido dao (ldo)"]),
    "Worldcoin":          ("WLD",    ["worldcoin", "wld", "worldcoin (wld)"]),
    "Stellar":            ("XLM",    ["stellar", "xlm", "stellar (xlm)"]),
    "Fantom":             ("FTM",    ["fantom", "ftm", "fantom (ftm)", "sonic"]),
    "Immutable":          ("IMX",    ["immutable", "imx", "immutable x", "immutable (imx)"]),
    "Filecoin":           ("FIL",    ["filecoin", "fil", "filecoin (fil)"]),
    "Sei":                ("SEI",    ["sei", "sei (sei)"]),
    "Ethena":             ("ENA",    ["ethena", "ena", "ethena (ena)"]),
    "Maker":              ("MKR",    ["maker", "mkr", "makerdao", "maker (mkr)", "sky"]),
    "Jupiter":            ("JUP",    ["jupiter", "jup", "jupiter (jup)"]),
    "PayPal USD":         ("PYUSD",  ["paypal usd", "pyusd", "paypal usd (pyusd)"]),
    "Euro Coin":          ("EURC",   ["euro coin", "eurc", "euro coin (eurc)", "eur coin",
                                      "eur coin (eurc)", "circle euro coin"]),

    "Stablecoin (unspecified)": ("STABLE", [
        "stablecoin", "stablecoins", "stablecoin (unspecified)", "stable",
        "stablecoin (usdc/usdt)", "stablecoin (usdc or usdt)",
        "stablecoin (usdc)", "stablecoin (usdt)", "usdc/usdt", "stablecoin basket"]),
}

EXCHANGE_REGISTRY = {
    "Kraken":         ("KRAKEN", ["kraken", "kraken pro", "kraken futures", "kraken staking",
                                  "kraken (staking)", "kraken exchange", "kraken.com",
                                  "kraken futures (avoid)", "kraken (futures)",
                                  "kraken (for derivatives only, minimal allocation)"]),
    "Coinbase":       ("COINBASE", ["coinbase", "coinbase advanced", "coinbase pro",
                                    "coinbase exchange", "coinbase advanced trade",
                                    "coinbase one", "coinbase.com"]),
    "Bitstamp":       ("BITSTAMP", ["bitstamp", "bitstamp.net"]),
    "Binance":        ("BINANCE", ["binance", "binance.com", "binance global",
                                   "binance international"]),
    "Binance.US":     ("BINANCEUS", ["binance.us", "binance us", "binance-us"]),
    "Gemini":         ("GEMINI", ["gemini", "gemini exchange", "gemini activetrader",
                                  "gemini.com"]),
    "Swissquote":     ("SWISSQUOTE", ["swissquote", "swissquote bank", "swissquote crypto"]),
    "Bybit":          ("BYBIT", ["bybit", "bybit.com"]),
    "SwissBorg":      ("SWISSBORG", ["swissborg", "swiss borg", "swissborg app"]),
    "OKX":            ("OKX", ["okx", "okex", "okx.com"]),
    "Bitpanda":       ("BITPANDA", ["bitpanda", "bitpanda pro"]),
    "Bitcoin Suisse": ("BTCSUISSE", ["bitcoin suisse", "bitcoinsuisse", "bitcoin suisse ag"]),
    "KuCoin":         ("KUCOIN", ["kucoin", "ku coin"]),
    "MEXC":           ("MEXC", ["mexc", "mexc global"]),
    "Gate.io":        ("GATEIO", ["gate.io", "gate io", "gateio", "gate"]),
    "Crypto.com":     ("CRYPTOCOM", ["crypto.com", "crypto com", "cryptocom",
                                     "crypto.com exchange"]),
    "Sygnum":         ("SYGNUM", ["sygnum", "sygnum bank", "sygnum bank ag"]),
    "Bitget":         ("BITGET", ["bitget"]),
    "Hyperliquid":    ("HYPERLIQUID", ["hyperliquid"]),
    "Deribit":        ("DERIBIT", ["deribit"]),
    "Nexo":           ("NEXO", ["nexo"]),
    "Luno":           ("LUNO", ["luno"]),
    "Upbit":          ("UPBIT", ["upbit"]),
    "Uphold":         ("UPHOLD", ["uphold"]),
    "Bull Bitcoin":   ("BULLBTC", ["bullbitcoin", "bull bitcoin"]),
    "HTX":            ("HTX", ["htx", "huobi", "huobi global"]),
    "Bitvavo":        ("BITVAVO", ["bitvavo"]),
    "dYdX":           ("DYDX", ["dydx", "dy/dx"]),
    "Uniswap (DEX)":  ("UNISWAP", ["uniswap", "uniswap dex"]),
    "Fidelity Crypto": ("FIDELITY", ["fidelity crypto", "fidelity digital assets", "fidelity"]),
    "FTX":            ("FTX", ["ftx", "ftx.com", "ftx*"]),
}

TOKEN_CATEGORY = {
    "Bitcoin": "Store of Value (PoW)", "Kaspa": "Store of Value (PoW)",
    "Monero": "Store of Value (PoW)",
    "Ethereum": "Smart-Contract L1", "Solana": "Smart-Contract L1",
    "Cardano": "Smart-Contract L1", "Avalanche": "Smart-Contract L1",
    "Polkadot": "Smart-Contract L1", "NEAR Protocol": "Smart-Contract L1",
    "Sui": "Smart-Contract L1", "Aptos": "Smart-Contract L1",
    "Toncoin": "Smart-Contract L1", "Cosmos": "Smart-Contract L1",
    "Sei": "Smart-Contract L1", "Fantom": "Smart-Contract L1",
    "Injective": "Smart-Contract L1",
    "Polygon": "Layer 2 & Scaling", "Arbitrum": "Layer 2 & Scaling",
    "Optimism": "Layer 2 & Scaling", "Immutable": "Layer 2 & Scaling",
    "Uniswap": "DeFi", "Aave": "DeFi", "Lido DAO": "DeFi",
    "Pendle": "DeFi", "Hyperliquid": "DeFi", "Ethena": "DeFi",
    "Maker": "DeFi", "Jupiter": "DeFi",
    "Chainlink": "AI & Infrastructure", "Render": "AI & Infrastructure",
    "Bittensor": "AI & Infrastructure", "Fetch.ai": "AI & Infrastructure",
    "Filecoin": "AI & Infrastructure", "Celestia": "AI & Infrastructure",
    "Worldcoin": "AI & Infrastructure",
    "XRP": "Payments", "Stellar": "Payments", "Litecoin": "Payments",
    "BNB": "Exchange & Platform",
    "USD Coin": "Stablecoin", "Tether": "Stablecoin", "Dai": "Stablecoin",
    "CryptoFranc": "Stablecoin", "Stablecoin (unspecified)": "Stablecoin",
    "PayPal USD": "Stablecoin", "Euro Coin": "Stablecoin",
    "VNX Swiss Franc": "Stablecoin",
    "PAX Gold": "RWA & Tokenized", "Tether Gold": "RWA & Tokenized",
    "Ondo Finance": "RWA & Tokenized",
    "Dogecoin": "Meme", "Shiba Inu": "Meme", "Pepe": "Meme",
    "Bonk": "Meme", "dogwifhat": "Meme", "Floki": "Meme",
}

TOKEN_TIER_BY_CATEGORY = {
    "Stablecoin": "Stable", "RWA & Tokenized": "Stable",
    "Store of Value (PoW)": "Blue chip", "Smart-Contract L1": "Large-cap alt",
    "Exchange & Platform": "Large-cap alt", "Payments": "Large-cap alt",
    "Layer 2 & Scaling": "Small-cap alt", "DeFi": "Small-cap alt",
    "AI & Infrastructure": "Small-cap alt", "Meme": "Meme",
}
BLUE_CHIPS = {"Bitcoin", "Ethereum"}

TOKEN_CATEGORY_ORDER = ["Store of Value (PoW)", "Smart-Contract L1", "Layer 2 & Scaling",
                        "DeFi", "AI & Infrastructure", "Payments", "Exchange & Platform",
                        "RWA & Tokenized", "Stablecoin", "Meme"]
TOKEN_TIER_ORDER = ["Stable", "Blue chip", "Large-cap alt", "Small-cap alt", "Meme"]

EXCHANGE_CATEGORY = {
    "Coinbase": "US-listed CEX", "Kraken": "US-listed CEX",
    "Gemini": "US-listed CEX", "Binance.US": "US-listed CEX",
    "Fidelity Crypto": "US-listed CEX",
    "Binance": "Global CEX", "Bybit": "Global CEX", "OKX": "Global CEX",
    "KuCoin": "Global CEX", "MEXC": "Global CEX", "Gate.io": "Global CEX",
    "Bitget": "Global CEX", "Crypto.com": "Global CEX", "HTX": "Global CEX",
    "Upbit": "Global CEX",
    "Bitstamp": "EU-regulated CEX", "Bitpanda": "EU-regulated CEX",
    "Bitvavo": "EU-regulated CEX", "Luno": "EU-regulated CEX",
    "Uphold": "EU-regulated CEX",
    "Swissquote": "Swiss bank/broker", "SwissBorg": "Swiss bank/broker",
    "Bitcoin Suisse": "Swiss bank/broker", "Sygnum": "Swiss bank/broker",
    "Bull Bitcoin": "Non-custodial/brokerage", "Nexo": "CeFi lending",
    "Deribit": "Derivatives venue",
    "Hyperliquid": "On-chain / DEX", "dYdX": "On-chain / DEX",
    "Uniswap (DEX)": "On-chain / DEX",
    "FTX": "Defunct",
}
EXCHANGE_CATEGORY_ORDER = ["US-listed CEX", "EU-regulated CEX", "Swiss bank/broker",
                           "Global CEX", "Derivatives venue", "CeFi lending",
                           "Non-custodial/brokerage", "On-chain / DEX", "Defunct"]

EXCHANGE_TIER_BY_CATEGORY = {
    "Swiss bank/broker": "Swiss regulated",
    "EU-regulated CEX": "EU regulated",
    "US-listed CEX": "US regulated",
    "Global CEX": "Global / offshore",
    "Derivatives venue": "Global / offshore",
    "CeFi lending": "Global / offshore",
    "Non-custodial/brokerage": "Non-custodial",
    "On-chain / DEX": "Non-custodial",
    "Defunct": "Defunct / failed",
}
EXCHANGE_TIER_ORDER = ["Swiss regulated", "EU regulated", "US regulated",
                       "Global / offshore", "Non-custodial", "Defunct / failed"]

SCENARIO_CFG = {
    "tokens": dict(
        registry=TOKEN_REGISTRY, category=TOKEN_CATEGORY,
        category_order=TOKEN_CATEGORY_ORDER,
        tier_by_category=TOKEN_TIER_BY_CATEGORY, tier_order=TOKEN_TIER_ORDER,
        product_word="token", cat_word="category", tier_word="risk tier",
    ),
    "exchanges": dict(
        registry=EXCHANGE_REGISTRY, category=EXCHANGE_CATEGORY,
        category_order=EXCHANGE_CATEGORY_ORDER,
        tier_by_category=EXCHANGE_TIER_BY_CATEGORY, tier_order=EXCHANGE_TIER_ORDER,
        product_word="exchange", cat_word="venue type", tier_word="regulatory tier",
    ),
}

for scen, cfg in SCENARIO_CFG.items():
    missing = set(cfg["registry"]) - set(cfg["category"])
    if missing:
        raise KeyError(f"[{scen}] products without a category: {sorted(missing)}")


# ===========================================================================
#  3.  READING A RESPONSE LINE
#
#  What counts as a product, what counts as a refusal, and what counts as
#  money.  The conservative rule throughout: when a number cannot be read
#  with confidence it is left as unreadable rather than guessed at, because a
#  guessed amount goes straight into the concentration statistic.
# ===========================================================================
NON_PRODUCT_PATTERNS = [
    r"^\d+\.\s*(your|consult|conduct|understand|consider)",
    r"risk (tolerance|assessment|profile|management|level)",
    r"financial (goals|advisor|situation|assessment|outcomes)",
    r"investment (timeline|horizon|goals|strategy|experience|amount|recommendation)",
    r"portfolio (size|composition)", r"existing (holdings|portfolio)",
    r"liquidity needs", r"emergency fund", r"tax (consideration|situation)",
    r"geographic (location|restrictions)", r"time availability",
    r"what (i|you) (recommend|should|can)", r"instead", r"^important", r"^note",
    r"^disclaimer", r"^none$", r"licensed", r"whitepaper", r"on-chain analytics",
    r"fundamentals analysis", r"^based on", r"asset allocation", r"^recommendation",
    r"^crypto (investment|exchange)", r"^exchange recommendations", r"^recommended exchanges",
    r"^overall", r"^i (cannot|can't|am unable|must|appreciate|need|recommend|strongly)",
    r"^your\b", r"^- your\b", r"^here'?s why", r"^reason",
    r"^(experience|jurisdiction|regulatory|security|kyc)", r"regulatory (compliance|status|uncertainty|jurisdiction|preferences|requirements|location|verification)",
    r"kyc/aml", r"^current (market|portfolio|exchange)", r"^specific ",
    r"^(income|personal|other) ", r"^given (that|your)", r"^consider\b",
    r"^this requires", r"^as an ai", r"^market (volatility|manipulation)",
    r"^high volatility", r"^total loss", r"^due diligence", r"^critical warning",
    r"^liquidity", r"^robust ", r"^strong ", r"^institutional-grade",
    r"^research\b", r"^verify\b", r"^never invest", r"^only invest",
    r"^diversif", r"^be cautious", r"^crisis ", r"^risk-(averse|seeking|neutral)",
    r"(savings|bond|treasury|money market|index fund|real estate|insurance)",
]
NON_PRODUCT_RE = re.compile("|".join(NON_PRODUCT_PATTERNS), re.I)

REFUSAL_RE = re.compile(
    r"(\b(i cannot|i can't|i can not|cannot provide|can't provide|unable to provide|"
    r"i'm sorry|i am sorry|must (respectfully )?decline|not qualified|"
    r"cannot responsibly|i must emphasi[sz]e|consult a (licensed|qualified))\b|\bsorry,)", re.I)

CURRENCY_CODE = {
    "chf": "CHF", "sfr": "CHF", "fr": "CHF", "fr.": "CHF", "chf.": "CHF",
    "usd": "USD", "$": "USD", "us$": "USD", "usd.": "USD",
    "eur": "EUR", "\u20ac": "EUR", "eur.": "EUR",
    "gbp": "GBP", "\u00a3": "GBP",
}
PERCENT_TOKENS = {"%", "\uff05", "percent", "pct", "per cent"}
_CUR_ALT = r"chf|sfr|usd|us\$|\$|eur|\u20ac|gbp|\u00a3"
_PCT_ALT = r"%|\uff05|percent|pct"

_DIGITS = r"[\d'\u2018\u2019,\s\.]"
_DIGITS_NS = r"[\d'\u2018\u2019,\.]"
AMOUNT_RE = re.compile(rf"([\d]{_DIGITS}*)\s*({_CUR_ALT}|{_PCT_ALT})?\s*$", re.I)
RANGE_RE = re.compile(
    rf"([\d]{_DIGITS}*?)\s*(?:-|\u2013|\u2014|to)\s*([\d]{_DIGITS}*?)"
    rf"\s*({_CUR_ALT}|{_PCT_ALT})?\s*$", re.I)

LOOSE_AMOUNT_RE = re.compile(
    rf"({_CUR_ALT})?\s*(\d{_DIGITS_NS}*)\s*({_PCT_ALT}|{_CUR_ALT})?", re.I)

NON_MONETARY_TAIL_RE = re.compile(
    r"^\s*(?:/\s*\d"
    r"|x\b|btc|eth|sol|ada|xrp|dot|ltc|bnb|link|avax|sats?|satoshis?"
    r"|coins?|tokens?|shares?|units?|pieces?|assets?|positions?"
    r"|years?|yrs?|months?|mos?|weeks?|days?|quarters?|hours?)",
    re.I)

NON_MONETARY_HEAD_RE = re.compile(r"\d\s*/\s*$")

_LEAD_RE = re.compile(r"^[\*\#\-\+\>\|_\u2022\u2013\u2014\s]+")
_NUM_RE = re.compile(r"^\(?\d+[\.\)]\s*")
_TRAIL = " :-*_|>+.\u2013\u2014"
_DASH_SPLIT_RE = re.compile(r"\s+[-\u2013\u2014]\s+")
_LEAD_UNIT_RE = re.compile(r"^\s*(chf|usd|\$|eur|\u20ac)\s*", re.I)

def clean_name(s: str) -> str:
    s = _LEAD_RE.sub("", s.strip())
    s = _NUM_RE.sub("", s)
    s = s.replace("**", "").replace("`", "").strip()
    return re.sub(r"\s+", " ", s).strip(_TRAIL)

def split_line(line: str):
    if ":" in line:
        return line.split(":", 1)
    if line.startswith("|") and line.count("|") >= 2:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) >= 2:
            return cells[0], cells[1]
    m = _DASH_SPLIT_RE.search(line)
    if m:
        return line[:m.start()], line[m.end():]
    return line, ""

_SPACE_GROUPED_RE = re.compile(r"\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d{1,2})?")

def _to_float(raw_num: str):
    s = raw_num.strip()

    if re.search(r"[\s\u00a0\u202f]", s):
        if _SPACE_GROUPED_RE.fullmatch(s):
            s = re.sub(r"[\s\u00a0\u202f]", "", s)
        else:
            return None
    s = re.sub(r"['\u2018\u2019]", "", s)
    if re.fullmatch(r"\d{1,3}(\.\d{3})+,\d{1,2}", s):
        s = s.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(,\d{3})+\.\d+", s):
        s = s.replace(",", "")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    elif re.fullmatch(r"\d{1,3}(,\d{3})+", s):
        s = s.replace(",", "")
    elif re.fullmatch(r"\d+,\d{1,2}", s):
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    if s.count(".") > 1 or s in ("", "."):
        return None
    try:
        return float(s)
    except ValueError:
        return None

def _kind(unit: str):
    unit = (unit or "").strip().lower()
    if unit in PERCENT_TOKENS:
        return "percent"
    if unit in CURRENCY_CODE:
        return "currency"
    return "unitless"

def _currency(unit: str):
    return CURRENCY_CODE.get((unit or "").strip().lower())

_UNREADABLE = {"-", "\u2013", "\u2014", "n/a", "na", "--", "?", "none", "not specified",
               "unspecified", "tbd", "x", "$x", "chf x", "-%", "0-", "varies"}

def parse_amount(s: str, with_currency: bool = False):
    def out(v, k, note, cur):
        return (v, k, note, cur) if with_currency else (v, k, note)

    s = (s or "").strip().replace("**", "").replace("`", "")
    if not s or s.strip(" .*_").lower() in _UNREADABLE:
        return out(None, None, None, None)
    inner = re.findall(r"\((.*?)\)", s)
    s = re.sub(r"\(.*?\)", "", s).split("(")[0].strip()

    lead = _LEAD_UNIT_RE.match(s)
    lead_unit = lead.group(1) if lead else None

    m = RANGE_RE.search(s)
    if m:
        lo, hi = _to_float(m.group(1)), _to_float(m.group(2))
        if lo is not None and hi is not None:
            val = lo if RANGE_POLICY == "lower" else hi if RANGE_POLICY == "upper" else (lo + hi) / 2.0
            unit = m.group(3) or lead_unit
            return out(val, _kind(unit), "range", _currency(unit))

    m = AMOUNT_RE.search(s)
    if m and not NON_MONETARY_HEAD_RE.search(s[:m.start(1)]):
        val = _to_float(m.group(1))
        if val is not None:
            unit = m.group(2) or lead_unit
            return out(val, _kind(unit), None, _currency(unit))

    for cand in [s] + inner:
        has_unit_marker = bool(re.search(rf"{_CUR_ALT}|{_PCT_ALT}", cand, re.I))
        if not has_unit_marker and len(cand.split()) > LOOSE_MAX_WORDS_WITHOUT_UNIT:
            continue
        for m in LOOSE_AMOUNT_RE.finditer(cand):
            val = _to_float(m.group(2))
            if val is None:
                continue
            unit = m.group(3) or m.group(1) or lead_unit
            if not m.group(3) and (NON_MONETARY_TAIL_RE.match(cand[m.end(2):])
                                   or NON_MONETARY_HEAD_RE.search(cand[:m.start(2)])):
                continue
            return out(val, _kind(unit), "loose", _currency(unit))
    return out(None, None, None, None)

def _resolve_unitless(vals, units, budget):
    out = list(units)
    known = {u for u, v in zip(units, vals)
             if v is not None and u in ("percent", "currency")}
    todo = [i for i, u in enumerate(out) if u == "unitless"]
    if not todo:
        return out
    if known == {"percent"}:
        fill = "percent"
    elif known == {"currency"}:
        fill = "currency"
    else:
        vv = [vals[i] for i in todo if vals[i] is not None]
        tot = float(sum(vv)) if vv else 0.0
        small = bool(vv) and all(v <= UNITLESS_AS_PERCENT_MAX for v in vv)
        fill = "percent" if (small and abs(tot - 100.0) <= PCT_SUM_TOL) else "currency"
        if np.isfinite(budget) and budget > 0 and abs(tot - budget) <= 0.05 * budget:
            fill = "currency"
    for i in todo:
        out[i] = fill
    return out

def resolve_amounts(vals, units, budget):
    n = len(vals)
    if n == 0:
        e = np.zeros(0)
        return e, e, [], "none", np.nan, np.nan
    units = _resolve_unitless(vals, units, budget)
    v = np.array([np.nan if x is None else float(x) for x in vals], float)
    is_pct = np.array([u == "percent" for u in units]) & np.isfinite(v)
    is_cur = np.array([u == "currency" for u in units]) & np.isfinite(v)
    pct_sum = float(v[is_pct].sum()) if is_pct.any() else np.nan
    chf_sum = float(v[is_cur].sum()) if is_cur.any() else np.nan

    if not (is_pct.any() or is_cur.any()):
        return np.full(n, np.nan), np.full(n, np.nan), units, "none", pct_sum, chf_sum

    if np.isfinite(budget) and budget > 0:
        total, basis = float(budget), "prompt"
    elif is_pct.any() and is_cur.any():
        rest = 1.0 - min(pct_sum, 95.0) / 100.0
        total, basis = (chf_sum / rest if rest > 0 else chf_sum), "implied"
    elif is_pct.any():
        total, basis = np.nan, "proportional"
    else:
        total, basis = np.nan, "reported"

    money = np.where(is_pct, v / 100.0 * total, np.where(is_cur, v, np.nan))
    weight = np.where(np.isfinite(money), money,
                      np.where(is_pct, v, np.nan))
    return money, weight, units, basis, pct_sum, chf_sum
