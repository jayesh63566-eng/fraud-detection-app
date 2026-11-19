"""Data collection: file upload, Google Drive link, Kaggle API, demo data, schema validation."""
import glob
import io
import os
import re
import subprocess
import tempfile
import zipfile

import numpy as np
import pandas as pd

REQUIRED = ["type", "amount", "oldbalanceOrg", "newbalanceOrig",
            "oldbalanceDest", "newbalanceDest", "isFraud"]
OPTIONAL = ["step", "nameOrig", "nameDest", "isFlaggedFraud"]
KAGGLE_SLUG = "ashishkumarjayswal/froud-detection-dataset"


# ---------------------------------------------------------------- reading files
def read_any(file, name=None) -> pd.DataFrame:
    """Read csv / csv.gz / xlsx / parquet / zip (first csv inside) from a path or file-like."""
    name = (name or getattr(file, "name", str(file))).lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(file) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if not csvs:
                raise ValueError("Zip ke andar koi .csv file nahi mili.")
            with z.open(csvs[0]) as f:
                return pd.read_csv(f)
    if name.endswith(".parquet"):
        return pd.read_parquet(file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    if name.endswith(".gz"):
        return pd.read_csv(file, compression="gzip")
    return pd.read_csv(file)


def _read_downloaded(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        head = f.read(300).lstrip().lower()
    if head.startswith(b"<!doctype html") or head.startswith(b"<html"):
        raise RuntimeError("Google ne HTML page diya, file nahi. Link 'Anyone with the link' "
                           "(public) hona chahiye, aur ek direct file hona chahiye (folder nahi).")
    if zipfile.is_zipfile(path):
        return read_any(path, "x.zip")
    try:
        return pd.read_csv(path)
    except (UnicodeDecodeError, pd.errors.ParserError):
        try:
            return pd.read_parquet(path)
        except Exception:
            return pd.read_excel(path)


# ---------------------------------------------------------------- Google Drive
def extract_drive_id(url: str) -> str:
    url = url.strip()
    for pat in (r"/d/([A-Za-z0-9_-]{10,})", r"[?&]id=([A-Za-z0-9_-]{10,})", r"^([A-Za-z0-9_-]{20,})$"):
        m = re.search(pat, url)
        if m:
            return m.group(1)
    raise ValueError("Drive link se file ID nahi mili. Share link paste karo, "
                     "jaise https://drive.google.com/file/d/<ID>/view")


def load_gdrive(url: str) -> pd.DataFrame:
    fid = extract_drive_id(url)
    out = os.path.join(tempfile.mkdtemp(), "drive_download")
    try:
        import gdown  # optional
        path = gdown.download(id=fid, output=out, quiet=True)
        if not path:
            raise RuntimeError("Download fail hua. File public (Anyone with the link) hai?")
    except ImportError:
        import requests
        r = requests.get("https://drive.google.com/uc", params={"export": "download", "id": fid}, timeout=120)
        r.raise_for_status()
        with open(out, "wb") as f:
            f.write(r.content)
        path = out
    return _read_downloaded(path)


# ---------------------------------------------------------------- Kaggle
def kaggle_command(slug: str = KAGGLE_SLUG) -> str:
    return f"kaggle datasets download -d {slug} --unzip"


def parse_kaggle_json(raw: bytes):
    import json
    d = json.loads(raw.decode("utf-8"))
    return d.get("username", ""), d.get("key", "")


def load_kaggle(slug: str, username: str, key: str):
    """Download a Kaggle dataset with the official CLI. Credentials live only in this process env."""
    if not (slug and username and key):
        raise ValueError("Dataset slug, username aur API key teeno chahiye.")
    env = os.environ.copy()
    env["KAGGLE_USERNAME"], env["KAGGLE_KEY"] = username.strip(), key.strip()
    tmp = tempfile.mkdtemp()
    cmd = ["kaggle", "datasets", "download", "-d", slug.strip(), "-p", tmp, "--unzip"]
    try:
        res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    except FileNotFoundError:
        raise RuntimeError("Kaggle CLI nahi mila. Terminal mein chalao: pip install kaggle")
    if res.returncode != 0:
        raise RuntimeError((res.stderr or res.stdout or "Kaggle download fail").strip())
    files = glob.glob(os.path.join(tmp, "**", "*.csv"), recursive=True)
    if not files:
        raise RuntimeError("Download hua par koi CSV nahi mili.")
    biggest = max(files, key=os.path.getsize)
    return pd.read_csv(biggest), [os.path.basename(f) for f in files]


# ---------------------------------------------------------------- validation
def validate_schema(df: pd.DataFrame):
    """Return (missing_required, missing_optional)."""
    cols = set(df.columns)
    return [c for c in REQUIRED if c not in cols], [c for c in OPTIONAL if c not in cols]


def apply_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """mapping = {expected_name: your_column_name}"""
    ren = {src: dst for dst, src in mapping.items() if src and src != dst}
    return df.rename(columns=ren)


# ---------------------------------------------------------------- demo data
def make_demo_data(n: int = 20000, fraud_rate: float = 0.02, seed: int = 42) -> pd.DataFrame:
    """Synthetic PaySim-like data so the app can be tried without any download.
    Fraud pattern (as in PaySim): only TRANSFER / CASH_OUT, account fully emptied,
    destination balance often not updated."""
    rng = np.random.default_rng(seed)
    types = rng.choice(["PAYMENT", "CASH_IN", "CASH_OUT", "TRANSFER", "DEBIT"], size=n,
                       p=[0.34, 0.22, 0.35, 0.08, 0.01])
    step = rng.integers(1, 744, size=n)
    old_org = np.round(rng.lognormal(9.0, 1.6, n), 2)
    amount = np.round(rng.lognormal(8.3, 1.5, n), 2)

    cand = np.isin(types, ["TRANSFER", "CASH_OUT"])
    p = min(1.0, fraud_rate * n / max(cand.sum(), 1))
    fraud = (cand & (rng.random(n) < p)).astype(int)

    cash_in = types == "CASH_IN"
    debit_like = ~cash_in
    # normal debits cannot exceed the balance
    amount = np.where(debit_like & (fraud == 0), np.minimum(amount, old_org * rng.uniform(0.02, 1.0, n)), amount)
    # some LEGIT customers also empty their account (makes the problem realistic / harder)
    legit_drain = debit_like & cand & (fraud == 0) & (rng.random(n) < 0.04)
    amount = np.where(legit_drain, old_org, amount)
    # fraud: usually drains the account, but ~35% are partial (harder to catch)
    old_org = np.where(fraud == 1, np.round(rng.lognormal(10.2, 1.5, n), 2), old_org)
    partial = rng.uniform(0.3, 0.95, n)
    amount = np.where(fraud == 1, np.where(rng.random(n) < 0.65, old_org, np.round(old_org * partial, 2)), amount)
    new_org = np.where(cash_in, old_org + amount, old_org - amount)
    new_org = np.round(np.maximum(new_org, 0), 2)

    merchant = np.isin(types, ["PAYMENT", "DEBIT"])
    old_dest = np.where(merchant, 0.0, np.round(rng.lognormal(9.5, 1.8, n) * (rng.random(n) > 0.3), 2))
    new_dest = np.where(merchant, 0.0,
                        np.where(cash_in, np.maximum(old_dest - amount, 0), old_dest + amount))
    # fraud: destination balance often not updated (mule / system quirk)
    unchanged = ((fraud == 1) & (rng.random(n) < 0.55)) | (cand & (fraud == 0) & (rng.random(n) < 0.08))
    new_dest = np.where(unchanged, old_dest, new_dest)
    new_dest = np.round(new_dest, 2)

    name_orig = ["C" + str(x) for x in rng.integers(10**8, 10**9, n)]
    name_dest = [("M" if m else "C") + str(x) for m, x in zip(merchant, rng.integers(10**8, 10**9, n))]
    flagged = ((types == "TRANSFER") & (amount > 200000) & (fraud == 1) & (rng.random(n) < 0.15)).astype(int)

    return pd.DataFrame({
        "step": step, "type": types, "amount": amount, "nameOrig": name_orig,
        "oldbalanceOrg": old_org, "newbalanceOrig": new_org, "nameDest": name_dest,
        "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
        "isFraud": fraud, "isFlaggedFraud": flagged,
    })
