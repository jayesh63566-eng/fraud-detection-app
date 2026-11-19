"""Feature engineering. The SAME function is used for training and for live prediction,
so there is no train/serve skew."""
import numpy as np
import pandas as pd

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

DEFAULT_OPTS = dict(
    balance_change=True,   # balanceChangeOrig / Dest
    error_balance=True,    # errorBalanceOrig / Dest (accounting mismatch)
    flags=True,            # accountEmptied, fullBalanceTransfer, destBalanceUnchanged, isHighAmount ...
    time_parts=True,       # hour_of_day, day, isNight
    log_amount=True,       # log1p(amount)
    merchant_flag=True,    # isMerchantDest (receiver id starts with 'M')
    txn_counts=False,      # sender/receiver transaction frequency (needs history)
    type_onehot=True,      # type_CASH_IN ... type_TRANSFER
    drop_ids=True,         # drop nameOrig / nameDest
    drop_flagged=True,     # drop isFlaggedFraud (system rule output)
)

FEATURE_HELP = {
    "balance_change": "Sender aur receiver ka balance kitna badla (new − old).",
    "error_balance": "Accounting mismatch: sahi transaction mein old − amount = new hona chahiye. Mismatch = suspicious.",
    "flags": "0/1 red-flag columns: account khaali, poora balance transfer, receiver balance unchanged, bada amount.",
    "time_parts": "step (ghanta) se hour_of_day, day aur isNight nikalta hai.",
    "log_amount": "Amount bahut skewed hota hai, log lene se model ko aasaan padta hai.",
    "merchant_flag": "Receiver merchant (M…) hai ya customer (C…).",
    "txn_counts": "Sender/receiver ne kitne transactions kiye. ⚠️ Live prediction mein history chahiye, isliye default OFF.",
    "type_onehot": "Transaction type ko 0/1 columns mein badalta hai (model numbers samajhta hai).",
    "drop_ids": "Account IDs (C123…) har baar alag hote hain, model ke kaam ke nahi. Hata dete hain.",
    "drop_flagged": "isFlaggedFraud purane rule-system ka output hai, usse hata dete hain.",
}

HIGH_AMOUNT = 200_000


def engineer(df: pd.DataFrame, opts: dict | None = None):
    """Return (new_df, list_of_created_feature_names)."""
    o = {**DEFAULT_OPTS, **(opts or {})}
    d = df.copy()
    created: list[str] = []

    def has(*cols):
        return all(c in d.columns for c in cols)

    def add(name, s):
        d[name] = s
        created.append(name)

    if o["balance_change"]:
        if has("newbalanceOrig", "oldbalanceOrg"):
            add("balanceChangeOrig", d["newbalanceOrig"] - d["oldbalanceOrg"])
        if has("newbalanceDest", "oldbalanceDest"):
            add("balanceChangeDest", d["newbalanceDest"] - d["oldbalanceDest"])

    if o["error_balance"]:
        if has("amount", "oldbalanceOrg", "newbalanceOrig"):
            add("errorBalanceOrig", d["newbalanceOrig"] + d["amount"] - d["oldbalanceOrg"])
        if has("amount", "oldbalanceDest", "newbalanceDest"):
            add("errorBalanceDest", d["oldbalanceDest"] + d["amount"] - d["newbalanceDest"])

    if o["flags"]:
        if has("amount"):
            add("isHighAmount", (d["amount"] > HIGH_AMOUNT).astype(int))
        if has("oldbalanceOrg", "newbalanceOrig"):
            add("accountEmptied", ((d["newbalanceOrig"] == 0) & (d["oldbalanceOrg"] > 0)).astype(int))
            if has("amount"):
                add("fullBalanceTransfer",
                    ((d["amount"] - d["oldbalanceOrg"]).abs() < 0.01).astype(int) * (d["amount"] > 0))
            add("isLargeBalanceChangeOrig", ((d["newbalanceOrig"] - d["oldbalanceOrg"]).abs() > 500_000).astype(int))
        if has("oldbalanceDest", "newbalanceDest"):
            add("destBalanceUnchanged",
                ((d["oldbalanceDest"] == d["newbalanceDest"]) & (d.get("amount", 1) > 0)).astype(int))
            add("isLargeBalanceChangeDest", ((d["newbalanceDest"] - d["oldbalanceDest"]).abs() > 500_000).astype(int))

    if o["time_parts"] and has("step"):
        add("hour_of_day", d["step"] % 24)
        add("day", d["step"] // 24)
        add("isNight", (d["hour_of_day"] < 6).astype(int))

    if o["log_amount"] and has("amount"):
        add("amount_log", np.log1p(d["amount"].clip(lower=0)))

    if o["merchant_flag"] and has("nameDest"):
        add("isMerchantDest", d["nameDest"].astype(str).str.startswith("M").astype(int))

    if o["txn_counts"]:
        if has("nameOrig"):
            add("senderTxnCount", d.groupby("nameOrig")["nameOrig"].transform("size"))
        if has("nameDest"):
            add("receiverTxnCount", d.groupby("nameDest")["nameDest"].transform("size"))

    if o["type_onehot"] and has("type"):
        t = d["type"].astype(str)
        for k in TYPES:
            add(f"type_{k}", (t == k).astype(int))
        d = d.drop(columns=["type"])

    if o["drop_ids"]:
        d = d.drop(columns=[c for c in ("nameOrig", "nameDest") if c in d.columns])
    if o["drop_flagged"] and "isFlaggedFraud" in d.columns:
        d = d.drop(columns=["isFlaggedFraud"])

    for c in d.columns:
        if d[c].dtype == bool:
            d[c] = d[c].astype(int)
    return d, created


def candidate_features(df: pd.DataFrame) -> list[str]:
    """Numeric columns usable by a model (target excluded)."""
    return [c for c in df.columns
            if c != "isFraud" and pd.api.types.is_numeric_dtype(df[c])]


def default_selected(cands: list[str]) -> list[str]:
    """`step`/`day` leave out by default: they don't generalise to future time."""
    return [c for c in cands if c not in ("step", "day")]
