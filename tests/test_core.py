"""Run:  python -m pytest tests -q   (or: python tests/test_core.py)"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import pandas as pd
from core import deploy, evaluation, features, loaders, modeling


def test_pipeline_end_to_end():
    df = loaders.make_demo_data(8000)
    assert loaders.validate_schema(df) == ([], [])
    feat, created = features.engineer(df)
    assert "errorBalanceOrig" in created and "type" not in feat.columns
    sel = features.default_selected(features.candidate_features(feat))
    sp = modeling.make_split(feat, sel)
    # no leakage: resampling only touches train
    n_test = len(sp["X_test"])
    r = modeling.train_model("Random Forest", sp, "SMOTE", 0.3)
    assert len(sp["X_test"]) == n_test and r["metrics"]["pr_auc"] > 0.3
    b = deploy.bundle_from_bytes(deploy.bundle_to_bytes(
        deploy.build_bundle("RF", r["pipeline"], sel, features.DEFAULT_OPTS, 0.5, r["metrics"])))
    raw = pd.DataFrame([dict(type="TRANSFER", amount=181000, oldbalanceOrg=181000, newbalanceOrig=0,
                             oldbalanceDest=0, newbalanceDest=0)])
    out = deploy.predict(b, raw)
    assert out["fraud_probability"].iloc[0] > 0.5


if __name__ == "__main__":
    test_pipeline_end_to_end(); print("OK")
