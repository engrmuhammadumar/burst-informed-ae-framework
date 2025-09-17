import yaml, pandas as pd
from pathlib import Path
from src.train import build_feature_table

if __name__ == "__main__":
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    feat_df = build_feature_table(cfg)
    out = Path(cfg["logging"]["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    p = out / "features.parquet"
    feat_df.to_parquet(p)
    print("Saved:", p)
