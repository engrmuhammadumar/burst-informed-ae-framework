from pathlib import Path
import yaml, json
from src.train import run_full_pipeline

if __name__ == "__main__":
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    Path(cfg["logging"]["out_dir"]).mkdir(parents=True, exist_ok=True)
    res = run_full_pipeline(cfg)
    Path(cfg["logging"]["out_dir"], "summary.json").write_text(json.dumps(res, indent=2))
    print("\n== Done ==\n", json.dumps(res, indent=2))
