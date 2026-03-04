from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import yaml


MODELS = {
    "unet": {"name": "unet", "args": {"base_channels": 32}},
    "resnet": {"name": "resnet", "args": {"width": 48, "depth": 8}},
    "transformer": {
        "name": "transformer",
        "args": {"image_size": 128, "patch_size": 8, "embed_dim": 256, "depth": 6, "num_heads": 8},
    },
}


def main() -> None:
    base_cfg_path = Path("configs/base.yaml")
    with base_cfg_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    for model_name, model_cfg in MODELS.items():
        cfg = copy.deepcopy(base_cfg)
        cfg["model"] = model_cfg
        cfg["train"]["output_dir"] = f"outputs/{model_name}"
        cfg_path = Path(f"configs/{model_name}.yaml")
        with cfg_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        print(f"\n=== Training {model_name} ===")
        subprocess.run(["python", "-m", "qsm_transformer.train", "--config", str(cfg_path)], check=True)


if __name__ == "__main__":
    main()
