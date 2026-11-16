"""Project paths and YAML config helpers."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
BAG_DIR = ROOT / "data" / "bags"
RESULTS_DIR = ROOT / "results"


def load_yaml(name: str) -> dict:
    with open(CONFIG_DIR / name, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def bag_path(bag_no: int) -> Path:
    bags = load_yaml("bags.yaml")["bags"]
    if bag_no not in bags:
        raise SystemExit(f"--bag {bag_no} is not in config/bags.yaml (have {sorted(bags)})")
    return BAG_DIR / bags[bag_no]["file"]


def topics() -> dict:
    """Role -> topic name, from config/topics.yaml (imu, gnss, image, odom)."""
    return load_yaml("topics.yaml")
