"""工具模块。"""
import logging
from pathlib import Path


def setup_logger(name: str, log_dir: str = "logs", level: str = "INFO") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    if not logger.handlers:
        fh = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(sh)
    return logger
