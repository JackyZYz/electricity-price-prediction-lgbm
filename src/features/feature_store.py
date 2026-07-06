"""特征存储：持久化特征矩阵与元数据。"""
import json
from pathlib import Path
from typing import Optional

import pandas as pd


class FeatureStore:
    """特征存储：保存/加载特征矩阵 Parquet 和 JSON 元数据。"""

    def __init__(self, store_dir: str = "./data/features"):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)

    def _get_paths(self, version: str) -> tuple[Path, Path]:
        parquet_path = self.store_dir / f"features_v{version}.parquet"
        meta_path = self.store_dir / f"features_v{version}.json"
        return parquet_path, meta_path

    def save(
        self,
        features: pd.DataFrame,
        feature_names: list,
        target_col: str,
        version: str,
        metadata: Optional[dict] = None,
    ) -> None:
        """保存特征矩阵和元数据。"""
        parquet_path, meta_path = self._get_paths(version)
        features.to_parquet(parquet_path, index=False)
        meta = {
            "version": version,
            "feature_names": feature_names,
            "target_col": target_col,
            "n_samples": len(features),
            "n_features": len(feature_names),
            "columns": list(features.columns),
        }
        if metadata:
            meta.update(metadata)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load(self, version: str) -> tuple[pd.DataFrame, dict]:
        """加载特征矩阵和元数据。"""
        parquet_path, meta_path = self._get_paths(version)
        if not parquet_path.exists():
            raise FileNotFoundError(f"Feature file not found: {parquet_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        df = pd.read_parquet(parquet_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return df, meta

    def list_versions(self) -> list[str]:
        """列出所有特征版本。"""
        versions = []
        for p in self.store_dir.glob("features_v*.parquet"):
            versions.append(p.stem.replace("features_v", ""))
        return sorted(versions)

    def latest_version(self) -> Optional[str]:
        """获取最新版本。"""
        versions = self.list_versions()
        return versions[-1] if versions else None
