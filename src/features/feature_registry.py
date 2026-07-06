"""特征注册中心。"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FeatureMeta:
    """特征元数据。"""
    name: str
    category: str
    status: str
    description: str = ""
    source_table: Optional[str] = None
    formula: Optional[str] = None
    dtype: str = "float"
    importance: float = 0.0
    version: int = 1


class FeatureRegistry:
    """特征注册中心：管理特征的全生命周期。"""

    VALID_STATUSES = {"OK", "PENDING", "DROPPED"}
    VALID_CATEGORIES = {"direct", "constructed", "lag", "date"}

    def __init__(self):
        self.features: dict[str, FeatureMeta] = {}

    def register(self, meta: FeatureMeta) -> None:
        """注册一个特征。"""
        if meta.status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status: {meta.status}")
        if meta.category not in self.VALID_CATEGORIES:
            raise ValueError(f"Invalid category: {meta.category}")
        self.features[meta.name] = meta

    def get_available_features(self) -> list[str]:
        """获取 status=OK 的特征列表。"""
        return [name for name, meta in self.features.items() if meta.status == "OK"]

    def get_features_by_category(self, category: str) -> list[str]:
        """按类别获取特征名。"""
        return [name for name, meta in self.features.items() if meta.category == category]

    def update_importance(self, name: str, importance: float) -> None:
        """更新特征重要性。"""
        if name not in self.features:
            raise KeyError(f"Feature {name} not registered")
        self.features[name].importance = importance

    def drop(self, name: str) -> None:
        """将特征标记为 DROPPED。"""
        if name in self.features:
            self.features[name].status = "DROPPED"

    def snapshot(self) -> dict:
        """保存当前特征状态。"""
        return {
            "version": 1,
            "features": {name: meta.__dict__ for name, meta in self.features.items()},
        }

    @classmethod
    def from_config(cls, config: dict) -> "FeatureRegistry":
        """从 features.yaml 配置构建注册中心。"""
        registry = cls()
        for item in config.get("features", []):
            meta = FeatureMeta(
                name=item["name"],
                category=item["category"],
                status=item.get("status", "OK"),
                description=item.get("description", ""),
                source_table=item.get("source_table"),
                formula=item.get("formula"),
                dtype=item.get("dtype", "float"),
            )
            registry.register(meta)
        return registry
