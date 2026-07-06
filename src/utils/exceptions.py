"""自定义异常类。"""


class DataLeakageException(Exception):
    """数据泄露异常：特征使用了目标时间之后的信息。"""
    pass


class ModelNotTrainedError(Exception):
    """模型未训练或未加载。"""
    pass


class DataQualityError(Exception):
    """数据质量不满足最低要求（如缺失率>50%）。"""
    pass


class FeatureMismatchError(Exception):
    """特征维度/名称不匹配。"""
    pass


class PredictionOutOfRangeWarning(UserWarning):
    """预测值超出合理范围。"""
    pass
