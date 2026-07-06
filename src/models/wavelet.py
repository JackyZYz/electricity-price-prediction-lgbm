"""小波分解模块。"""
import numpy as np
import pywt


class WaveletDecomposer:
    """小波分解/重构：WT-LGBM 的核心前置模块"""

    def __init__(self, wavelet: str = "db4", level: int = 2):
        self.wavelet = wavelet
        self.level = level

    def decompose(self, signal: np.ndarray) -> dict:
        """对历史电价序列做小波分解"""
        coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
        result = {"low": coeffs[0]}
        for i, coeff in enumerate(coeffs[1:], start=1):
            result[f"high_{i}"] = coeff
        return result

    def reconstruct(self, components: dict) -> np.ndarray:
        """从小波分量重构原始信号"""
        coeffs = [components["low"]]
        for i in range(self.level, 0, -1):
            coeffs.append(components[f"high_{i}"])
        return pywt.waverec(coeffs, self.wavelet)

    def pad_to_power_of_two(self, signal: np.ndarray) -> tuple:
        """将信号填充到2的幂次长度"""
        orig_len = len(signal)
        target_len = 2 ** int(np.ceil(np.log2(orig_len)))
        if orig_len == target_len:
            return signal, orig_len
        padded = np.pad(signal, (0, target_len - orig_len), mode="reflect")
        return padded, orig_len
