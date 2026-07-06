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

    def stationary_decompose(self, signal: np.ndarray) -> dict:
        """
        使用平稳小波变换（SWT）分解，保持各分量长度与原始信号一致。
        trim_approx=True 返回 [cA_L, cD_L, cD_{L-1}, ..., cD_1]，
        映射为 {"low": cA_L, "high_1": cD_L, ..., "high_L": cD_1}。
        """
        signal = np.asarray(signal)
        orig_len = len(signal)
        # SWT 要求信号长度是 2^level 的整数倍
        pad_len = (2 ** self.level) - (orig_len % (2 ** self.level))
        if pad_len == 2 ** self.level:
            pad_len = 0
        if pad_len > 0:
            signal = np.pad(signal, (0, pad_len), mode="edge")
        coeffs = pywt.swt(signal, self.wavelet, level=self.level, trim_approx=True)
        # coeffs 为 [cA_L, cD_L, cD_{L-1}, ..., cD_1]
        result = {"low": coeffs[0][:orig_len]}
        for i in range(1, self.level + 1):
            result[f"high_{i}"] = coeffs[i][:orig_len]
        return result

    def stationary_reconstruct(self, components: dict) -> np.ndarray:
        """从 SWT 分量重构原始信号；分量顺序需与 stationary_decompose 对应。"""
        coeffs = [components["low"]]
        for i in range(1, self.level + 1):
            coeffs.append(components[f"high_{i}"])
        return pywt.iswt(coeffs, self.wavelet)

    def pad_to_power_of_two(self, signal: np.ndarray) -> tuple:
        """将信号填充到2的幂次长度"""
        orig_len = len(signal)
        target_len = 2 ** int(np.ceil(np.log2(orig_len)))
        if orig_len == target_len:
            return signal, orig_len
        padded = np.pad(signal, (0, target_len - orig_len), mode="reflect")
        return padded, orig_len
