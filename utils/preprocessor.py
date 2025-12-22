#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
초음파 신호 전처리 및 특성 추출 모듈
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
import pywt
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class UltrasoundPreprocessor:
    """초음파 신호 전처리 클래스"""

    def __init__(self, sample_rate: float = 1e6):
        self.sample_rate = sample_rate
        print(f"초음파 전처리기 초기화: 샘플링 주파수 {sample_rate/1e6:.1f} MHz")

    def normalize_signal(self, voltage_data: np.ndarray) -> np.ndarray:
        """신호 정규화"""
        if len(voltage_data) == 0:
            return voltage_data

        # Z-score 정규화
        mean_val = np.mean(voltage_data)
        std_val = np.std(voltage_data)

        if std_val > 0:
            normalized = (voltage_data - mean_val) / std_val
        else:
            normalized = voltage_data - mean_val

        return normalized

    def remove_dc_offset(self, voltage_data: np.ndarray) -> np.ndarray:
        """DC 오프셋 제거"""
        return voltage_data - np.mean(voltage_data)

    def apply_bandpass_filter(self, voltage_data: np.ndarray,
                            low_freq: float = 1e5, high_freq: float = 4e5) -> np.ndarray:
        """대역 통과 필터 적용"""
        if len(voltage_data) < 10:
            return voltage_data

        # 정규화된 컷오프 주파수
        nyquist = self.sample_rate / 2
        low = low_freq / nyquist
        high = high_freq / nyquist

        # 필터 계수 계산
        b, a = signal.butter(4, [low, high], btype='band')

        # 필터 적용
        filtered = signal.filtfilt(b, a, voltage_data)
        return filtered

    def extract_time_domain_features(self, voltage_data: np.ndarray) -> Dict[str, float]:
        """시간 도메인 특성 추출"""
        if len(voltage_data) == 0:
            return {}

        features = {
            'mean': np.mean(voltage_data),
            'std': np.std(voltage_data),
            'rms': np.sqrt(np.mean(voltage_data**2)),
            'peak_to_peak': np.max(voltage_data) - np.min(voltage_data),
            'crest_factor': np.max(np.abs(voltage_data)) / np.sqrt(np.mean(voltage_data**2)) if np.mean(voltage_data**2) > 0 else 0,
            'kurtosis': self._kurtosis(voltage_data),
            'skewness': self._skewness(voltage_data),
        }

        return features

    def extract_frequency_domain_features(self, voltage_data: np.ndarray) -> Dict[str, float]:
        """주파수 도메인 특성 추출"""
        if len(voltage_data) < 10:
            return {}

        # FFT 계산
        fft_vals = fft(voltage_data)
        fft_freq = fftfreq(len(voltage_data), d=1/self.sample_rate)

        # 양의 주파수만 사용
        pos_mask = fft_freq > 0
        fft_magnitude = np.abs(fft_vals[pos_mask])
        fft_freq_pos = fft_freq[pos_mask]

        features = {
            'dominant_freq': fft_freq_pos[np.argmax(fft_magnitude)],
            'mean_freq': np.sum(fft_freq_pos * fft_magnitude) / np.sum(fft_magnitude),
            'median_freq': self._weighted_median(fft_freq_pos, fft_magnitude),
            'freq_centroid': np.sum(fft_freq_pos * fft_magnitude) / np.sum(fft_magnitude),
            'freq_spread': np.sqrt(np.sum(((fft_freq_pos - features.get('mean_freq', 0))**2) * fft_magnitude) / np.sum(fft_magnitude)) if 'mean_freq' in locals() else 0,
        }

        return features

    def extract_wavelet_features(self, voltage_data: np.ndarray,
                               wavelet: str = 'db4', level: int = 4) -> Dict[str, float]:
        """웨이블릿 변환을 통한 특성 추출"""
        if len(voltage_data) < 32:  # 웨이블릿 변환을 위한 최소 길이
            return {}

        try:
            # 웨이블릿 분해
            coeffs = pywt.wavedec(voltage_data, wavelet, level=level)

            features = {}
            for i, coeff in enumerate(coeffs):
                features[f'wavelet_level_{i}_energy'] = np.sum(coeff**2)
                features[f'wavelet_level_{i}_entropy'] = self._shannon_entropy(coeff)

            return features

        except Exception as e:
            print(f"웨이블릿 변환 실패: {e}")
            return {}

    def extract_envelope(self, voltage_data: np.ndarray) -> np.ndarray:
        """신호 엔벨로프 추출 (Hilbert 변환)"""
        if len(voltage_data) == 0:
            return voltage_data

        # Hilbert 변환으로 엔벨로프 추출
        analytic_signal = signal.hilbert(voltage_data)
        envelope = np.abs(analytic_signal)

        return envelope

    def detect_peaks(self, voltage_data: np.ndarray,
                    height: Optional[float] = None,
                    distance: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """피크 검출"""
        if len(voltage_data) == 0:
            return np.array([]), {}

        peaks, properties = signal.find_peaks(voltage_data,
                                            height=height or np.mean(np.abs(voltage_data)),
                                            distance=distance)

        peak_info = {
            'num_peaks': len(peaks),
            'mean_peak_height': np.mean(properties['peak_heights']) if len(peaks) > 0 else 0,
            'max_peak_height': np.max(properties['peak_heights']) if len(peaks) > 0 else 0,
        }

        return peaks, peak_info

    def preprocess_signal(self, voltage_data: np.ndarray,
                         normalize: bool = True,
                         remove_dc: bool = True,
                         filter_signal: bool = True) -> np.ndarray:
        """전체 전처리 파이프라인"""
        if len(voltage_data) == 0:
            return voltage_data

        processed = voltage_data.copy()

        # DC 오프셋 제거
        if remove_dc:
            processed = self.remove_dc_offset(processed)

        # 필터링
        if filter_signal:
            processed = self.apply_bandpass_filter(processed)

        # 정규화
        if normalize:
            processed = self.normalize_signal(processed)

        return processed

    def extract_all_features(self, voltage_data: np.ndarray) -> Dict[str, float]:
        """모든 특성 추출"""
        if len(voltage_data) == 0:
            return {}

        # 전처리
        processed_signal = self.preprocess_signal(voltage_data)

        # 각 도메인별 특성 추출
        time_features = self.extract_time_domain_features(processed_signal)
        freq_features = self.extract_frequency_domain_features(processed_signal)
        wavelet_features = self.extract_wavelet_features(processed_signal)

        # 피크 분석
        _, peak_info = self.detect_peaks(processed_signal)

        # 엔벨로프 특성
        envelope = self.extract_envelope(processed_signal)
        envelope_features = self.extract_time_domain_features(envelope)

        # 모든 특성 통합
        all_features = {}
        all_features.update({f'time_{k}': v for k, v in time_features.items()})
        all_features.update({f'freq_{k}': v for k, v in freq_features.items()})
        all_features.update({f'wavelet_{k}': v for k, v in wavelet_features.items()})
        all_features.update({f'peak_{k}': v for k, v in peak_info.items()})
        all_features.update({f'envelope_{k}': v for k, v in envelope_features.items()})

        return all_features

    def _kurtosis(self, data: np.ndarray) -> float:
        """첨도 계산"""
        if len(data) < 2:
            return 0
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        return np.mean(((data - mean_val) / std_val) ** 4)

    def _skewness(self, data: np.ndarray) -> float:
        """왜도 계산"""
        if len(data) < 2:
            return 0
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0
        return np.mean(((data - mean_val) / std_val) ** 3)

    def _weighted_median(self, values: np.ndarray, weights: np.ndarray) -> float:
        """가중치 기반 중앙값 계산"""
        if len(values) == 0:
            return 0

        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]

        cumsum = np.cumsum(sorted_weights)
        cutoff = cumsum[-1] / 2

        return sorted_values[np.searchsorted(cumsum, cutoff)]

    def _shannon_entropy(self, data: np.ndarray) -> float:
        """샤논 엔트로피 계산"""
        if len(data) == 0:
            return 0

        # 히스토그램 생성
        hist, _ = np.histogram(data, bins=50, density=True)
        hist = hist[hist > 0]  # 0 값 제거

        if len(hist) == 0:
            return 0

        return -np.sum(hist * np.log2(hist))

def main():
    """메인 함수 - 전처리 데모"""
    print("초음파 신호 전처리 모듈 테스트")

    # 샘플 데이터 생성
    sample_rate = 1e6
    t = np.linspace(0, 1e-5, 1000)  # 10μs
    frequency = 5e6  # 5MHz
    sample_signal = np.sin(2 * np.pi * frequency * t) + 0.1 * np.random.randn(len(t))

    # 전처리기 초기화
    preprocessor = UltrasoundPreprocessor(sample_rate)

    # DC 오프셋 제거
    dc_removed = preprocessor.remove_dc_offset(sample_signal)

    # 밴드패스 필터 적용
    filtered = preprocessor.apply_bandpass_filter(dc_removed, low_freq=1e5, high_freq=4e5)

    # 정규화
    normalized = preprocessor.normalize_signal(filtered)

    # 특성 추출
    features = preprocessor.extract_all_features(normalized)

    print(f"샘플 신호 길이: {len(sample_signal)}")
    print(f"DC 제거 후 평균: {np.mean(dc_removed):.6f}")
    print(f"필터 적용 후 RMS: {np.sqrt(np.mean(filtered**2)):.6f}")
    print(f"정규화 후 평균: {np.mean(normalized):.6f}")
    print(f"추출된 특성 수: {len(features)}")

    print("전처리 모듈 테스트 완료!")

if __name__ == "__main__":
    main()