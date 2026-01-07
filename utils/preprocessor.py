#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
초음파 신호 전처리 및 특성 추출 모듈
피부 조직 두께 측정용
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy import interpolate
from scipy.fft import fft, fftfreq
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("Warning: pywt not available. Wavelet features will be skipped.")
from typing import Dict, List, Tuple, Optional
import warnings
import matplotlib.pyplot as plt
import os
import json
import glob

warnings.filterwarnings('ignore')

class UltrasoundPreprocessor:
    """초음파 신호 전처리 클래스 - 피부 조직 두께 측정용"""

    def __init__(self, sample_rate: float = 5e6, speed_of_sound: float = 1540.0):
        self.sample_rate = sample_rate
        self.speed_of_sound = speed_of_sound  # m/s
        print(f"초음파 전처리기 초기화: 샘플링 주파수 {sample_rate/1e6:.1f} MHz, 음속 {speed_of_sound} m/s")

    def load_ultrasound_data(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """CSV 파일에서 초음파 데이터를 로드"""
        try:
            df = pd.read_csv(file_path, header=0)
            # 첫 번째 열은 시간, 두 번째 열은 전압
            time_col = df.iloc[:, 0]
            voltage_col = df.iloc[:, 1]

            # 시간 데이터를 숫자로 변환
            time_data = pd.to_numeric(time_col, errors='coerce').values
            voltage_data = pd.to_numeric(voltage_col, errors='coerce').values

            # NaN 값 제거
            valid_mask = ~np.isnan(time_data) & ~np.isnan(voltage_data)
            time_data = time_data[valid_mask]
            voltage_data = voltage_data[valid_mask]

            print(f"데이터 로드 완료: {len(voltage_data)} 샘플, 시간 범위: {time_data[0]:.2e} ~ {time_data[-1]:.2e} s")
            return time_data, voltage_data

        except Exception as e:
            print(f"데이터 로드 실패: {e}")
            return np.array([]), np.array([])

    def normalize_signal(self, voltage_data: np.ndarray, method: str = 'zscore') -> np.ndarray:
        """신호 정규화 - 초음파 신호에 최적화"""
        if len(voltage_data) == 0:
            return voltage_data

        if method == 'zscore':
            # Z-score 정규화 (기존)
            mean_val = np.mean(voltage_data)
            std_val = np.std(voltage_data)
            if std_val > 0:
                normalized = (voltage_data - mean_val) / std_val
            else:
                normalized = voltage_data - mean_val
        elif method == 'minmax':
            # Min-Max 정규화 (0-1 범위)
            min_val = np.min(voltage_data)
            max_val = np.max(voltage_data)
            if max_val > min_val:
                normalized = (voltage_data - min_val) / (max_val - min_val)
            else:
                normalized = voltage_data - min_val
        elif method == 'abs_max':
            # 절대 최대값으로 정규화 (-1 ~ 1 범위)
            max_abs = np.max(np.abs(voltage_data))
            if max_abs > 0:
                normalized = voltage_data / max_abs
            else:
                normalized = voltage_data
        else:
            normalized = voltage_data

        return normalized

    def remove_dc_offset(self, voltage_data: np.ndarray) -> np.ndarray:
        """DC 오프셋 제거"""
        return voltage_data - np.mean(voltage_data)

    def apply_bandpass_filter(self, voltage_data: np.ndarray,
                            low_freq: float = 1e5, high_freq: float = 4e5) -> np.ndarray:
        """대역 통과 필터 적용 - 5MHz 탐촉자 데이터용"""
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
        if not PYWT_AVAILABLE:
            return {}

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

    def apply_spline_interpolation(self, time_data: np.ndarray, voltage_data: np.ndarray, 
                                 num_points: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        """스플라인 보간 적용"""
        if len(time_data) < 4:  # 스플라인 보간을 위한 최소 점 수
            return time_data, voltage_data
        
        try:
            # Cubic Spline 보간
            cs = interpolate.CubicSpline(time_data, voltage_data)
            
            # 보간된 시간 점 생성
            time_interp = np.linspace(time_data[0], time_data[-1], num_points)
            voltage_interp = cs(time_interp)
            
            return time_interp, voltage_interp
        except Exception as e:
            print(f"스플라인 보간 실패: {e}")
            return time_data, voltage_data

    def detect_peaks(self, voltage_data: np.ndarray,
                    height: Optional[float] = None,
                    distance: Optional[int] = None,
                    prominence: Optional[float] = None,
                    width: Optional[float] = None,
                    max_peaks: int = 4) -> Tuple[np.ndarray, Dict]:
        """피크 검출 (최대 max_peaks개로 제한, 개선된 파라미터)"""
        if len(voltage_data) == 0:
            return np.array([]), {}

        peaks, properties = signal.find_peaks(voltage_data,
                                            height=height or np.mean(np.abs(voltage_data)),
                                            distance=distance,
                                            prominence=prominence,
                                            width=width)

        # 피크가 max_peaks개보다 많으면 시간 순서대로 앞쪽 max_peaks개 선택
        if len(peaks) > max_peaks:
            peaks = peaks[:max_peaks]
            if 'peak_heights' in properties:
                properties['peak_heights'] = properties['peak_heights'][:max_peaks]

        peak_info = {
            'num_peaks': len(peaks),
            'mean_peak_height': np.mean(properties['peak_heights']) if len(peaks) > 0 and 'peak_heights' in properties else 0,
            'max_peak_height': np.max(properties['peak_heights']) if len(peaks) > 0 and 'peak_heights' in properties else 0,
        }

        return peaks, peak_info

    def visualize_multiple_signals_overlay(self, results: List[Dict], patient_id: str, save_path: Optional[str] = None):
        """여러 부위의 신호를 중첩시켜 시각화"""
        if not results:
            return

        plt.figure(figsize=(15, 10))

        # 색상 팔레트 (8개 부위용)
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        alpha_values = [0.8, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]  # 첫 번째 신호는 더 진하게

        # 첫 번째 서브플롯: 전체 신호 중첩
        plt.subplot(2, 1, 1)
        max_time = 0
        min_time = float('inf')

        for i, result in enumerate(results):
            time_data = result['time_data']
            processed_data = result['processed_data']
            peaks = result['peaks']
            position = result['position']

            # 시간 데이터를 μs 단위로 변환
            time_us = time_data * 1e6

            # 스케일링 및 음수 처리 (절대값, 1/100 스케일)
            voltage_scaled = np.abs(processed_data) * 0.01

            # 최대/최소 시간 업데이트
            max_time = max(max_time, np.max(time_us))
            min_time = min(min_time, np.min(time_us))

            # 신호 플롯
            color = colors[i % len(colors)]
            alpha = alpha_values[i] if i < len(alpha_values) else 0.4
            label = f'Position {position} (Peaks: {len(peaks)})'
            plt.plot(time_us, voltage_scaled, color=color, alpha=alpha, linewidth=1.5, label=label)

            # 피크 표시 (첫 번째 부위만 크게 표시)
            if i == 0:
                plt.scatter(time_us[peaks], voltage_scaled[peaks], color=color, s=80, zorder=10, marker='o', edgecolors='black', linewidth=1)
            else:
                plt.scatter(time_us[peaks], voltage_scaled[peaks], color=color, s=40, zorder=5, alpha=0.7)

        plt.xlabel('Time (μs)')
        plt.ylabel('Voltage (V) - 0.01x Scaled')
        plt.title(f'{patient_id} - All 8 Positions Ultrasound Signals Overlay')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xlim(min_time, max_time)

        # 두 번째 서브플롯: 진피 두께 비교
        plt.subplot(2, 2, 3)
        positions = [r['position'] for r in results]
        dermal_thicknesses = [r['dermal_thickness'] for r in results]

        bars = plt.bar(positions, dermal_thicknesses, color=colors[:len(results)], alpha=0.7)
        plt.xlabel('Position')
        plt.ylabel('Dermal Thickness (mm)')
        plt.title('Dermal Thickness by Position')
        plt.grid(True, alpha=0.3)

        # 막대 위에 값 표시
        for bar, thickness in zip(bars, dermal_thicknesses):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    '.2f', ha='center', va='bottom', fontsize=9)

        # 세 번째 서브플롯: 피크 수 비교
        plt.subplot(2, 2, 4)
        peak_counts = [r['peaks_detected'] for r in results]

        bars = plt.bar(positions, peak_counts, color=colors[:len(results)], alpha=0.7)
        plt.xlabel('Position')
        plt.ylabel('Number of Peaks')
        plt.title('Detected Peaks by Position')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, max(peak_counts) + 1)

        # 막대 위에 값 표시
        for bar, count in zip(bars, peak_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom', fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"중첩 시각화 저장됨: {save_path}")

        plt.show()

    def calculate_thickness(self, time_data: np.ndarray, peaks: np.ndarray) -> Dict[str, float]:
        """피크를 이용한 조직 두께 계산"""
        if len(peaks) < 2:
            return {'thickness_mm': 0.0, 'num_layers': 0}

        # 인접 피크 간 시간 차이 계산
        peak_times = time_data[peaks]
        time_diffs = np.diff(peak_times)

        # 두께 계산: thickness = (time_diff * speed_of_sound) / 2
        # 왕복 거리이므로 2로 나눔
        thicknesses = (time_diffs * self.speed_of_sound) / 2

        # mm 단위로 변환
        thicknesses_mm = thicknesses * 1000

        # 평균 두께 계산
        avg_thickness = np.mean(thicknesses_mm) if len(thicknesses_mm) > 0 else 0.0

        return {
            'thickness_mm': avg_thickness,
            'num_layers': len(peaks) - 1,
            'individual_thicknesses': thicknesses_mm.tolist()
        }

    def visualize_signal_and_peaks(self, time_data: np.ndarray, voltage_data: np.ndarray,
                                 peaks: np.ndarray, patient_id: str = "",
                                 save_path: Optional[str] = None):
        """신호와 피크를 시각화"""
        if len(time_data) == 0 or len(voltage_data) == 0:
            return

        plt.figure(figsize=(12, 8))

        # 시간 데이터를 μs 단위로 변환
        time_us = time_data * 1e6

        # 시인성을 위해 전압 스케일 1/100로 축소 및 음수 처리 (절대값)
        voltage_scaled = np.abs(voltage_data) * 0.01

        # 신호 플롯
        plt.subplot(2, 1, 1)
        plt.plot(time_us, voltage_scaled, 'b-', linewidth=1, alpha=0.7, label='Ultrasound Signal (0.01x scaled)')
        plt.scatter(time_us[peaks], voltage_scaled[peaks], color='red', s=50, zorder=5, label='Peaks')
        
        plt.xlabel('Time (μs)')
        plt.ylabel('Voltage (V) - 0.01x Scaled')
        
        plt.xlabel('Time (μs)')
        plt.ylabel('Voltage (V) - 0.01x Scaled')
        plt.title(f'Ultrasound Signal - {patient_id}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 피크 위치 표시 (시간과 절대 거리 정보 포함)
        start_time_us = time_us[peaks[0]] if len(peaks) > 0 else 0
        for i, peak_idx in enumerate(peaks[:8]):  # 최대 8개 피크만 표시
            time_us_peak = time_us[peak_idx]
            voltage = voltage_scaled[peak_idx]  # 스케일된 전압 사용

            # 절대 거리 계산 (첫 피크부터의 누적 거리)
            absolute_distance = (time_us_peak - start_time_us) * 1e-6 * self.speed_of_sound / 2 * 1000  # mm

            # 피크 레이블 (절대 시간과 절대 거리)
            label = f'P{i+1}\n{time_us_peak:.1f}μs\n{absolute_distance:.2f}mm'

            plt.annotate(label, (time_us_peak, voltage),
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                        fontsize=8)

        # 확대된 뷰
        plt.subplot(2, 1, 2)
        if len(peaks) >= 2:
            # 첫 두 피크 사이 영역 확대
            start_idx = max(0, peaks[0] - 50)
            end_idx = min(len(time_data), peaks[1] + 50)
            plt.plot(time_us[start_idx:end_idx], voltage_scaled[start_idx:end_idx], 'b-', linewidth=1.5)
            plt.scatter(time_us[peaks[:2]], voltage_scaled[peaks[:2]], color='red', s=100, zorder=5)

            # 두께 표시 (첫 두 피크 사이)
            thickness_info = self.calculate_thickness(time_data, peaks[:2])
            if thickness_info['thickness_mm'] > 0:
                mid_time = (time_us[peaks[0]] + time_us[peaks[1]]) / 2
                mid_voltage = (voltage_scaled[peaks[0]] + voltage_scaled[peaks[1]]) / 2
                plt.annotate(f'{thickness_info["thickness_mm"]:.2f}mm',
                            (mid_time, mid_voltage), xytext=(0, 20),
                            textcoords='offset points', ha='center',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

        plt.xlabel('Time (μs)')
        plt.ylabel('Voltage (V)')
        plt.title('Zoomed View - First Two Peaks')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"플롯 저장됨: {save_path}")

        plt.show()

    def apply_moving_average_filter(self, voltage_data: np.ndarray, window_size: int = 3) -> np.ndarray:
        """이동 평균 필터로 노이즈 제거 - 약한 필터링"""
        if len(voltage_data) < window_size:
            return voltage_data
        return np.convolve(voltage_data, np.ones(window_size)/window_size, mode='same')

    def apply_median_filter(self, voltage_data: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        """미디안 필터로 노이즈 제거 - 약한 필터링"""
        if len(voltage_data) < kernel_size:
            return voltage_data
        return signal.medfilt(voltage_data, kernel_size)

    def preprocess_signal(self, voltage_data: np.ndarray,
                         normalize: bool = True, remove_dc: bool = True,
                         filter_signal: bool = True, norm_method: str = 'abs_max',
                         apply_denoising: bool = True) -> np.ndarray:
        """전체 전처리 파이프라인 - 노이즈 제거 강화"""
        if len(voltage_data) == 0:
            return voltage_data

        processed = voltage_data.copy()

        # DC 오프셋 제거
        if remove_dc:
            processed = self.remove_dc_offset(processed)

        # 노이즈 제거 (다중 필터 적용) - 선택적 적용
        if apply_denoising:
            # 1. 미디안 필터 (임펄스 노이즈 제거) - 매우 약하게
            processed = self.apply_median_filter(processed, kernel_size=3)
            # 2. 이동 평균 필터 (고주파 노이즈 평활화) - 매우 약하게
            processed = self.apply_moving_average_filter(processed, window_size=3)

        # 필터링
        if filter_signal:
            processed = self.apply_bandpass_filter(processed)

        # 정규화
        if normalize:
            processed = self.normalize_signal(processed, method=norm_method)

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

    def load_manual_label(self, label_file: str) -> Optional[Dict]:
        """
        manual_boundaries의 JSON 레이블 파일 로드

        Args:
            label_file: JSON 레이블 파일 경로

        Returns:
            레이블 데이터 딕셔너리 또는 None
        """
        try:
            with open(label_file, 'r', encoding='utf-8') as f:
                label_data = json.load(f)

            print(f"레이블 로드 완료: {label_file}")
            print(f"  - Start point: {label_data.get('start_point_us', 0):.2f}μs")
            print(f"  - Positions: {label_data.get('num_positions', 0)}")

            return label_data

        except Exception as e:
            print(f"레이블 로드 실패 ({label_file}): {e}")
            return None

    def find_label_for_data(self, csv_file_path: str, label_dir: str = "./manual_boundaries") -> Optional[Dict]:
        """
        CSV 데이터 파일에 해당하는 JSON 레이블 파일 찾기

        Args:
            csv_file_path: CSV 데이터 파일 경로
            label_dir: 레이블 디렉토리 경로

        Returns:
            레이블 데이터 또는 None
        """
        # CSV 파일명에서 기본 이름 추출
        base_name = os.path.basename(csv_file_path).replace('.csv', '')

        # 해당하는 JSON 파일 찾기
        label_file = os.path.join(label_dir, f"{base_name}_positions.json")

        if os.path.exists(label_file):
            return self.load_manual_label(label_file)
        else:
            print(f"레이블 파일 없음: {label_file}")
            return None

    def create_training_dataset(self, data_dir: str = "./data",
                               label_dir: str = "./manual_boundaries",
                               output_file: str = "./training_dataset.npz") -> Dict:
        """
        학습용 데이터셋 생성

        Args:
            data_dir: CSV 데이터 디렉토리
            label_dir: JSON 레이블 디렉토리
            output_file: 출력 파일 경로 (npz 형식)

        Returns:
            데이터셋 통계 정보
        """
        print(f"\n=== 학습 데이터셋 생성 시작 ===")
        print(f"데이터 디렉토리: {data_dir}")
        print(f"레이블 디렉토리: {label_dir}")

        # 모든 레이블 파일 찾기
        label_files = glob.glob(os.path.join(label_dir, "*_positions.json"))
        print(f"발견된 레이블 파일: {len(label_files)}개")

        dataset = {
            'signals': [],          # 전처리된 신호
            'time_data': [],        # 시간 데이터
            'start_points': [],     # 시작점 (μs)
            'dermis_times': [],     # 진피 시작 시간 (μs)
            'fascia_times': [],     # 근막 시작 시간 (μs)
            'dermis_thickness': [], # 진피 두께 (mm)
            'fascia_thickness': [], # 근막 두께 (mm)
            'file_names': [],       # 파일명
        }

        successful = 0
        failed = 0

        for label_file in label_files:
            # 레이블 로드
            label_data = self.load_manual_label(label_file)
            if label_data is None:
                failed += 1
                continue

            # 해당 CSV 파일 찾기
            source_file = label_data.get('source_file', '')
            if not os.path.exists(source_file):
                # 상대 경로로 다시 시도
                base_name = os.path.basename(label_file).replace('_positions.json', '')
                source_file = os.path.join(data_dir, f"{base_name}.csv")

            if not os.path.exists(source_file):
                print(f"  ⚠ 소스 파일 없음: {source_file}")
                failed += 1
                continue

            # 신호 데이터 로드
            time_data, voltage_data = self.load_ultrasound_data(source_file)
            if len(voltage_data) == 0:
                failed += 1
                continue

            # 신호 전처리
            processed_signal = self.preprocess_signal(voltage_data,
                                                     normalize=True,
                                                     remove_dc=True,
                                                     filter_signal=True,
                                                     apply_denoising=False)

            # 레이블 정보 추출
            start_point = label_data.get('start_point_us', 0)
            positions = label_data.get('positions', [])

            if len(positions) < 2:
                print(f"  ⚠ 위치 정보 부족: {label_file}")
                failed += 1
                continue

            # Dermis (position 1)
            dermis_time = positions[0].get('time_us', 0)
            dermis_thickness = positions[0].get('thickness_mm', 0)

            # Fascia (position 2)
            fascia_time = positions[1].get('time_us', 0)
            fascia_thickness = positions[1].get('thickness_mm', 0)

            # 데이터셋에 추가
            dataset['signals'].append(processed_signal)
            dataset['time_data'].append(time_data)
            dataset['start_points'].append(start_point)
            dataset['dermis_times'].append(dermis_time)
            dataset['fascia_times'].append(fascia_time)
            dataset['dermis_thickness'].append(dermis_thickness)
            dataset['fascia_thickness'].append(fascia_thickness)
            dataset['file_names'].append(os.path.basename(source_file))

            successful += 1

            if successful % 50 == 0:
                print(f"  진행 중: {successful}개 처리 완료...")

        print(f"\n=== 데이터셋 생성 완료 ===")
        print(f"성공: {successful}개")
        print(f"실패: {failed}개")

        # numpy 배열로 변환 (signals는 길이가 다를 수 있으므로 object 타입)
        dataset_arrays = {
            'signals': np.array(dataset['signals'], dtype=object),
            'time_data': np.array(dataset['time_data'], dtype=object),
            'start_points': np.array(dataset['start_points']),
            'dermis_times': np.array(dataset['dermis_times']),
            'fascia_times': np.array(dataset['fascia_times']),
            'dermis_thickness': np.array(dataset['dermis_thickness']),
            'fascia_thickness': np.array(dataset['fascia_thickness']),
            'file_names': np.array(dataset['file_names'], dtype=object),
        }

        # 저장
        if output_file:
            np.savez(output_file, **dataset_arrays)
            print(f"\n데이터셋 저장됨: {output_file}")

        # 통계 정보
        stats = {
            'total_samples': successful,
            'avg_dermis_thickness': np.mean(dataset['dermis_thickness']),
            'std_dermis_thickness': np.std(dataset['dermis_thickness']),
            'avg_fascia_thickness': np.mean(dataset['fascia_thickness']),
            'std_fascia_thickness': np.std(dataset['fascia_thickness']),
        }

        print(f"\n=== 데이터셋 통계 ===")
        print(f"총 샘플 수: {stats['total_samples']}")
        print(f"진피 두께 평균: {stats['avg_dermis_thickness']:.4f}mm (±{stats['std_dermis_thickness']:.4f})")
        print(f"근막 두께 평균: {stats['avg_fascia_thickness']:.4f}mm (±{stats['std_fascia_thickness']:.4f})")

        return stats

def analyze_patient_data(patient_id: str, data_dir: str = "./data"):
    """특정 환자의 모든 8개 부위 데이터를 분석하고 표피 두께 일관성 비교"""
    print(f"\n=== {patient_id} 환자 데이터 분석 시작 ===")

    preprocessor = UltrasoundPreprocessor(sample_rate=5e6, speed_of_sound=1540.0)

    results = []
    epidermal_thicknesses = []

    # 8개 부위 데이터 분석
    for i in range(1, 9):
        file_name = f"{patient_id}-5M-{i}.csv"
        file_path = os.path.join(data_dir, file_name)

        if not os.path.exists(file_path):
            print(f"파일 없음: {file_path}")
            continue

        print(f"\n--- 부위 {i} 분석 중 ---")

        # 데이터 로드
        time_data, voltage_data = preprocessor.load_ultrasound_data(file_path)

        if len(voltage_data) == 0:
            continue

        # 전처리
        processed = preprocessor.preprocess_signal(voltage_data, normalize=True, remove_dc=True, filter_signal=True, apply_denoising=False)

        # 피크 검출 (최대 4개로 제한, 개선된 파라미터)
        # 동적 height 설정: 신호의 표준편차 기반
        dynamic_height = max(0.05, np.std(processed) * 0.3)  # 더 낮은 임계값
        peaks, peak_info = preprocessor.detect_peaks(processed, 
                                                   height=dynamic_height, 
                                                   distance=100,  # 더 가까운 거리 허용
                                                   max_peaks=4)

        # 두께 계산
        thickness_info = preprocessor.calculate_thickness(time_data, peaks)

        # 피부 두께 계산 (현실적인 값 기반)
        dermal_thickness = 0.0  # 진피 두께 (주요 측정값)
        epidermal_thickness = 0.0  # 표피 두께 (참고값)
        
        if len(peaks) >= 2:
            # 첫 번째와 두 번째 피크 사이 거리 = 진피 두께
            dermal_thickness = (time_data[peaks[1]] - time_data[peaks[0]]) * preprocessor.speed_of_sound / 2 * 1000
            
            # 표피 두께는 일반적인 값 범위로 설정 (직접 측정 어려움)
            # 실제로는 고주파 초음파로 측정하는 것이 일반적
            epidermal_thickness = 0.15  # 평균값 0.15mm (참고용)
        
        # 진피 두께를 주요 측정값으로 사용
        epidermal_thicknesses.append(dermal_thickness)

        result = {
            'position': i,
            'file': file_name,
            'peaks_detected': len(peaks),
            'avg_thickness': thickness_info['thickness_mm'],
            'dermal_thickness': dermal_thickness,  # 진피 두께 (주요 측정값)
            'epidermal_thickness': epidermal_thickness,  # 표피 두께 (참고값)
            'time_data': time_data,
            'processed_data': processed,
            'peaks': peaks
        }
        results.append(result)

        print(f"- 검출된 피크 수: {len(peaks)}")
        print(f"- 평균 두께: {thickness_info['thickness_mm']:.2f} mm")
        print(f"- 진피 두께: {dermal_thickness:.2f} mm")
        print(f"- 표피 두께 (참고): {epidermal_thickness:.2f} mm")

    # 진피 두께 일관성 분석
    if epidermal_thicknesses:
        valid_thicknesses = [t for t in epidermal_thicknesses if t > 0]
        if valid_thicknesses:
            mean_dermal = np.mean(valid_thicknesses)
            std_dermal = np.std(valid_thicknesses)
            cv_dermal = std_dermal / mean_dermal * 100 if mean_dermal > 0 else 0

            print("\n=== 진피 두께 일관성 분석 ===")
            print(f"- 평균 진피 두께: {mean_dermal:.2f} mm")
            print(f"- 표준편차: {std_dermal:.2f} mm")
            print(f"- 변동계수: {cv_dermal:.1f}%")
            print(f"- 측정된 부위 수: {len(valid_thicknesses)}/8")

            # 일관성 평가
            if cv_dermal < 10:
                consistency = "매우 일관됨 (CV < 10%)"
            elif cv_dermal < 20:
                consistency = "일관됨 (CV < 20%)"
            else:
                consistency = "불일관함 (CV >= 20%)"

            print(f"- 일관성 평가: {consistency}")

    return results, epidermal_thicknesses

def main():
    """메인 함수 - 초음파 데이터 분석 데모"""
    print("초음파 피부 조직 두께 측정 시스템")

    # bhjung 환자의 8개 부위 데이터 분석
    patient_id = "bhjung"
    results, epidermal_thicknesses = analyze_patient_data(patient_id)

    # 각 부위별 시각화 생성
    if results:
        print("\n=== 개별 부위 시각화 생성 ===")
        for result in results:
            patient_pos_id = f"{patient_id}_{result['position']}"
            save_path = f"./results/{patient_pos_id}_analysis.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            preprocessor = UltrasoundPreprocessor(sample_rate=5e6, speed_of_sound=1540.0)
            preprocessor.visualize_signal_and_peaks(
                result['time_data'],
                result['processed_data'],
                result['peaks'],
                patient_pos_id,
                save_path
            )
            print(f"- {patient_pos_id} 시각화 저장됨")

        # 모든 부위 중첩 시각화 생성
        print("\n=== 모든 부위 중첩 시각화 생성 ===")
        overlay_save_path = f"./results/{patient_id}_all_positions_overlay.png"
        os.makedirs(os.path.dirname(overlay_save_path), exist_ok=True)

        preprocessor = UltrasoundPreprocessor(sample_rate=5e6, speed_of_sound=1540.0)
        preprocessor.visualize_multiple_signals_overlay(results, patient_id, overlay_save_path)

    print("\n분석 완료!")

if __name__ == "__main__":
    main()