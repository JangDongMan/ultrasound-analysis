"""
C++ 기반 피부층 경계 검출 알고리즘의 Python 래퍼
"""

import ctypes
import numpy as np
import os

# 결과 구조체 정의
class DetectionResult(ctypes.Structure):
    _fields_ = [
        ("dermis_idx", ctypes.c_int),
        ("fascia_idx", ctypes.c_int),
        ("success", ctypes.c_bool)
    ]


class BoundaryDetector:
    """통계 기반 피부층 경계 검출기"""

    def __init__(self, lib_path=None):
        """
        Args:
            lib_path: 공유 라이브러리 경로 (None이면 자동 검색)
        """
        if lib_path is None:
            # 현재 디렉토리에서 라이브러리 찾기
            current_dir = os.path.dirname(os.path.abspath(__file__))
            lib_name = "libboundary_detector.so"
            lib_path = os.path.join(current_dir, lib_name)

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"Shared library not found: {lib_path}")

        # 라이브러리 로드
        self.lib = ctypes.CDLL(lib_path)

        # 함수 시그니처 설정
        self.lib.detect_boundaries_c.argtypes = [
            ctypes.POINTER(ctypes.c_double),  # time_us
            ctypes.POINTER(ctypes.c_double),  # voltage
            ctypes.c_int,                      # n
            ctypes.c_double                    # reference_start_us
        ]
        self.lib.detect_boundaries_c.restype = DetectionResult

    def detect(self, time_us, voltage, reference_start_us):
        """
        피부층 경계 검출

        Args:
            time_us: 시간 데이터 (μs), numpy array
            voltage: 전압 데이터, numpy array
            reference_start_us: 시작점 시간 (μs)

        Returns:
            tuple: (dermis_idx, fascia_idx, success)
                - dermis_idx: 진피 인덱스
                - fascia_idx: 근막 인덱스
                - success: 검출 성공 여부
        """
        # numpy array를 C 타입으로 변환
        time_us = np.ascontiguousarray(time_us, dtype=np.float64)
        voltage = np.ascontiguousarray(voltage, dtype=np.float64)

        n = len(time_us)
        if n != len(voltage):
            raise ValueError("time_us and voltage must have the same length")

        # C 함수 호출
        time_ptr = time_us.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        voltage_ptr = voltage.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

        result = self.lib.detect_boundaries_c(
            time_ptr,
            voltage_ptr,
            n,
            reference_start_us
        )

        return result.dermis_idx, result.fascia_idx, result.success


def detect_skin_boundaries(time_us, voltage, reference_start_us, lib_path=None):
    """
    편의 함수: 피부층 경계 검출

    Args:
        time_us: 시간 데이터 (μs)
        voltage: 전압 데이터
        reference_start_us: 시작점 시간 (μs)
        lib_path: 라이브러리 경로 (옵션)

    Returns:
        tuple: (dermis_idx, fascia_idx, success)
    """
    detector = BoundaryDetector(lib_path)
    return detector.detect(time_us, voltage, reference_start_us)


if __name__ == "__main__":
    # 테스트 코드
    print("Testing boundary detector wrapper...")

    # 더미 데이터 생성
    time_us = np.linspace(0, 10, 1000)
    voltage = np.sin(time_us) + np.random.normal(0, 0.1, 1000)

    try:
        dermis_idx, fascia_idx, success = detect_skin_boundaries(
            time_us, voltage, reference_start_us=0.0
        )

        if success:
            print(f"✓ Detection successful!")
            print(f"  Dermis index: {dermis_idx}")
            print(f"  Fascia index: {fascia_idx}")
        else:
            print("✗ Detection failed")

    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print("  Please compile the C++ library first:")
        print("  cd utils2 && ./build.sh")
