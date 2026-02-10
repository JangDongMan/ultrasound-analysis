"""
시리얼 통신을 통한 초음파 ADC 데이터 수집 모듈
"""

import serial
import serial.tools.list_ports
import time
import numpy as np
from typing import Optional, List, Tuple


class UltrasoundSerial:
    """초음파 ADC 데이터 수집을 위한 시리얼 통신 클래스"""

    def __init__(self, port: str = None, baudrate: int = 115200, timeout: float = 0.5):
        """
        Args:
            port: 시리얼 포트 (예: '/dev/ttyUSB0', 'COM3')
                  None이면 자동 검색
            baudrate: 통신 속도
            timeout: 읽기 타임아웃 (초)
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser: Optional[serial.Serial] = None

        # ADC 설정
        self.sample_interval_ns = 10  # 10ns 간격
        self.expected_samples = 4200  # 최대 샘플 수

    @staticmethod
    def list_ports() -> List[Tuple[str, str, str]]:
        """사용 가능한 시리얼 포트 목록 반환

        Returns:
            [(포트, 설명, 하드웨어ID), ...]
        """
        ports = serial.tools.list_ports.comports()
        return [(p.device, p.description, p.hwid) for p in ports]

    def connect(self) -> bool:
        """시리얼 포트 연결

        Returns:
            연결 성공 여부
        """
        if self.port is None:
            ports = self.list_ports()
            if not ports:
                print("사용 가능한 시리얼 포트가 없습니다.")
                return False
            self.port = ports[0][0]
            print(f"자동 선택된 포트: {self.port}")

        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            # 버퍼 비우기
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            print(f"연결됨: {self.port} @ {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"연결 실패: {e}")
            return False

    def disconnect(self):
        """시리얼 포트 연결 해제"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("연결 해제됨")

    def send_command(self, cmd: str) -> bool:
        """커맨드 전송

        Args:
            cmd: 전송할 커맨드 (개행 문자 자동 추가)

        Returns:
            전송 성공 여부
        """
        if not self.ser or not self.ser.is_open:
            print("시리얼 포트가 연결되지 않았습니다.")
            return False

        try:
            # 커맨드에 개행 문자 추가
            if not cmd.endswith('\n'):
                cmd += '\n'
            self.ser.write(cmd.encode('ascii'))
            self.ser.flush()
            return True
        except serial.SerialException as e:
            print(f"커맨드 전송 실패: {e}")
            return False

    def read_adc_data(self, num_samples: int = None, capture_timeout: float = 10.0,
                      callback=None) -> Tuple[np.ndarray, np.ndarray]:
        """VB5K ADC data receive

        VB5K 출력 순서:
          1. 커맨드 에코 (pwm start 5 1)
          2. FPGA capture done(0:...)
          3. 초기 쓰레기 값 (85, 204, 170, 51 등)
          4. VB5K > 프롬프트 ← 데이터 시작 마커
          5. 실제 ADC 데이터

        Phase 1: VB5K 프롬프트까지 스킵
        Phase 2: 프롬프트 이후 ADC 데이터 수집 (idle timeout으로 종료)

        Args:
            num_samples: max samples to read (None = read until end)
            capture_timeout: 전체 캡처 타임아웃 (초)
            callback: callback(current_count, value) per sample

        Returns:
            (time_ns, adc_values)
        """
        if not self.ser or not self.ser.is_open:
            return np.array([]), np.array([])

        if num_samples is None:
            num_samples = self.expected_samples

        start_time = time.time()

        # Phase 1: FPGA 확인 후 VB5K 프롬프트까지 스킵
        # FPGA는 커맨드 실행에서만 나오므로, wait_ready에서 누출된 VB5K와 구분 가능
        fpga_found = False
        prompt_found = False
        while time.time() - start_time < capture_timeout:
            try:
                line = self.ser.readline().decode('ascii', errors='ignore').strip()
                if not line:
                    continue
                if 'FPGA' in line:
                    fpga_found = True
                if fpga_found and 'VB5K' in line:
                    prompt_found = True
                    break
            except serial.SerialException as e:
                print(f"Read error: {e}")
                break

        if not prompt_found:
            if not fpga_found:
                print("Timeout: FPGA response not found")
            else:
                print("Timeout: VB5K prompt not found")
            return np.array([]), np.array([])

        # Phase 2: 프롬프트 이후 ADC 데이터 수집
        values = []
        last_data_time = time.time()
        IDLE_TIMEOUT = 1.0

        while len(values) < num_samples:
            if time.time() - start_time > capture_timeout:
                print(f"Timeout: {len(values)} samples received")
                break

            # idle timeout: 데이터 수신 후 1초간 데이터 없으면 종료
            if len(values) > 0 and (time.time() - last_data_time) > IDLE_TIMEOUT:
                print(f"Done: {len(values)} samples")
                break

            try:
                line = self.ser.readline().decode('ascii', errors='ignore').strip()
                if not line:
                    continue

                for token in line.split():
                    try:
                        value = int(token)
                        if 0 <= value <= 255:
                            values.append(value)
                            last_data_time = time.time()
                            if callback:
                                callback(len(values), value)
                    except ValueError:
                        pass

            except UnicodeDecodeError:
                pass
            except serial.SerialException as e:
                print(f"Read error: {e}")
                break

        # Convert to numpy arrays
        adc_values = np.array(values, dtype=np.int32)
        time_ns = np.arange(len(values)) * self.sample_interval_ns

        return time_ns, adc_values

    def flush_buffer(self):
        """시리얼 버퍼 완전히 비우기"""
        if not self.ser or not self.ser.is_open:
            return
        self.ser.reset_input_buffer()
        time.sleep(0.1)
        while self.ser.in_waiting > 0:
            self.ser.read(self.ser.in_waiting)
            time.sleep(0.05)
        self.ser.reset_input_buffer()

    def capture(self, command: str = "pwm start 5 1", num_samples: int = None,
                capture_timeout: float = 10.0, callback=None) -> Tuple[np.ndarray, np.ndarray]:
        """Send command and capture ADC data from VB5K

        Args:
            command: command to send (must start with "pwm")
            num_samples: max samples to read
            capture_timeout: 전체 캡처 타임아웃 (초)
            callback: progress callback

        Returns:
            (time_ns, adc_values)
        """
        # 버퍼 비우기
        self.flush_buffer()

        # Send command
        if not self.send_command(command):
            return np.array([]), np.array([])

        # Receive data
        time_ns, adc_values = self.read_adc_data(num_samples, capture_timeout, callback)

        return time_ns, adc_values

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
        return False


def save_to_csv(filename: str, time_ns: np.ndarray, adc_values: np.ndarray,
                metadata: dict = None):
    """ADC 데이터를 CSV 파일로 저장 (콘솔 출력 형식: ADC 값만 한 줄에 하나씩)

    Args:
        filename: 저장할 파일 경로
        time_ns: 시간 배열 (나노초) - 미사용, 인터페이스 호환용
        adc_values: ADC 값 배열
        metadata: 추가 메타데이터 - 미사용, 인터페이스 호환용
    """
    with open(filename, 'w') as f:
        for adc in adc_values:
            f.write(f"{int(adc)}\n")

    print(f"Saved: {filename} ({len(adc_values)} samples)")


def load_from_csv(filename: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """CSV 파일에서 ADC 데이터 로드 (3가지 형식 지원)

    지원 형식:
        1. ADC 값만 (한 줄에 하나씩, 0-255)
        2. 기존 형식 (x-axis,1 / second,Volt)
        3. time_ns,adc_value 형식

    Args:
        filename: 파일 경로

    Returns:
        (time_ns, adc_values, metadata) 튜플
    """
    SAMPLE_INTERVAL_NS = 10
    ADC_MAX = 255
    VREF = 1.25

    metadata = {}
    adc_values = []
    is_compatible_format = False
    is_time_adc_format = False

    with open(filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 기존 형식 감지
        if line == "x-axis,1":
            is_compatible_format = True
            metadata['format'] = 'compatible'
            continue
        if line == "second,Volt":
            continue
        if line == "time_ns,adc_value":
            is_time_adc_format = True
            continue

        if line.startswith('#'):
            if ':' in line:
                key, value = line[1:].split(':', 1)
                metadata[key.strip()] = value.strip()
            continue

        parts = line.split(',')

        if is_compatible_format and len(parts) >= 2:
            try:
                voltage = float(parts[1])
                adc_values.append(int((voltage / VREF) * ADC_MAX))
            except ValueError:
                pass
        elif is_time_adc_format and len(parts) >= 2:
            try:
                adc_values.append(int(parts[1]))
            except ValueError:
                pass
        elif len(parts) == 1:
            # ADC 값만 있는 형식
            try:
                value = int(parts[0])
                if 0 <= value <= 255:
                    adc_values.append(value)
            except ValueError:
                pass

    adc_arr = np.array(adc_values, dtype=np.int32)
    time_ns = np.arange(len(adc_values)) * SAMPLE_INTERVAL_NS

    return time_ns, adc_arr, metadata


if __name__ == "__main__":
    # 사용 가능한 포트 출력
    print("사용 가능한 시리얼 포트:")
    for port, desc, hwid in UltrasoundSerial.list_ports():
        print(f"  {port}: {desc}")

    # 테스트 (실제 장치 연결 시)
    # with UltrasoundSerial('/dev/ttyUSB0') as us:
    #     time_ns, adc_values = us.capture("GET_ADC", num_samples=2000)
    #     save_to_csv("test_capture.csv", time_ns, adc_values)
