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
                stopbits=serial.STOPBITS_ONE,
                rtscts=False,   # HW flow control 비활성화
                dsrdtr=False,   # DSR/DTR flow control 비활성화
                xonxoff=False,  # SW flow control 비활성화
            )
            # DTR=HIGH 유지 (터미널과 동일, LOW 시 디바이스 응답 안 함)
            self.ser.dtr = True
            self.ser.rts = True
            # 버퍼 비우기
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            print(f"연결됨: {self.port} @ {self.baudrate} baud (DTR=on, RTS=on)")
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
        """커맨드 전송 (한 글자씩 5ms 간격)

        디바이스가 TX 중일 때 RX 오버플로우를 방지하기 위해
        각 문자를 개별적으로 전송.

        Args:
            cmd: 전송할 커맨드 (개행 문자 자동 추가)

        Returns:
            전송 성공 여부
        """
        if not self.ser or not self.ser.is_open:
            print("시리얼 포트가 연결되지 않았습니다.")
            return False

        try:
            # 터미널은 Enter = CR+LF(\r\n) 전송 → 디바이스가 \r\n 기대할 수 있음
            cmd = cmd.rstrip('\r\n') + '\r\n'
            encoded = cmd.encode('ascii')
            print(f"  TX ({len(encoded)}B): {repr(cmd)}")
            print(f"  TX hex: {' '.join(f'{b:02X}' for b in encoded)}")
            # 한 글자씩 5ms 간격으로 전송
            # 디바이스 UART가 TX 중이어도 각 문자 수신 가능
            for ch in encoded:
                self.ser.write(bytes([ch]))
                time.sleep(0.005)
            self.ser.flush()
            # 전송 후 100ms 대기하여 에코/응답 확인 (peek only, 버퍼는 유지)
            time.sleep(0.1)
            n = self.ser.in_waiting
            if n > 0:
                peek = self.ser.read(n)
                print(f"  RX after TX ({n}B): buffered for Phase1 parsing")
                # 읽은 데이터를 다시 앞에 넣을 수 없으므로 Phase 1에서 처리할 수 있게
                # _pending_echo에 저장
                self._pending_echo = peek
            else:
                print(f"  RX after TX: (none)")
                self._pending_echo = b''
            return True
        except serial.SerialException as e:
            print(f"커맨드 전송 실패: {e}")
            return False

    def read_adc_data(self, num_samples: int = None, capture_timeout: float = 10.0,
                      callback=None) -> Tuple[np.ndarray, np.ndarray]:
        """VB5K ADC data receive

        VB5K 출력 순서:
          1. 커맨드 에코 (pwm start 5 1) - VB5K 프롬프트 포함 가능
          2. FPGA capture done(0:...)
          3. 초기 쓰레기 값 (85, 204, 170, 51 등)
          4. VB5K > 프롬프트 ← 데이터 시작 마커
          5. 실제 ADC 데이터

        Phase 1: FPGA 확인 후 VB5K 프롬프트까지 스킵
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

        # send_command()가 에코를 미리 읽었으면 먼저 처리
        pending = getattr(self, '_pending_echo', b'')
        self._pending_echo = b''

        # Phase 1: FPGA 확인 후 데이터 시작 감지
        #   - 'pwm start 5 1': FPGA → 쓰레기값 → VB5K 프롬프트 → 데이터
        #   - 'pwm start ulso': FPGA → 쓰레기값 → 바로 데이터 (VB5K 프롬프트 없음)
        fpga_found = False
        prompt_found = False
        phase1_lines = 0
        # Phase 2 변수를 미리 초기화 (Phase 1에서 데이터가 시작될 수 있음)
        values = []
        last_data_time = time.time()
        IDLE_TIMEOUT = 1.0

        # pending echo 데이터를 완전히 파싱 (FPGA 감지 + 데이터 값 수집)
        TERM_SEQ = [0xFE, 0x7F, 0x01, 0x7F]
        term_found = False
        if pending:
            print(f"  [pending {len(pending)}B lines]:")
            for line in pending.decode('ascii', errors='ignore').splitlines():
                line = line.strip()
                if not line:
                    continue
                print(f"    P> {line[:60]}")
                phase1_lines += 1
                if 'FPGA' in line:
                    fpga_found = True
                    continue
                if fpga_found and 'VB5K' in line:
                    prompt_found = True
                    continue
                if fpga_found:
                    for token in line.split():
                        try:
                            value = int(token)
                            if 0 <= value <= 255:
                                values.append(value)
                                last_data_time = time.time()
                                if callback:
                                    callback(len(values), value)
                                if not prompt_found:
                                    prompt_found = True
                                if (len(values) >= max(100, num_samples - 500) and
                                        values[-4:] == TERM_SEQ):
                                    del values[-4:]
                                    term_found = True
                        except ValueError:
                            pass
                    if term_found:
                        break

        if not prompt_found:
            # pending에서 데이터를 못 찾았으면 Phase 1 while loop 진행
            pass

        while not prompt_found and time.time() - start_time < capture_timeout:
            try:
                line = self.ser.readline().decode('ascii', errors='ignore').strip()
                if not line:
                    continue
                phase1_lines += 1
                print(f"  [ph1 L{phase1_lines}] {line[:60]}")
                if 'FPGA' in line:
                    fpga_found = True
                    continue
                if fpga_found and 'VB5K' in line:
                    # 'pwm start 5 1' 방식: VB5K 프롬프트가 데이터 시작 마커
                    prompt_found = True
                    break
                if fpga_found:
                    # 'pwm start ulso' 방식: VB5K 없이 바로 숫자 데이터가 시작됨
                    has_data = False
                    for token in line.split():
                        try:
                            value = int(token)
                            if 0 <= value <= 255:
                                values.append(value)
                                last_data_time = time.time()
                                if callback:
                                    callback(len(values), value)
                                has_data = True
                        except ValueError:
                            pass
                    if has_data:
                        prompt_found = True
                        break
            except serial.SerialException as e:
                print(f"Read error: {e}")
                break

        if not prompt_found:
            if not fpga_found:
                print(f"Timeout: FPGA not found ({phase1_lines} lines read)")
                # 디버그: 실제로 무엇을 받았는지 출력
                n = self.ser.in_waiting
                print(f"  in_waiting after timeout: {n} bytes")
                if n > 0:
                    raw = self.ser.read(min(n, 200))
                    print(f"  raw bytes: {raw[:80]}")
            else:
                print(f"Timeout: VB5K not found after FPGA ({phase1_lines} lines)")
            return np.array([]), np.array([])

        # Phase 2: 데이터 수집 계속 (values에 Phase 1/pending에서 넣은 값도 포함됨)
        # TERM_SEQ, term_found는 pending 처리 시 이미 선언됨
        print(f"  Phase2 start: values so far={len(values)}, term_found={term_found}")

        while not term_found and len(values) < num_samples:
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
                            # 종결 시퀀스 감지: 충분한 샘플 수집 후에만 확인 (중간 데이터 오검출 방지)
                            if (len(values) >= max(100, num_samples - 500) and
                                    values[-4:] == TERM_SEQ):
                                del values[-4:]
                                term_found = True
                                break
                    except ValueError:
                        pass
                if term_found:
                    print(f"Done (term seq): {len(values)} samples")
                    break

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
        """시리얼 버퍼 완전히 비우기 - 1초간 데이터 없을 때까지 drain"""
        if not self.ser or not self.ser.is_open:
            return
        self.ser.reset_input_buffer()
        last_rx = time.time()
        while time.time() - last_rx < 1.0:
            n = self.ser.in_waiting
            if n > 0:
                self.ser.read(n)
                last_rx = time.time()
            time.sleep(0.02)
        self.ser.reset_input_buffer()

    def wait_ready(self, timeout: float = 3.0) -> bool:
        """VB5K 프롬프트 확인하여 디바이스 idle 상태 확인

        readline() 대신 누적 버퍼 방식으로 'VB5K' 감지:
        - VB5K > 프롬프트는 줄바꿈 없이 오는 경우가 있어
          readline()이 timeout까지 기다리는 문제 방지

        Returns:
            준비 완료 여부
        """
        if not self.ser or not self.ser.is_open:
            return False

        # 잔여 데이터 제거
        self.ser.reset_input_buffer()
        time.sleep(0.1)
        self.ser.reset_input_buffer()

        # Enter 전송 → VB5K 프롬프트 응답 대기 (터미널과 동일하게 CR+LF)
        self.ser.write(b'\r\n')
        self.ser.flush()

        buf = b''
        start = time.time()
        while time.time() - start < timeout:
            try:
                n = self.ser.in_waiting
                if n > 0:
                    buf += self.ser.read(n)
                    text = buf.decode('ascii', errors='ignore')
                    print(f"  wait_ready rx: {repr(text[-40:])}")
                    if 'VB5K' in text:
                        self.ser.reset_input_buffer()
                        return True
                else:
                    time.sleep(0.02)
            except serial.SerialException:
                return False

        print(f"  wait_ready timeout, buf={repr(buf[:80])}")
        return False

    def capture(self, command: str = "pwm start 5 1", num_samples: int = None,
                capture_timeout: float = 10.0, callback=None) -> Tuple[np.ndarray, np.ndarray]:
        """Send command and capture ADC data from VB5K

        디바이스 UART는 TX 중 RX 오버플로우 가능성이 있으므로:
        1. wait_ready로 디바이스 idle 확인
        2. 커맨드를 한 글자씩 5ms 간격으로 전송

        Args:
            command: command to send (must start with "pwm")
            num_samples: max samples to read
            capture_timeout: 전체 캡처 타임아웃 (초)
            callback: progress callback

        Returns:
            (time_ns, adc_values)
        """
        # 1. 디바이스 idle 확인
        if not self.wait_ready():
            print("Warning: Device not ready, flushing and retrying")
            # flush_buffer() 대신 input buffer만 비우고 진행
            # (flush_buffer는 1초 drain → 디바이스 응답 데이터도 버림)
            self.ser.reset_input_buffer()
            time.sleep(0.1)

        # 2. 커맨드 전송 (한 글자씩 5ms 간격, UART RX 오버플로우 방지)
        if not self.send_command(command):
            return np.array([]), np.array([])

        print(f"  command sent, waiting for response...")

        # 3. 응답 수신
        return self.read_adc_data(num_samples, capture_timeout, callback)

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
