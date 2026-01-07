#!/usr/bin/env python3
"""
CNN-based Ultrasound Boundary Detection Model
피부층 경계 자동 검출을 위한 1D CNN 모델
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import os


class BoundaryDetectionCNN:
    """1D CNN 기반 초음파 경계 검출 모델"""

    def __init__(self, input_length=2000, speed_of_sound=1540.0):
        """
        Args:
            input_length: 입력 신호 길이
            speed_of_sound: 음속 (m/s)
        """
        self.input_length = input_length
        self.speed_of_sound = speed_of_sound
        self.model = None
        self.history = None

    def build_model(self):
        """1D CNN 모델 구축"""

        inputs = keras.Input(shape=(self.input_length, 1), name='signal_input')

        # 1D Convolutional layers
        x = layers.Conv1D(32, kernel_size=11, activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        x = layers.Conv1D(64, kernel_size=7, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        x = layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        x = layers.Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling1D()(x)

        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

        # Output: 2개 경계 위치 (dermis_time, fascia_time in μs)
        outputs = layers.Dense(2, activation='linear', name='boundary_positions')(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name='boundary_detector')

        # 모델 컴파일
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return self.model

    def prepare_data(self, dataset_file='training_dataset.npz'):
        """
        데이터셋 로드 및 전처리

        Returns:
            X_train, X_val, y_train, y_val
        """
        print(f"\n{'='*60}")
        print("데이터셋 로드 및 전처리")
        print(f"{'='*60}\n")

        # 데이터 로드
        data = np.load(dataset_file, allow_pickle=True)

        signals = data['signals']
        start_points = data['start_points']
        dermis_times = data['dermis_times']
        fascia_times = data['fascia_times']

        print(f"총 샘플 수: {len(signals)}")
        print(f"신호 길이 범위: {min([len(s) for s in signals])} ~ {max([len(s) for s in signals])}")

        # 데이터 준비
        X = []
        y = []

        for i in range(len(signals)):
            signal = signals[i]

            # 신호 길이 맞추기 (패딩 또는 자르기)
            if len(signal) < self.input_length:
                # 패딩
                padded = np.zeros(self.input_length)
                padded[:len(signal)] = signal
                signal = padded
            else:
                # 자르기
                signal = signal[:self.input_length]

            X.append(signal)

            # 레이블: 시작점 대비 상대 시간 (μs)
            dermis_relative = dermis_times[i] - start_points[i]
            fascia_relative = fascia_times[i] - start_points[i]

            y.append([dermis_relative, fascia_relative])

        X = np.array(X)
        y = np.array(y)

        # 신호 정규화
        X = X.reshape(-1, self.input_length, 1)

        print(f"\n입력 형태: {X.shape}")
        print(f"출력 형태: {y.shape}")
        print(f"레이블 범위:")
        print(f"  Dermis: {y[:, 0].min():.2f} ~ {y[:, 0].max():.2f} μs")
        print(f"  Fascia: {y[:, 1].min():.2f} ~ {y[:, 1].max():.2f} μs")

        # Train/Validation split (80/20)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\n학습 데이터: {len(X_train)} 샘플")
        print(f"검증 데이터: {len(X_val)} 샘플")

        return X_train, X_val, y_train, y_val

    def train(self, X_train, y_train, X_val, y_val,
              epochs=100, batch_size=16, patience=15):
        """
        모델 학습

        Args:
            X_train, y_train: 학습 데이터
            X_val, y_val: 검증 데이터
            epochs: 최대 에포크 수
            batch_size: 배치 크기
            patience: Early stopping patience
        """
        print(f"\n{'='*60}")
        print("모델 학습 시작")
        print(f"{'='*60}\n")

        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # 학습
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def evaluate(self, X_val, y_val):
        """
        모델 평가

        Returns:
            평가 결과 딕셔너리
        """
        print(f"\n{'='*60}")
        print("모델 평가")
        print(f"{'='*60}\n")

        # 예측
        y_pred = self.model.predict(X_val, verbose=0)

        # 전체 손실
        loss, mae = self.model.evaluate(X_val, y_val, verbose=0)

        # 개별 경계 평가
        dermis_mae = mean_absolute_error(y_val[:, 0], y_pred[:, 0])
        fascia_mae = mean_absolute_error(y_val[:, 1], y_pred[:, 1])

        dermis_rmse = np.sqrt(mean_squared_error(y_val[:, 0], y_pred[:, 0]))
        fascia_rmse = np.sqrt(mean_squared_error(y_val[:, 1], y_pred[:, 1]))

        # 시간을 거리로 변환 (μs -> mm)
        dermis_mae_mm = dermis_mae * 1e-6 * self.speed_of_sound / 2 * 1000
        fascia_mae_mm = fascia_mae * 1e-6 * self.speed_of_sound / 2 * 1000

        dermis_rmse_mm = dermis_rmse * 1e-6 * self.speed_of_sound / 2 * 1000
        fascia_rmse_mm = fascia_rmse * 1e-6 * self.speed_of_sound / 2 * 1000

        results = {
            'loss': loss,
            'mae': mae,
            'dermis_mae_us': dermis_mae,
            'fascia_mae_us': fascia_mae,
            'dermis_mae_mm': dermis_mae_mm,
            'fascia_mae_mm': fascia_mae_mm,
            'dermis_rmse_us': dermis_rmse,
            'fascia_rmse_us': fascia_rmse,
            'dermis_rmse_mm': dermis_rmse_mm,
            'fascia_rmse_mm': fascia_rmse_mm
        }

        print(f"전체 손실 (MSE): {loss:.4f}")
        print(f"전체 MAE: {mae:.4f} μs")
        print(f"\n경계별 성능:")
        print(f"  Dermis MAE: {dermis_mae:.4f} μs ({dermis_mae_mm:.4f} mm)")
        print(f"  Fascia MAE: {fascia_mae:.4f} μs ({fascia_mae_mm:.4f} mm)")
        print(f"  Dermis RMSE: {dermis_rmse:.4f} μs ({dermis_rmse_mm:.4f} mm)")
        print(f"  Fascia RMSE: {fascia_rmse:.4f} μs ({fascia_rmse_mm:.4f} mm)")

        return results, y_pred

    def plot_training_history(self, save_path='results/training_history.png'):
        """학습 이력 시각화"""

        if self.history is None:
            print("학습 이력이 없습니다.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax1.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        ax1.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss (MSE)', fontsize=12)
        ax1.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # MAE
        ax2.plot(self.history.history['mae'], label='Train MAE', linewidth=2)
        ax2.plot(self.history.history['val_mae'], label='Val MAE', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('MAE (μs)', fontsize=12)
        ax2.set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n학습 이력 저장: {save_path}")
        plt.close()

    def plot_predictions(self, y_val, y_pred, save_path='results/predictions.png'):
        """예측 결과 시각화"""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # Dermis 산점도
        ax1.scatter(y_val[:, 0], y_pred[:, 0], alpha=0.5, s=30)
        ax1.plot([y_val[:, 0].min(), y_val[:, 0].max()],
                [y_val[:, 0].min(), y_val[:, 0].max()],
                'r--', linewidth=2, label='Perfect Prediction')
        ax1.set_xlabel('True Dermis Time (μs)', fontsize=11)
        ax1.set_ylabel('Predicted Dermis Time (μs)', fontsize=11)
        ax1.set_title('Dermis Boundary Prediction', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Fascia 산점도
        ax2.scatter(y_val[:, 1], y_pred[:, 1], alpha=0.5, s=30)
        ax2.plot([y_val[:, 1].min(), y_val[:, 1].max()],
                [y_val[:, 1].min(), y_val[:, 1].max()],
                'r--', linewidth=2, label='Perfect Prediction')
        ax2.set_xlabel('True Fascia Time (μs)', fontsize=11)
        ax2.set_ylabel('Predicted Fascia Time (μs)', fontsize=11)
        ax2.set_title('Fascia Boundary Prediction', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Dermis 오차 분포
        dermis_errors = y_pred[:, 0] - y_val[:, 0]
        ax3.hist(dermis_errors, bins=30, edgecolor='black', alpha=0.7)
        ax3.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax3.set_xlabel('Prediction Error (μs)', fontsize=11)
        ax3.set_ylabel('Frequency', fontsize=11)
        ax3.set_title(f'Dermis Error Distribution (μ={dermis_errors.mean():.2f}μs, σ={dermis_errors.std():.2f}μs)',
                     fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        # Fascia 오차 분포
        fascia_errors = y_pred[:, 1] - y_val[:, 1]
        ax4.hist(fascia_errors, bins=30, edgecolor='black', alpha=0.7)
        ax4.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
        ax4.set_xlabel('Prediction Error (μs)', fontsize=11)
        ax4.set_ylabel('Frequency', fontsize=11)
        ax4.set_title(f'Fascia Error Distribution (μ={fascia_errors.mean():.2f}μs, σ={fascia_errors.std():.2f}μs)',
                     fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"예측 결과 저장: {save_path}")
        plt.close()

    def save_model(self, model_path='saved_models/boundary_detector.keras'):
        """모델 저장"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"\n모델 저장: {model_path}")

    def load_model(self, model_path='saved_models/boundary_detector.keras'):
        """모델 로드"""
        self.model = keras.models.load_model(model_path)
        print(f"모델 로드: {model_path}")


def main():
    """메인 실행 함수"""

    print("\n" + "="*60)
    print("CNN-Based Ultrasound Boundary Detection")
    print("="*60 + "\n")

    # 모델 생성
    detector = BoundaryDetectionCNN(input_length=2000, speed_of_sound=1540.0)

    # 모델 구축
    model = detector.build_model()
    print("\n모델 구조:")
    model.summary()

    # 데이터 준비
    X_train, X_val, y_train, y_val = detector.prepare_data('training_dataset.npz')

    # 학습
    history = detector.train(
        X_train, y_train,
        X_val, y_val,
        epochs=100,
        batch_size=16,
        patience=15
    )

    # 평가
    results, y_pred = detector.evaluate(X_val, y_val)

    # 시각화
    detector.plot_training_history('results/training_history.png')
    detector.plot_predictions(y_val, y_pred, 'results/predictions.png')

    # 모델 저장
    detector.save_model('saved_models/boundary_detector.keras')

    # 결과 저장
    results_file = 'results/model_evaluation.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n평가 결과 저장: {results_file}")

    print("\n" + "="*60)
    print("학습 완료!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
