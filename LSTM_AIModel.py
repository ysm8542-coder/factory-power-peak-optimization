#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM_AIModel.py
===============
다변량 LSTM 전력 수요 예측 모델 (기본 모델).

직전 24시간의 전력 · 생산 · 기상 등 13개 변수로 다음 1시간 전력(`평균`, kW)을 예측한다.
`LSTM_AIModel.ipynb`의 공통 함수를 모듈로 정리한 것으로, 다른 노트북에서 import 해서 사용한다.

사용 예
-------
    from LSTM_AIModel import train_scenario_d
    result = train_scenario_d("okm_augumented_2021_processed.xlsx")
    print(result["metrics"])          # RMSE, MAE, MAPE(%), R2

직접 실행하면 학습 구간 시나리오 A~D의 성능을 비교해 출력한다.
    $ python LSTM_AIModel.py
"""
import os
import platform
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------------------------------------------------------
# 한글 폰트 설정
# ---------------------------------------------------------------------------
if platform.system() == "Darwin":
    plt.rc("font", family="AppleGothic")
elif platform.system() == "Windows":
    plt.rc("font", family="Malgun Gothic")
else:
    plt.rc("font", family="NanumGothic")
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# 공통 설정값
#   다른 파일에서 `from LSTM_AIModel import FEATURES, TARGET, WINDOW_SIZE` 로 재사용
# ---------------------------------------------------------------------------
DEFAULT_FILE_PATH = "okm_augumented_2021_processed.xlsx"
FEATURES = ["평균", "생산량", "기온", "풍속", "습도", "강수량", "전기요금(계절)",
            "공장인원", "시간", "전력 구간합계", "전력 최대값", "전력 최소값", "전력 변동폭"]   # 입력 변수 13개
TARGET = "평균"          # 예측 대상: 시간 평균 전력(kW)
WINDOW_SIZE = 24         # 입력 시퀀스 길이(시간)


# ---------------------------------------------------------------------------
# 데이터 로드
# ---------------------------------------------------------------------------
def load_data(file_path: str = DEFAULT_FILE_PATH) -> pd.DataFrame:
    """엑셀을 읽어 datetime 정렬 · 결측 보간 후 1~9월 데이터(df_1_9)를 반환한다."""
    df = pd.read_excel(file_path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.interpolate(method="linear").dropna()

    df_1_9 = df[(df["datetime"].dt.month >= 1) & (df["datetime"].dt.month <= 9)].copy()
    return df_1_9.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 시퀀스 생성
# ---------------------------------------------------------------------------
def make_sequences(data, target_idx, window_size=WINDOW_SIZE):
    """X = data[i : i+W] (W시간 입력), y = data[i+W, target_idx] (다음 1시간 target)."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size, target_idx])
    return np.array(X), np.array(y)


# ---------------------------------------------------------------------------
# 모델 학습
# ---------------------------------------------------------------------------
def train_lstm_model(X_train, y_train, input_shape):
    """LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(16) → Dense(1), EarlyStopping(patience=10)."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    return model


# ---------------------------------------------------------------------------
# 모델 평가
# ---------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, scaler, target_idx, features):
    """예측 후 원래 단위(kW)로 역변환해 (지표 dict, y_true, y_pred)를 반환한다."""
    y_pred_scaled = model.predict(X_test, verbose=0)

    # 역스케일링용 더미 배열의 열 수는 scaler가 학습한 컬럼 수를 따른다.
    # (features + [target] 으로 스케일링했다면 n_features_in_ = len(features) + 1)
    n_features = scaler.n_features_in_ if hasattr(scaler, "n_features_in_") else len(features)
    if target_idx >= n_features:
        target_idx = 0

    dummy_array = np.zeros((len(y_pred_scaled), n_features))
    dummy_array[:, target_idx] = y_pred_scaled.flatten()
    y_pred = scaler.inverse_transform(dummy_array)[:, target_idx]

    dummy_test_y = np.zeros((len(y_test), n_features))
    dummy_test_y[:, target_idx] = np.asarray(y_test).flatten()
    y_true = scaler.inverse_transform(dummy_test_y)[:, target_idx]

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)

    return {"RMSE": rmse, "MAE": mae, "MAPE(%)": mape, "R2": r2}, y_true, y_pred


# ---------------------------------------------------------------------------
# 시나리오 D (1~9월 전체, 80:20 분할)
# ---------------------------------------------------------------------------
def prepare_scenario_d_data(file_path: str = DEFAULT_FILE_PATH,
                             features=None, target=TARGET, window_size=WINDOW_SIZE):
    """시나리오 D의 학습/테스트 시퀀스와 스케일러를 준비해 dict로 반환한다."""
    features = features or FEATURES
    df_1_9 = load_data(file_path)

    data_values = df_1_9[features + [target]].values
    target_idx = len(features)  # target을 features 뒤에 붙였으므로 마지막 열

    # 시간 순서 80:20 분할 (테스트는 시퀀스 생성을 위해 window_size 만큼 앞에서 시작)
    train_size = int(len(data_values) * 0.8)
    train_data = data_values[:train_size]
    test_data = data_values[train_size - window_size:]

    # 학습 구간으로만 스케일러 fit (데이터 누수 방지)
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = make_sequences(train_scaled, target_idx, window_size)
    X_test, y_test = make_sequences(test_scaled, target_idx, window_size)

    return {
        "df_1_9": df_1_9,
        "X_train": X_train, "y_train": y_train,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler, "target_idx": target_idx,
        "features": features, "target": target, "window_size": window_size,
        "train_size": train_size,
    }


def train_scenario_d(file_path: str = DEFAULT_FILE_PATH, **kwargs):
    """시나리오 D 데이터 준비 → 학습 → 평가를 한 번에 수행한다."""
    data = prepare_scenario_d_data(file_path, **kwargs)
    model = train_lstm_model(data["X_train"], data["y_train"],
                              (data["X_train"].shape[1], data["X_train"].shape[2]))
    metrics, y_true, y_pred = evaluate_model(model, data["X_test"], data["y_test"],
                                              data["scaler"], data["target_idx"], data["features"])
    data.update({"model": model, "metrics": metrics, "y_true": y_true, "y_pred": y_pred})
    return data


# ---------------------------------------------------------------------------
# 학습 구간 시나리오 A~D 비교 (스크립트 직접 실행 시)
# ---------------------------------------------------------------------------
def _run_all_scenarios_comparison():
    """시나리오별 · 구간별로 80:20 학습/평가해 성능 표를 출력하고 반환한다."""
    df_1_9 = load_data()

    scenarios = {
        "시나리오 A: 2개 구간 (1~4월 / 5~9월)": [
            ("1~4월", df_1_9["datetime"].dt.month <= 4),
            ("5~9월", df_1_9["datetime"].dt.month >= 5)
        ],
        "시나리오 B: 3개 구간 (1~3월 / 4~6월 / 7~9월)": [
            ("1~3월", df_1_9["datetime"].dt.month <= 3),
            ("4~6월", (df_1_9["datetime"].dt.month >= 4) & (df_1_9["datetime"].dt.month <= 6)),
            ("7~9월", df_1_9["datetime"].dt.month >= 7)
        ],
        "시나리오 C: 4개 구간 (1~2월 / 3~4월 / 5~6월 / 7~9월)": [
            ("1~2월", df_1_9["datetime"].dt.month <= 2),
            ("3~4월", (df_1_9["datetime"].dt.month >= 3) & (df_1_9["datetime"].dt.month <= 4)),
            ("5~6월", (df_1_9["datetime"].dt.month >= 5) & (df_1_9["datetime"].dt.month <= 6)),
            ("7~9월", df_1_9["datetime"].dt.month >= 7)
        ],
        "시나리오 D: 1개 구간": [
            ("1~9월", df_1_9["datetime"].dt.month <= 9)
        ]
    }

    all_detailed_results = []
    for scenario_name, periods in scenarios.items():
        print(f"\n{'='*50}\n 실험 시작: {scenario_name}\n{'='*50}")
        for period_name, condition in periods:
            sub_df = df_1_9[condition].copy()
            if len(sub_df) <= WINDOW_SIZE:
                print(f"[{period_name}] 데이터 부족으로 건너뜁니다.")
                continue

            data_values = sub_df[FEATURES + [TARGET]].values
            target_idx = len(FEATURES)

            train_size = int(len(data_values) * 0.8)
            train_data = data_values[:train_size]
            test_data = data_values[train_size - WINDOW_SIZE:]

            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train_data)
            test_scaled = scaler.transform(test_data)

            X_train, y_train = make_sequences(train_scaled, target_idx, WINDOW_SIZE)
            X_test, y_test = make_sequences(test_scaled, target_idx, WINDOW_SIZE)
            if len(X_train) == 0 or len(X_test) == 0:
                continue

            model = train_lstm_model(X_train, y_train, (X_train.shape[1], X_train.shape[2]))
            metrics, _, _ = evaluate_model(model, X_test, y_test, scaler, target_idx, FEATURES)

            all_detailed_results.append({
                "시나리오": scenario_name, "대상 구간": period_name,
                "RMSE": metrics["RMSE"], "MAE": metrics["MAE"],
                "MAPE(%)": metrics["MAPE(%)"], "R2": metrics["R2"]
            })
            print(f"  - [{period_name}] 완료 | RMSE:{metrics['RMSE']:.4f} MAE:{metrics['MAE']:.4f} "
                  f"MAPE:{metrics['MAPE(%)']:.2f}% R2:{metrics['R2']:.4f}")

    detailed_results_df = pd.DataFrame(all_detailed_results)
    print("\n==================== [최종] 시나리오별 구간별 상세 성능 비교 ====================")
    print(detailed_results_df.to_string(index=False))
    return detailed_results_df


if __name__ == "__main__":
    _run_all_scenarios_comparison()

