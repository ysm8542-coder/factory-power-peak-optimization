# -*- coding: utf-8 -*-
"""
LSTM_AIModel_Enhanced.py
========================
오차 원인 분석(Error_Driver_Pipeline) 결과를 반영한 '보강 + 보정' 전력 예측 모델.

구조
----
1단계 보강 LSTM   : 기존 LSTM(LSTM_AIModel과 같은 구조) 입력에 오차 원인 변수를 추가
                    - 시간대/근무일, 평소 패턴(전체·최근), 생산·인원 변화, 고온·누적 고온, 직전 전력 변화
                    - 예측 대상 시점(다음 1시간)의 계획/예보 값(nx_*)
                    - peak 가중 학습: 부하가 높은 시점의 오차에 더 큰 가중치 (peak 과소예측 완화)
2단계 잔차 보정    : 실제 − LSTM 예측(잔차)을 예측해서 더함
                    - 학습 구간 out-of-fold LSTM 예측의 잔차로 학습 (in-sample 잔차 과소평가 방지)
                    - GBM(고부하 시점 가중) 2개의 평균:
                        ① peak 보정형: 보강 LSTM 예측 + 오차 원인 변수 + 최근 peak 수준 + 어제/지난주 같은 시각 + 직전 오차
                        ② 스태킹형  : ① + 일반 LSTM 예측(out-of-fold)
                      ①은 7월 같은 새로운 최고 부하에서 peak 오차가 작고, ②는 저부하·전체 오차가 작아 평균으로 균형
                    - (선택) RESID_MODEL="hybrid" 이면 Ridge + GBM
                    - 보정 강도 alpha는 학습 구간 마지막 20%에서 선택 (테스트 구간 미사용)

사용 예
-------
    from LSTM_AIModel_Enhanced import *
    df = load_data("okm_augumented_2021_processed.xlsx")
    D  = add_driver_features(df)
    res = fit_predict_split(D, tr_end=int(len(D)*0.8), te_end=len(D))   # 8:2
    folds = expanding_window(D)                                          # 확장 윈도우
    print(metrics(res["y"], res["preds"]["보강+보정"]))

가정
----
USE_KNOWN_FUTURE=True 이면 생산량·공장인원·기온의 1시간 앞 값(생산 계획·근무표·기온 예보)과
당일 최고기온 예보를 안다고 본다. 모르면 False → 1시간 전 값으로 대체.
"""
import os, time, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================================
# 설정
# =====================================================================
FEATURES = ['평균', '생산량', '기온', '풍속', '습도', '강수량', '전기요금(계절)', '공장인원', '시간',
            '전력 구간합계', '전력 최대값', '전력 최소값', '전력 변동폭']      # 일반 LSTM 입력 (LSTM_AIModel과 동일)
TARGET = '평균'
WINDOW_SIZE = 24

CFG = dict(
    USE_KNOWN_FUTURE=True,   # 계획/예보 값을 안다고 가정
    HOT_BASE=25.0,           # 냉방도시 기준 온도(℃)
    HOT_TH=28.0,             # 고온 연속시간 기준(℃)
    RECENT_DAYS=5,           # '최근 평소 패턴'에 쓰는 최근 일수 (같은 시각·근무유형)
    N_OOF=3,                 # 잔차용 out-of-fold LSTM 개수
    PEAK_Q=0.85,             # peak 가중 학습: 학습 구간 target 상위 (1-PEAK_Q) 를 peak로 봄
    PEAK_W=2.0,              # peak 시점 가중치 추가분 (0이면 가중 안 함)
    RESID_MODEL="gbm",       # "gbm" | "hybrid"(Ridge + GBM)
    RESID_PEAK_W=2.0,        # 잔차 보정 학습 시 고부하(LSTM 예측 상위 15%) 가중치 추가분
    STACK_BASE=True,         # 일반 LSTM 예측(out-of-fold)도 보정 모델 입력으로 사용
    EPOCHS=50, BATCH=32, PATIENCE=10, VAL_SPLIT=0.2,   # LSTM 학습 설정
    SEED=42,
    VERBOSE=True,
)

MODEL_NAMES = ["일반 LSTM", "보강 LSTM", "보강+보정"]


def set_seed(seed):
    """numpy · TensorFlow 난수 시드 고정."""
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


# =====================================================================
# 데이터
# =====================================================================
def load_data(path="okm_augumented_2021_processed.xlsx"):
    """1~9월 데이터 + 휴일/주말 플래그 + 요일 이름."""
    df = pd.read_excel(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    df = df.interpolate(method="linear").dropna()
    df = df[(df["datetime"].dt.month >= 1) & (df["datetime"].dt.month <= 9)].reset_index(drop=True)
    df["is_holiday"] = (df["day"] == 8).astype(int)
    df["is_weekend"] = df["day"].isin([6, 7]).astype(int)
    df["day_name"] = df["day"].map({1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토", 7: "일", 8: "공휴일"})
    return df


def make_sequences(data, target_idx, window_size=WINDOW_SIZE):
    """LSTM_AIModel과 동일: X = data[i:i+W], y = data[i+W, target]."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size)])
        y.append(data[i + window_size, target_idx])
    return np.array(X), np.array(y)


def _streak(cond):
    """조건이 연속으로 참인 시간 수 (거짓이면 0으로 초기화)."""
    out, c = [], 0
    for v in cond:
        c = c + 1 if v else 0
        out.append(c)
    return np.array(out)


def add_driver_features(df, cfg=CFG):
    """오차 원인 변수 생성. 모든 변수는 과거 값 또는 (USE_KNOWN_FUTURE일 때) 계획/예보 값만 사용."""
    D = df.copy().reset_index(drop=True)
    D["datetime"] = pd.to_datetime(D["datetime"])
    P = TARGET
    dt = D["datetime"]

    # 시간 간격이 정확히 k시간일 때만 유효한 lag / lead
    def lag(s, k=1):
        ok = (dt - dt.shift(k)).dt.total_seconds().div(3600) == k
        return s.shift(k).where(ok)

    def lead(s, k=1):
        ok = (dt.shift(-k) - dt).dt.total_seconds().div(3600) == k
        return s.shift(-k).where(ok)

    # ---- 일정 ----
    D["hour"] = dt.dt.hour
    D["dow"] = dt.dt.dayofweek
    D["sin_h"] = np.sin(2 * np.pi * D["hour"] / 24)
    D["cos_h"] = np.cos(2 * np.pi * D["hour"] / 24)
    D["workday"] = ((D["is_weekend"] == 0) & (D["is_holiday"] == 0)).astype(int)
    D["day_type"] = np.where(D["workday"] == 1, "근무일", "비근무일")
    D["daytime"] = D["hour"].between(8, 19).astype(int)

    # ---- 평소 패턴: 전체 과거 평균 + 최근 N일 평균 (계절이 바뀌면 전체 평균은 늦게 따라오므로 최근 평균 추가) ----
    grp = D.groupby(["day_type", "hour"])[P]
    fallback = D[P].shift(1).rolling(24, min_periods=1).mean().fillna(D[P].iloc[0])
    D["usual_load"] = grp.transform(lambda s: s.shift(1).expanding(3).mean()).fillna(fallback)
    D["recent_usual"] = grp.transform(lambda s: s.shift(1).rolling(cfg["RECENT_DAYS"], min_periods=2).mean()).fillna(D["usual_load"])
    D["recent_peak_h"] = grp.transform(lambda s: s.shift(1).rolling(cfg["RECENT_DAYS"], min_periods=2).max()).fillna(D["recent_usual"])
    D["usual_delta"] = (D["usual_load"] - lag(D["usual_load"])).fillna(0)
    D["recent_delta"] = (D["recent_usual"] - lag(D["recent_usual"])).fillna(0)

    # ---- 생산·인원 ----
    D["d_prod_log"] = (np.log1p(D["생산량"]) - np.log1p(lag(D["생산량"]))).fillna(0)
    D["d_staff_log"] = (np.log1p(D["공장인원"]) - np.log1p(lag(D["공장인원"]))).fillna(0)
    D["run_hours"] = _streak(D["생산량"] > 0)
    D["prod_x_staff"] = np.log1p(D["생산량"]) * np.log1p(D["공장인원"])

    # ---- 고온·누적 고온 ----
    heat = (D["기온"] - cfg["HOT_BASE"]).clip(lower=0)
    D["heat"] = heat
    D["temp_r24"] = D["기온"].rolling(24, min_periods=1).mean()
    D["CDH24"] = heat.rolling(24, min_periods=1).sum()
    D["CDH72"] = heat.rolling(72, min_periods=1).sum()
    D["hot_streak"] = _streak(D["기온"] >= cfg["HOT_TH"])
    D["prod_x_heat"] = np.log1p(D["생산량"]) * heat
    D["heat_work_day"] = heat * D["workday"] * D["daytime"]          # 근무일 주간 냉방 부하
    D["tmax_day"] = D.groupby(dt.dt.date)["기온"].transform("max")   # 당일 최고기온 (예보값으로 가정)

    # ---- 직전 전력 ----
    D["p_diff1"] = (D[P] - lag(D[P])).fillna(0)

    # ---- 다음 시간(예측 대상 시점)의 값을 현재 행에 붙임 (nx_*) ----
    cal_next = ["sin_h", "cos_h", "workday", "usual_load", "usual_delta", "recent_usual", "recent_delta", "recent_peak_h"]
    exog_next = ["생산량", "공장인원", "기온", "d_prod_log", "d_staff_log", "prod_x_staff", "CDH24",
                 "prod_x_heat", "heat_work_day", "tmax_day"]
    next_cols = cal_next + (exog_next if cfg["USE_KNOWN_FUTURE"] else [])
    for c in next_cols:
        D[f"nx_{c}"] = lead(D[c]).ffill()

    # ---- 보강 LSTM 입력 = 일반 LSTM 입력 + 요일/주말 + 오차 원인 변수 ----
    driver = ["sin_h", "cos_h", "workday", "usual_load", "usual_delta", "recent_usual", "d_prod_log", "d_staff_log",
              "run_hours", "prod_x_staff", "temp_r24", "CDH24", "CDH72", "hot_streak", "prod_x_heat", "p_diff1"] \
             + [f"nx_{c}" for c in next_cols]
    aug = list(dict.fromkeys(FEATURES + ["day", "is_weekend"] + driver))
    D[aug] = D[aug].ffill().fillna(0)

    # ---- 잔차 보정 모델의 정적 입력 (예측 대상 시점 t 기준) ----
    exog_t = ["생산량", "공장인원", "기온", "d_prod_log", "d_staff_log", "run_hours", "prod_x_staff",
              "temp_r24", "CDH24", "CDH72", "hot_streak", "prod_x_heat", "heat_work_day", "tmax_day"]
    R0 = D[["hour", "sin_h", "cos_h", "dow", "workday", "daytime", "usual_load", "usual_delta",
            "recent_usual", "recent_delta", "recent_peak_h"]].copy()
    for c in exog_t:
        R0[c] = D[c] if cfg["USE_KNOWN_FUTURE"] else lag(D[c])
    R0["p_lag1"] = lag(D[P])
    R0["p_lag24"] = lag(D[P], 24)                                   # 어제 같은 시각
    R0["p_lag168"] = lag(D[P], 168)                                 # 지난주 같은 시각
    R0["p_max24"] = lag(D[P]).rolling(24, min_periods=1).max()      # 최근 24h 최대
    R0["p_trend"] = lag(D[P]) - lag(D[P], 2)
    R0["dev_prev"] = lag(D[P]) - lag(D["recent_usual"])

    D.attrs.update(base_feats=list(FEATURES), aug_feats=aug, R0=R0, cfg=dict(cfg))
    return D


# =====================================================================
# LSTM (LSTM_AIModel과 같은 구조 + 선택적 sample_weight)
# =====================================================================
def train_lstm_model(X_train, y_train, input_shape, sample_weight=None, cfg=CFG):
    """LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(16) → Dense(1). sample_weight로 peak 가중 학습."""
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    es = EarlyStopping(monitor='val_loss', patience=cfg["PATIENCE"], restore_best_weights=True)
    model.fit(X_train, y_train, sample_weight=sample_weight, validation_split=cfg["VAL_SPLIT"],
              epochs=cfg["EPOCHS"], batch_size=cfg["BATCH"], callbacks=[es], verbose=0)
    return model


def _predict(m, X):
    return np.asarray(m.predict(X, verbose=0)).ravel()


def _inv(sc, y, t):
    """MinMaxScaler 역변환 (t번째 열만)."""
    return np.asarray(y).ravel() * sc.data_range_[t] + sc.data_min_[t]


def _peak_weight(y_scaled, cfg):
    """target 상위 (1 - PEAK_Q) 시점에 가중치 1 + PEAK_W, 나머지 1."""
    if not cfg["PEAK_W"]:
        return None
    th = np.quantile(y_scaled, cfg["PEAK_Q"])
    return 1.0 + cfg["PEAK_W"] * (y_scaled >= th)


def lstm_fit_predict(D, feats, tr_end, te_end, n_oof=0, weighted=False, cfg=CFG, seed=None):
    """rows[:tr_end] 학습 → rows[tr_end:te_end] 예측(kW). n_oof>0이면 rows[W:tr_end]의 out-of-fold 예측도."""
    seed = cfg["SEED"] if seed is None else seed
    W = WINDOW_SIZE
    vals = D[feats + [TARGET]].values.astype(float)
    t = len(feats)
    sc = MinMaxScaler().fit(vals[:tr_end])
    a = sc.transform(vals)
    Xtr, ytr = make_sequences(a[:tr_end], t, W)
    Xte, yte = make_sequences(a[tr_end - W:te_end], t, W)
    sw = _peak_weight(ytr, cfg) if weighted else None
    set_seed(seed)
    m = train_lstm_model(Xtr, ytr, Xtr.shape[1:], sw, cfg)
    out = {"pred_te": _inv(sc, _predict(m, Xte), t), "y_te": _inv(sc, yte, t), "model": m, "scaler": sc}
    if n_oof:
        idx = np.arange(len(ytr))
        oof = np.full(len(ytr), np.nan)
        for k, blk in enumerate(np.array_split(idx, n_oof)):
            keep = (idx < blk[0] - W) | (idx > blk[-1] + W)      # 예측 블록 ± W 시간은 학습에서 제외
            set_seed(seed + k + 1)
            mk = train_lstm_model(Xtr[keep], ytr[keep], Xtr.shape[1:], None if sw is None else sw[keep], cfg)
            oof[blk] = _predict(mk, Xtr[blk])
        out["pred_oof"] = _inv(sc, oof, t)
    return out


# =====================================================================
# 2단계: 잔차 보정
# =====================================================================
# hybrid 모드에서 Ridge(선형)에 쓰는 변수
LINEAR_FEATS = ["lstm_pred", "heat_work_day", "tmax_day", "CDH24", "pred_vs_recent", "pred_vs_peak_h",
                "prod_x_staff", "workday", "daytime", "lag_resid", "pred_jump"]


def residual_inputs(D, lstm_pred, base_pred=None):
    """잔차 보정 모델 입력 R과 잔차(실제 − LSTM 예측)를 만든다. base_pred가 있으면 스태킹 변수 추가."""
    P = TARGET
    dt = D["datetime"]
    ok1 = (dt - dt.shift(1)).dt.total_seconds().div(3600) == 1
    R = D.attrs["R0"].copy()
    resid = D[P] - lstm_pred
    R["lag_resid"] = resid.shift(1).where(ok1)
    R["lag_resid3"] = R["lag_resid"].rolling(3, min_periods=1).mean()
    R["lag_resid24"] = resid.shift(24)                               # 어제 같은 시각의 오차
    R["lstm_pred"] = lstm_pred
    R["pred_jump"] = lstm_pred - R["p_lag1"]
    R["pred_vs_usual"] = lstm_pred - D["usual_load"]
    R["pred_vs_recent"] = lstm_pred - D["recent_usual"]
    R["pred_vs_peak_h"] = lstm_pred - D["recent_peak_h"]
    if base_pred is not None:                                        # 일반 LSTM 예측 (스태킹)
        R["base_pred"] = base_pred
        R["base_vs_aug"] = base_pred - lstm_pred
    return R, resid


class ResidualCorrector:
    """Ridge(외삽 가능한 선형 추세) + GBM(비선형 상호작용)."""

    def __init__(self, mode="hybrid", seed=42, peak_w=0.0):
        self.mode, self.seed, self.peak_w = mode, seed, peak_w

    def fit(self, X, y):
        sw = None
        if self.peak_w:
            lp = X["lstm_pred"].values
            sw = 1.0 + self.peak_w * (lp >= np.nanquantile(lp, 0.85))
        X = X.copy()
        self.cols = list(X.columns)
        self.lin_cols = [c for c in LINEAR_FEATS if c in X.columns]
        self.fill = X.median()
        base = np.zeros(len(y))
        if self.mode == "hybrid":
            self.ss = StandardScaler().fit(X[self.lin_cols].fillna(self.fill))
            self.lin = Ridge(alpha=10.0).fit(self.ss.transform(X[self.lin_cols].fillna(self.fill)), y, sample_weight=sw)
            base = self.lin.predict(self.ss.transform(X[self.lin_cols].fillna(self.fill)))
        self.gbm = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.04, max_iter=400, min_samples_leaf=30,
                                                 l2_regularization=1.0, random_state=self.seed)
        self.gbm.fit(X[self.cols], y - base, sample_weight=sw)
        return self

    def predict(self, X):
        out = self.gbm.predict(X[self.cols])
        if self.mode == "hybrid":
            out = out + self.lin.predict(self.ss.transform(X[self.lin_cols].fillna(self.fill)))
        return out


def _fit_corrector(R, resid, tr_i, te_i, cfg):
    """잔차 보정 모델 1개 학습. alpha는 학습 구간 마지막 20%에서 선택 (테스트 구간 미사용)."""
    feats = list(R.columns)
    lo, hi = resid.loc[tr_i].quantile([0.005, 0.995])
    cut = int(len(tr_i) * 0.8)
    fi, vi = tr_i[:cut], tr_i[cut:]
    mk = lambda: ResidualCorrector(cfg["RESID_MODEL"], cfg["SEED"], cfg["RESID_PEAK_W"])
    cv = mk().fit(R.loc[fi, feats], resid.loc[fi].clip(lo, hi)).predict(R.loc[vi, feats])
    alphas = np.round(np.arange(0, 1.01, 0.1), 2)
    alpha = float(alphas[np.argmin([np.mean(np.abs(resid.loc[vi].values - a * cv)) for a in alphas])])
    model = mk().fit(R.loc[tr_i, feats], resid.loc[tr_i].clip(lo, hi))
    corr = pd.Series(np.nan, index=R.index)
    corr.loc[tr_i] = alpha * model.predict(R.loc[tr_i, feats])
    corr.loc[te_i] = alpha * model.predict(R.loc[te_i, feats])
    return corr, alpha, model


def fit_predict_split(D, tr_end, te_end, name="split", with_baseline=True, cfg=CFG, aug_cache=None, base_cache=None):
    """
    한 번의 학습/테스트: rows[:tr_end] 학습 → rows[tr_end:te_end] 예측.
    반환 dict: y, preds{일반 LSTM, 보강 LSTM, 보강+보정}, rows, datetime, alpha, full_pred 등.

    보정 = 두 보정 모델의 평균 (앙상블)
      ① peak 보정형 : 보강 LSTM 예측 + 오차 원인 변수 → peak(일 최대전력) 오차가 가장 작음
      ② 스태킹형    : ① + 일반 LSTM 예측            → 저부하·전체 MAE/MAPE가 가장 작음
    STACK_BASE=False면 ①만 사용.
    """
    t0 = time.time()
    W = WINDOW_SIZE
    stack = cfg.get("STACK_BASE", False)
    base = base_cache if base_cache is not None else (
        lstm_fit_predict(D, D.attrs["base_feats"], tr_end, te_end, n_oof=cfg["N_OOF"] if stack else 0, cfg=cfg)
        if (with_baseline or stack) else None)
    aug = aug_cache if aug_cache is not None else lstm_fit_predict(
        D, D.attrs["aug_feats"], tr_end, te_end, n_oof=cfg["N_OOF"], weighted=cfg["PEAK_W"] > 0, cfg=cfg)

    lstm_pred = pd.Series(np.nan, index=D.index)
    lstm_pred.iloc[W:tr_end] = aug["pred_oof"]
    lstm_pred.iloc[tr_end:te_end] = aug["pred_te"]
    te_i = D.index[tr_end:te_end]

    R1, resid = residual_inputs(D, lstm_pred)
    tr_i = D.index[W + 2:tr_end]
    tr_i = tr_i[R1.loc[tr_i, "lstm_pred"].notna().values]
    corr1, a1, m1 = _fit_corrector(R1, resid, tr_i, te_i, cfg)
    corrs, alphas, models = [corr1], [a1], [m1]
    if stack:
        base_pred = pd.Series(np.nan, index=D.index)
        base_pred.iloc[W:tr_end] = base["pred_oof"]
        base_pred.iloc[tr_end:te_end] = base["pred_te"]
        R2, _ = residual_inputs(D, lstm_pred, base_pred)
        corr2, a2, m2 = _fit_corrector(R2, resid, tr_i, te_i, cfg)
        corrs.append(corr2); alphas.append(a2); models.append(m2)
    corr = sum(corrs) / len(corrs)
    final = np.clip(aug["pred_te"] + corr.loc[te_i].values, 0, None)

    preds = {"보강 LSTM": aug["pred_te"], "보강+보정": final}
    if base is not None and with_baseline:
        assert np.allclose(base["y_te"], aug["y_te"], atol=1e-6)
        preds = {"일반 LSTM": base["pred_te"], **preds}
    res = dict(name=name, rows=te_i, datetime=D.loc[te_i, "datetime"].values, y=aug["y_te"], preds=preds,
               alpha=alphas, tr_end=tr_end, te_end=te_end, correctors=models, R=R1, resid=resid,
               full_pred=(lstm_pred + corr).clip(lower=0), aug=aug, base=base)
    if cfg["VERBOSE"]:
        print(f"[{name}] 학습 {tr_end:,} / 테스트 {len(te_i):,} | alpha={alphas} | " +
              " | ".join(f"{k} MAE {mean_absolute_error(res['y'], v):.2f} R2 {r2_score(res['y'], v):.4f}" for k, v in preds.items())
              + f" ({time.time() - t0:.0f}s)", flush=True)
    return res


# =====================================================================
# 검증 방식
# =====================================================================
def split_82(D, cfg=CFG, **kw):
    """앞 80% 학습 → 뒤 20% 테스트."""
    tr = int(len(D) * 0.8)
    return fit_predict_split(D, tr, len(D), name="8:2", cfg=cfg, **kw)


def expanding_window(D, initial_ratio=0.5, n_folds=4, cfg=CFG, **kw):
    """LSTM_AIModel의 시나리오 D 확장 윈도우와 동일: 초기 50% 학습, 나머지를 4등분해 순서대로 테스트."""
    total = len(D)
    init = int(total * initial_ratio)
    step = (total - init) // n_folds
    out, tr_end = [], init
    for k in range(1, n_folds + 1):
        te_end = min(total, tr_end + step) if k < n_folds else total
        out.append(fit_predict_split(D, tr_end, te_end, name=f"Fold {k}", cfg=cfg, **kw))
        tr_end += step
    return out


# =====================================================================
# 평가 지표
# =====================================================================
def metrics(y, p, tol_pct=10, min_kw=8):
    """MSE · RMSE · MAE · MAPE · R2 + 허용(±tol_pct%) 비율 · 과소/과대 비율. 절대오차 < min_kw 는 허용으로 봄."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    pct = np.where(y > 0, e / np.where(y > 0, y, 1) * 100, np.nan)
    bad = (np.abs(pct) > tol_pct) & (np.abs(e) >= min_kw)
    nz = y != 0
    return {"MSE": mean_squared_error(y, p), "RMSE": np.sqrt(mean_squared_error(y, p)), "MAE": mean_absolute_error(y, p),
            "MAPE(%)": np.mean(np.abs(e[nz] / y[nz])) * 100, "R2": r2_score(y, p),
            "평균오차(+과소)": e.mean(), "허용±10%(%)": 100 - bad.mean() * 100,
            "과소(%)": (bad & (pct > 0)).mean() * 100, "과대(%)": (bad & (pct < 0)).mean() * 100}


def daily_peak_table(datetimes, y, preds):
    """일별 최대전력(peak): 실제 peak vs 예측 peak, 오차(+ = 과소예측)."""
    df = pd.DataFrame({"datetime": pd.to_datetime(datetimes), "y": y, **preds})
    df["date"] = df["datetime"].dt.date
    g = df.groupby("date")
    out = pd.DataFrame({"실제 peak": g["y"].max()})
    for k in preds:
        out[f"{k} peak"] = g[k].max()
        out[f"{k} peak오차"] = out["실제 peak"] - out[f"{k} peak"]
    return out


def peak_metrics(datetimes, y, preds, workday_only=True, min_peak=100):
    """peak 지표: 근무일 일 최대전력 오차(MAE·평균), 실제 부하 상위 10% 시간의 MAE·평균오차."""
    t = daily_peak_table(datetimes, y, preds)
    t = t[t["실제 peak"] >= min_peak] if workday_only else t
    y = np.asarray(y)
    top = y >= np.quantile(y, 0.9)
    rows = {}
    for k, p in preds.items():
        p = np.asarray(p)
        rows[k] = {"일 peak MAE": t[f"{k} peak오차"].abs().mean(), "일 peak 평균오차(+과소)": t[f"{k} peak오차"].mean(),
                   "상위10% 시간 MAE": np.mean(np.abs(y[top] - p[top])), "상위10% 평균오차(+과소)": np.mean(y[top] - p[top])}
    return pd.DataFrame(rows).T


def combine(results):
    """여러 fold 결과를 이어 붙임."""
    keys = list(results[0]["preds"])
    return dict(rows=np.concatenate([r["rows"] for r in results]),
                datetime=np.concatenate([r["datetime"] for r in results]),
                y=np.concatenate([r["y"] for r in results]),
                preds={k: np.concatenate([r["preds"][k] for r in results]) for k in keys})


if __name__ == "__main__":
    df = load_data()
    D = add_driver_features(df)
    r82 = split_82(D)
    print(pd.DataFrame({k: metrics(r82["y"], v) for k, v in r82["preds"].items()}).T.round(3))
    ex = expanding_window(D)
    c = combine(ex)
    print(pd.DataFrame({k: metrics(c["y"], v) for k, v in c["preds"].items()}).T.round(3))
