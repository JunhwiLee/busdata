"""
1) comfort_dataset.csv(또는 comfort_dataset_clean.csv) 로드
2) comfort IQR 기반 이상치 제거
3) 8:2 Train / Test 분할
4) 연속형 표준화(Train 통계 사용) + 원-핫 인코딩
5) Gradient Descent 선형회귀 학습
6) Train·Test RMSE & R² 출력
7) 결과 파일 저장(model .pkl, 손실 곡선, 계수 CSV)

필수 패키지: pandas numpy matplotlib
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.utils import shuffle   # sklearn 설치돼 있지 않으면 numpy로 바꿔도 OK

# ============ 0. 기본 설정 ============ #
RAW_CSV   = Path("data.csv")
CLEAN_CSV = Path("comfort_dataset_clean.csv")   # 없으면 RAW 사용
OUT_DIR   = Path(r"model_outputs_split").resolve()
OUT_DIR.mkdir(exist_ok=True)

TEST_RATIO = 0.2
LR         = 0.01
N_ITERS    = 2000
PRINT_EVERY = 100
RANDOM_STATE = 42

# ============ 1. 데이터 로드 & IQR 클린 ============ #
df = pd.read_csv(CLEAN_CSV if CLEAN_CSV.exists() else RAW_CSV)

# (필요하면 IQR 클린 실행 ─ 이미 clean 파일 있으면 스킵)
if not CLEAN_CSV.exists():
    q1, q3 = df["comfort"].quantile([0.25, 0.75])
    iqr    = q3 - q1
    low, high = q1 - 1.5*iqr, q3 + 1.5*iqr
    df = df[(df["comfort"] >= low) & (df["comfort"] <= high)]
    df.to_csv(CLEAN_CSV, index=False)

# 파생 열
df["departure_hour"] = df["departure_time"].str.slice(0, 2).astype(int)

cont_cols  = ["travel_time_min", "walk_distance_m",
              "congestion", "transfers", "departure_hour"]
cat_cols   = ["bus_line", "weekday"]

# ============ 2. Train / Test 8:2 ============ #
df = shuffle(df, random_state=RANDOM_STATE).reset_index(drop=True)
n_total = len(df)
n_test  = int(TEST_RATIO * n_total)

df_test  = df.iloc[:n_test].reset_index(drop=True)
df_train = df.iloc[n_test:].reset_index(drop=True)

print(f"Train size: {len(df_train)}\nTest  size: {len(df_test)}")

# ============ 3. 인코딩 & 표준화 ============ #
def prepare(df_source, mean=None, std=None, fit_stats=False):
    X_cont = df_source[cont_cols].copy()
    X_cat  = pd.get_dummies(df_source[cat_cols], drop_first=True).astype(float)

    if fit_stats:  # Train
        mean = X_cont.mean();  std = X_cont.std()
    X_cont = (X_cont - mean) / std

    X_df = pd.concat([X_cont, X_cat], axis=1).astype(float)
    return X_df, df_source["comfort"].values.reshape(-1, 1), mean, std

X_train_df, y_train, cont_mean, cont_std = prepare(df_train, fit_stats=True)
# 테스트는 train 통계 사용
X_test_df , y_test , _, _               = prepare(df_test , cont_mean, cont_std)

all_features = X_train_df.columns  # 순서 고정

# 테스트 세트에 없는 더미 열을 0으로 추가
for col in all_features:
    if col not in X_test_df:
        X_test_df[col] = 0.0
X_test_df = X_test_df[all_features]

# ============ 4. NumPy 행렬 ============ #
def to_matrix(X_df):
    X_mat = X_df.values.astype(float)
    X_b   = np.hstack([np.ones((len(X_mat), 1)), X_mat])
    return X_b

Xb_train = to_matrix(X_train_df)
Xb_test  = to_matrix(X_test_df)

# ============ 5. Gradient Descent ============ #
n_samples, n_features = Xb_train.shape
theta = np.zeros((n_features, 1))
loss_hist = []

for i in range(N_ITERS):
    preds = Xb_train @ theta
    errors = preds - y_train
    gradients = (2 / n_samples) * Xb_train.T @ errors
    theta -= LR * gradients

    if i % PRINT_EVERY == 0 or i == N_ITERS - 1:
        mse = float((errors**2).mean())
        loss_hist.append((i, mse))
        print(f"iter {i:4d} | Train MSE {mse:.4f}")

# ============ 6. 평가 ============ #
def rmse_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    rmse   = np.sqrt(ss_res / len(y_true))
    r2     = 1 - ss_res / ss_tot
    return rmse, r2

train_rmse, train_r2 = rmse_r2(y_train, Xb_train @ theta)
test_rmse , test_r2  = rmse_r2(y_test , Xb_test  @ theta)

print("\n===== Performance =====")
print(f"Train RMSE: {train_rmse:.4f} | R²: {train_r2:.4f}")
print(f"Test  RMSE: {test_rmse :.4f} | R²: {test_r2 :.4f}")

# ============ 7. 파일 저장 ============ #
# 7-1 손실 곡선
pd.DataFrame(loss_hist, columns=["iter", "mse"]).to_csv(
    OUT_DIR / "loss_history.csv", index=False)

plt.figure(figsize=(6,4))
plt.plot(*zip(*loss_hist))
plt.xlabel("iteration"); plt.ylabel("MSE"); plt.title("GD Loss (Train)")
plt.tight_layout(); plt.savefig(OUT_DIR / "loss_curve.png"); plt.close()

# 7-2 계수 CSV
coef_df = pd.DataFrame({
    "feature": ["bias"] + list(all_features),
    "theta"  : theta.ravel()
})
coef_df.to_csv(OUT_DIR / "coefficients.csv", index=False)

# 7-3 모델 PKL (파라미터 + 전처리 통계 + feature 순서)
model = {
    "theta"        : theta,
    "feature_names": ["bias"] + list(all_features),
    "cont_mean"    : cont_mean,
    "cont_std"     : cont_std
}
with open(OUT_DIR / "linear_reg_split_model.pkl", "wb") as f:
    pickle.dump(model, f)

print(f"\n모든 결과가 '{OUT_DIR}'에 저장되었습니다.")
