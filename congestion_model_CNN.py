import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Input, Reshape
from tensorflow.keras.callbacks import EarlyStopping

# 1) 데이터 로드 및 피벗
input_dir = Path("BasicLAB/bus_data_transfer")
files = sorted(input_dir.glob("bus_congestion_2025*.csv"))
if not files:
    raise FileNotFoundError(f"No files found in {input_dir}")

df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
# 날짜+정류장 단위로 시계열 생성
pivot = df.pivot_table(index=['기종점','정류장명','날짜'], columns='시간', values='혼잡도')
# 시간 순서 정렬
pivot = pivot.reindex(columns=sorted(pivot.columns, key=lambda t: int(t.replace('시',''))))

data = pivot.values.astype(float)  # shape: (samples, 24)

# 2) 시퀀스 생성 (window_size->next value)
window_size = 6
X, y = [], []
for series in data:
    for i in range(series.shape[0] - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
X = np.array(X)  # (n_samples, window_size)
y = np.array(y)  # (n_samples,)

# 3) 데이터 분리 및 모양 변환
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# CNN 입력은 (timesteps, features)
X_train = X_train.reshape(-1, window_size, 1)
X_test = X_test.reshape(-1, window_size, 1)

# 4) CNN 모델 정의
model = Sequential([
    Input(shape=(window_size,1)),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 5) 모델 학습
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=128,
    validation_split=0.2,
    callbacks=[es]
)

# 6) 성능 평가
y_pred = model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"CNN Test RMSE : {rmse:.3f}")
print(f"CNN Test R²   : {r2:.3f}")

# 7) 모델 저장
model.save("congestion_cnn_model.h5")
print("Saved CNN model to congestion_cnn_model.h5")