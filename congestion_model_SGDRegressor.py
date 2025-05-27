import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import numpy as np

# 1) 데이터 로드
input_dir = Path("BasicLAB/bus_data_transfer")
files = sorted(input_dir.glob("bus_congestion_2025*.csv"))
if not files:
    raise FileNotFoundError(f"No files found in {input_dir} matching pattern")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# 2) 피처 엔지니어링
df['날짜'] = pd.to_datetime(df['날짜'], format='%Y-%m-%d')
df['day'] = df['날짜'].dt.day  # 수치형 변수

# 입력 변수 및 타깃
X = df[['노선', '기종점', '정류장명', '요일', '시간', 'day']]
y = df['혼잡도']

# 3) sparse 행렬
column_transformer = ColumnTransformer([
    ('ohe', OneHotEncoder(sparse_output=True, handle_unknown='ignore'), ['노선', '기종점', '정류장명', '요일', '시간'])
], remainder='passthrough')  # day 컬럼은 그대로 전달

X_sparse = column_transformer.fit_transform(X)

# 4) 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(
    X_sparse, y, test_size=0.2, random_state=42
)

# 5) 희소 입력을 지원하는 선형 회귀 모델 사용 (SGD)
model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
model.fit(X_train, y_train)

# 6) 성능 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
print(f"SGDR RMSE : {rmse:.3f}")
print(f"SGDR R²   : {r2:.3f}")

# 7) 모델 + 전처리 파이프라인 저장
with open("congestion_model_sparse.pkl", "wb") as f:
    pickle.dump({'transformer': column_transformer, 'model': model}, f)
print("Saved sparse model to congestion_model_sparse.pkl")
