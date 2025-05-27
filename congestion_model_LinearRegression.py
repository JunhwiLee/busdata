import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# 1) 데이터 로드
input_dir = Path("BasicLAB/bus_data_transfer")
files = sorted(input_dir.glob("bus_congestion_2025*.csv"))
if not files:
    raise FileNotFoundError(f"No files found in {input_dir} matching pattern")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# 2) 피처 엔지니어링
df['날짜'] = pd.to_datetime(df['날짜'], format='%Y-%m-%d')
df['day'] = df['날짜'].dt.day  # 수치형 변수

# 3) 입력 변수 및 타깃 (기종점과 정류장명 제외)
X = df[['노선', '요일', '시간', 'day']]
y = df['혼잡도']

# 4) One-Hot Encoding & sparse matrix 생성
column_transformer = ColumnTransformer([
    (
        'ohe',
        OneHotEncoder(drop='first',           # 더미 변수 트랩 방지
                     sparse_output=True,
                     handle_unknown='ignore'),
        ['노선', '요일', '시간']
    )
], remainder='passthrough')              # 'day' 컬럼은 그대로 전달

X_sparse = column_transformer.fit_transform(X)

# 5) 학습/테스트 분리
X_train_sparse, X_test_sparse, y_train, y_test = train_test_split(
    X_sparse, y, test_size=0.2, random_state=42
)

# 6) LinearRegression은 sparse 입력을 densify 하므로 toarray()로 변환
X_train = X_train_sparse.toarray()
X_test  = X_test_sparse.toarray()

# 7) 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 8) 성능 평가
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
print(f"LinearRegression RMSE : {rmse:.3f}")
print(f"LinearRegression R²   : {r2:.3f}")

# 9) 모델 + 전처리 파이프라인 저장
with open("congestion_model_linear_no_stop.pkl", "wb") as f:
    pickle.dump({'transformer': column_transformer, 'model': model}, f)
print("Saved linear regression model to congestion_model_linear.pkl")
