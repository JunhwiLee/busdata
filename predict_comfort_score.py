import joblib
import pandas as pd

# 1. 저장된 모델 로드
model = joblib.load(r'C:\Users\jhlee\Desktop\Python\BasicLAB\linear_regression_model.pkl')

#45.0, 1, 300.0, 0.65, 240, Line3, Mon

# 2. 사용자 입력값 정의
feature_input = {
    'travel_time_min': float(input()),
    'transfers': int(input()),
    'walk_distance_m': float(input()),
    'congestion_level': float(input()),
    'departure_hour': float(input()),
    'bus_line': f'Line{int(input())}',
    'day_of_week': input()
}

# 3. DataFrame 생성
df_input = pd.DataFrame([feature_input])

# 4. 원-핫 인코딩 (학습 시와 동일하게)
df_encoded = pd.get_dummies(df_input, columns=['bus_line', 'day_of_week'], drop_first=True)

# 5. 모델이 기대하는 피처 컬럼으로 정렬 및 누락 컬럼 채우기
feature_cols = model.feature_names_in_  # 학습된 모델의 feature 이름
df_aligned = df_encoded.reindex(columns=feature_cols, fill_value=0)

# 6. 예측
predicted_score = model.predict(df_aligned)[0]
print(f"예측된 Comfort Score: {predicted_score:.1f}")
