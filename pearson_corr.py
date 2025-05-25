import pandas as pd

# 1) 데이터 읽기 -------------------------------------------------
df = pd.read_csv("data.csv")

# 2) 출발 시각 문자열 → 시간(정수)로 변환 ------------------------
df["departure_hour"] = df["departure_time"].str.slice(0, 2).astype(int)

# 3) 연속 변수 목록 정의 ----------------------------------------
continuous_cols = [
    "travel_time_min",   # 이동 소요 시간
    "walk_distance_m",   # 도보 거리
    "congestion",        # 혼잡도
    "departure_hour",
    "transfers",   # 출발 시각(시)
    "comfort"            # 쾌적도
]

# 4) Pearson 상관계수 계산 -------------------------------------
corr_matrix = df[continuous_cols].corr(method="pearson").round(3)

print("=== Pearson Correlation Matrix ===")
print(corr_matrix)

# 5) CSV 저장(선택) --------------------------------------------
corr_matrix.to_csv("pearson_corr.csv", index=True)
print("\n상관계수 행렬을 pearson_corr.csv 로 저장했습니다.")
