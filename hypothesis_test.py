import pandas as pd
from pathlib import Path
from scipy import stats
import sys

# 설정
input_dir = Path("BasicLAB/bus_data_transfer")
pattern = "bus_congestion_2025*.csv"

# CSV 파일 수집
files = sorted(input_dir.glob(pattern))
if not files:
    print(f"[Error] '{input_dir}' 경로에 '{pattern}' 형식의 파일이 없습니다.")
    sys.exit(1)

# 데이터 로드
df_list = []
for f in files:
    try:
        df_list.append(pd.read_csv(f, parse_dates=['날짜']))
    except Exception as e:
        print(f"[Error] 파일 읽기 실패: {f} -> {e}")
        sys.exit(1)

# 데이터 합치기
df = pd.concat(df_list, ignore_index=True)

# 요일별 혼잡도 그룹화 및 기술통계 출력
group_stats = df.groupby('요일')['혼잡도'].agg(['count', 'mean', 'std', 'min', 'max'])
print("\n=== 요일별 혼잡도 기술통계 ===")
print(group_stats.to_string(), "\n")

# One-way ANOVA 검정
groups = [grp['혼잡도'].values for _, grp in df.groupby('요일')]
f_stat, p_value = stats.f_oneway(*groups)
print("=== One-way ANOVA 결과 ===")
print(f"F-statistic = {f_stat:.3f}")
print(f"p-value     = {p_value}\n")

# Kruskal-Wallis 검정
h_stat, kr_p = stats.kruskal(*groups)
print("=== Kruskal-Wallis 검정 결과 ===")
print(f"H-statistic = {h_stat:.3f}")
print(f"p-value     = {kr_p}\n")

# 결과 해석 안내
alpha = 0.05
if p_value < alpha:
    print(f"ANOVA에서 p < {alpha}: 요일에 따라 혼잡도 차이가 통계적으로 유의합니다.")
else:
    print(f"ANOVA에서 p >= {alpha}: 요일에 따른 혼잡도 차이가 통계적으로 유의하지 않습니다.")

if kr_p < alpha:
    print(f"Kruskal-Wallis에서 p < {alpha}: 요일에 따라 혼잡도 분포가 유의미하게 다릅니다.")
else:
    print(f"Kruskal-Wallis에서 p >= {alpha}: 요일에 따른 분포 차이가 유의하지 않습니다.")
