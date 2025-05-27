import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
input_dir = Path("BasicLAB/bus_data_transfer")
files = sorted(input_dir.glob("bus_congestion_2025*.csv"))
df_list = [pd.read_csv(f) for f in files]
df = pd.concat(df_list, ignore_index=True)

# 요일·시간별 평균 계산
df_group = df.groupby(['요일', '시간'])['혼잡도'].mean().reset_index()

# 피벗 테이블 생성 및 정렬
weekday_order = ['월', '화', '수', '목', '금', '토', '일']
time_order = sorted(df_group['시간'].unique(), key=lambda t: int(t.replace('시', '')))
pivot = df_group.pivot(index='시간', columns='요일', values='혼잡도')
pivot = pivot.reindex(index=time_order, columns=weekday_order)

# 선그래프 그리기
plt.figure(figsize=(10, 6))
for day in weekday_order:
    plt.plot(pivot.index, pivot[day], label=day)
plt.xlabel("시간대")
plt.ylabel("평균 혼잡도")
plt.title("요일별 평균 혼잡도 추이")
plt.legend(title="요일")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
input_dir = Path("BasicLAB/bus_data_transfer")
files = sorted(input_dir.glob("bus_congestion_2025*.csv"))
if not files:
    raise FileNotFoundError(f"No files found in {input_dir} matching pattern")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# 1) Boxplot: 요일별 분포 시각화
weekday_order = ['월', '화', '수', '목', '금', '토', '일']
data_by_day = [df[df['요일'] == day]['혼잡도'].dropna().values for day in weekday_order]
plt.figure()
plt.boxplot(data_by_day, labels=weekday_order)
plt.xlabel("요일")
plt.ylabel("혼잡도")
plt.title("요일별 혼잡도 분포")
plt.tight_layout()
plt.show()

# 2) Bar chart: 요일별 평균 ± 표준오차 시각화
grouped = df.groupby('요일')['혼잡도']
means = grouped.mean().reindex(weekday_order)
sems = grouped.sem().reindex(weekday_order)
x = np.arange(len(weekday_order))
plt.figure()
plt.bar(x, means, yerr=sems, capsize=5)
plt.xticks(x, weekday_order)
plt.xlabel("요일")
plt.ylabel("평균 혼잡도")
plt.title("요일별 평균 혼잡도 (±표준오차)")
plt.tight_layout()
plt.show()
