import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import random

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
input_dir = Path("BasicLAB/bus_data_transfer")
files = sorted(input_dir.glob("bus_congestion_2025*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# 무작위 3개 노선 선택
unique_routes = df['노선'].unique().tolist()
selected_routes = random.sample(unique_routes, 3)

# 선택된 노선 데이터 필터링 및 평균 혼잡도 계산
df_sel = df[df['노선'].isin(selected_routes)]
grouped = df_sel.groupby(['노선', '시간'])['혼잡도'].mean().reset_index()

# 피벗 테이블 생성 및 시간 순서 정렬
pivot = grouped.pivot(index='시간', columns='노선', values='혼잡도')
pivot = pivot.reindex(sorted(pivot.index, key=lambda t: int(t.replace('시',''))))

# 시각화
plt.figure(figsize=(10, 6))
for route in selected_routes:
    plt.plot(pivot.index, pivot[route], label=f"노선 {route}")
plt.xlabel("시간대")
plt.ylabel("평균 혼잡도")
plt.title("시간대별 평균 혼잡도")
plt.legend(title="노선")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
