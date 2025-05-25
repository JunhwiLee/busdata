
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# === 1. 경로 설정 ===
CSV_PATH   = Path("data.csv")           # 데이터 파일
OUT_DIR    = Path(r"C:\Users\jhlee\Desktop\visualization")      # 그래프 저장 폴더
OUT_DIR.mkdir(exist_ok=True)

# === 2. 데이터 로드 ===
df = pd.read_csv(CSV_PATH)

# === 3. 산점도(연속·이산 수치형 변수) ===
numeric_x = [
    "travel_time_min",   # 이동 소요 시간
    "walk_distance_m",   # 도보 거리
    "congestion",        # 혼잡도
    "transfers"          # 환승 횟수
]

for col in numeric_x:
    plt.figure(figsize=(6, 4))
    plt.scatter(df[col], df["comfort"], s=6, alpha=0.5)
    plt.xlabel(col)
    plt.ylabel("comfort")
    plt.title(f"{col} vs comfort")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{col}_vs_comfort.png")
    plt.close()

# === 4. 출발 시각(시간대) ===
depart_hour = df["departure_time"].str.slice(0, 2).astype(int)
plt.figure(figsize=(6, 4))
plt.scatter(depart_hour, df["comfort"], s=6, alpha=0.5)
plt.xlabel("departure hour")
plt.ylabel("comfort")
plt.title("departure hour vs comfort")
plt.tight_layout()
plt.savefig(OUT_DIR / "departure_hour_vs_comfort.png")
plt.close()

# === 5. 범주형 변수: 박스플롯 ===
categorical_x = ["bus_line", "weekday"]

for col in categorical_x:
    categories = df[col].unique()
    data = [df.loc[df[col] == cat, "comfort"] for cat in categories]
    plt.figure(figsize=(6, 4))
    plt.boxplot(data, labels=categories, vert=True)
    plt.xlabel(col)
    plt.ylabel("comfort")
    plt.title(f"{col} vs comfort (boxplot)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"{col}_box_vs_comfort.png")
    plt.close()

print(f"모든 그래프가 '{OUT_DIR.resolve()}' 폴더에 저장되었습니다.")
