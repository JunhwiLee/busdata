import pandas as pd
import matplotlib.pyplot as plt

# 데이터 로드
df = pd.read_csv(r'C:\Users\jhlee\Desktop\Python\BasicLAB\bus_comfort_data_2000.csv')

# 1) Comfort Score 분포 히스토그램
plt.figure()
df['comfort_score'].hist(bins=50)
plt.title('Comfort Score Distribution')
plt.xlabel('Comfort Score')
plt.ylabel('Frequency')
plt.show()

# 2) 이동 시간 대비 Comfort Score 산점도
plt.figure()
plt.scatter(df['travel_time_min'], df['comfort_score'])
plt.title('Travel Time vs Comfort Score')
plt.xlabel('Travel Time (min)')
plt.ylabel('Comfort Score')
plt.show()

# 3) 혼잡도 대비 Comfort Score 산점도
plt.figure()
plt.scatter(df['congestion_level'], df['comfort_score'])
plt.title('Congestion Level vs Comfort Score')
plt.xlabel('Congestion Level')
plt.ylabel('Comfort Score')
plt.show()

plt.figure()
plt.scatter(df['walk_distance_m'], df['comfort_score'])
plt.title('walk_distance_m vs Comfort Score')
plt.xlabel('walk_distance_m')
plt.ylabel('Comfort Score')
plt.show()

import seaborn as sns

sns.jointplot(data = df, x = 'departure_hour', y = 'comfort_score', kind = 'kde', fill = True)
plt.title("Contour plot")
plt.xlabel("departure hour")
plt.ylabel("comfort_score")
plt.show()

sns.jointplot(data = df, x = 'departure_hour', y = 'congestion_level', kind = 'kde', fill = True)
plt.title("Contour plot")
plt.xlabel("departure hour")
plt.ylabel("congestion_level")
plt.show()