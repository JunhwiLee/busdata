import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1) CSV 파일 경로 지정 (필요에 따라 수정하세요)
paths = glob.glob(r'C:\Users\jhlee\Desktop\Python\BasicLAB\combined.csv')  # 실제 CSV 위치에 맞추어 수정

# 2) 모든 파일 읽어서 하나의 DataFrame으로 병합
df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

# 3) 'on' 및 'off' 컬럼들만 추출하여 시간대별 합계 계산
on_cols = [col for col in df.columns if col.endswith('on')]
off_cols = [col for col in df.columns if col.endswith('off')]

hourly_on = df[on_cols].mean().reset_index()
hourly_on.columns = ['hour_col', 'count']
hourly_on['hour'] = hourly_on['hour_col'].str.slice(0, 2).astype(int)
hourly_on['type'] = 'on'

hourly_off = df[off_cols].mean().reset_index()
hourly_off.columns = ['hour_col', 'count']
hourly_off['hour'] = hourly_off['hour_col'].str.slice(0, 2).astype(int)
hourly_off['type'] = 'off'

# 4) 온·오프 데이터를 하나로 합치기
hourly_long = pd.concat([hourly_on, hourly_off], ignore_index=True)

# 5) Seaborn displot으로 겹쳐서 그리기
sns.set_theme(style='whitegrid')
g = sns.displot(
    data=hourly_long,
    x='hour',
    weights='count',
    hue='type',
    bins=24,
    discrete=True,
    multiple='layer',
    height=5,
    aspect=1.8,
)

g.set_axis_labels('Hour of Day', 'passangers (mean)')
g.fig.suptitle('Hourly Onboardings vs Alightings', y=1.02)
plt.xticks(range(24))
plt.tight_layout()
plt.show()
