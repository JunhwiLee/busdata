import pandas as pd
from scipy.stats import chi2_contingency, kruskal

# 1) 데이터 불러오기 (df)
df = pd.read_csv("combined.csv")          # YYYYMM, 06on, 06off, …

# 2) Long → contingency table (카이제곱)
cols_onoff = [c for c in df.columns if c.endswith(("on", "off"))]
long = df.melt(id_vars=['YYYYMM'], value_vars=cols_onoff,
               var_name='hour_type', value_name='cnt')
long['hour'] = long['hour_type'].str[:2]    # "06on" → "06"

ct = long.pivot_table(index='YYYYMM', columns='hour', values='cnt',
                      aggfunc='sum', fill_value=0)

chi2, p, dof, ex = chi2_contingency(ct)
print(f"χ² ={chi2:.2f}, p ={p:.4g}")

# 3) Kruskal–Wallis (비모수)
groups = [long.loc[long['YYYYMM'] == m, 'cnt'] for m in long['YYYYMM'].unique()]
H, p_kw = kruskal(*groups)
print(f"H ={H:.2f}, p ={p_kw:.4g}")
