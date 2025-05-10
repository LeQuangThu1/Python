# 1) Load & filter players > 900 mins
# ---------------------------------
df = pd.read_csv('SourceCode/results.csv')
df = df[df['Name'] != 'Mohammed Salah']
minute_col = next(c for c in df.columns if 'min' in c.lower())
df = df[df[minute_col] > 900].copy()
print(f"Players >900 mins: {len(df)}")