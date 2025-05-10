import time
import re
import numpy as np
import pandas as pd
import tempfile

from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------
# 1) Load & filter players > 900 mins
# ---------------------------------
df = pd.read_csv('SourceCode/results.csv')
df = df[df['Name'] != 'Mohammed Salah']
minute_col = next(c for c in df.columns if 'min' in c.lower())
df = df[df[minute_col] > 900].copy()
print(f"Players >900 mins: {len(df)}")

# ---------------------------------
# 2) Scrape transfer values
# ---------------------------------
scraped = []
edge_driver_path = r'I:/edgedriver_win64/msedgedriver.exe'
edge_options = Options()
edge_options.add_argument("--headless")
edge_options.add_argument("--disable-gpu")
edge_options.add_argument("--disable-features=SignInService")
edge_options.add_argument("--no-sandbox")
edge_options.page_load_strategy = 'eager'
# Unique profile
temp_profile = tempfile.mkdtemp()
edge_options.add_argument(f"--user-data-dir={temp_profile}")
service = Service(edge_driver_path)
driver = webdriver.Edge(service=service, options=edge_options)

def scrape_stats(url):
    driver.get(url)
    time.sleep(5)
    rows = driver.find_elements(By.CSS_SELECTOR, 'tbody#player-table-body > tr')
    data = []
    for row in rows:
        try:
            name = row.find_element(By.CSS_SELECTOR, 'td.td-player div.text > a').text
            price = row.find_element(By.CSS_SELECTOR, 'span.player-tag').text
            data.append({'Name': name, 'transfer_value': price})
        except:
            continue
    return data

url = 'https://www.footballtransfers.com/us/values/players/most-valuable-soccer-players/playing-in-uk-premier-league'
for i in range(1, 23):
    page = url if i == 1 else f"{url}/{i}"
    scraped.extend(scrape_stats(page))
driver.quit()
vals_df = pd.DataFrame(scraped).drop_duplicates('Name')
print(f"Unique scraped values: {len(vals_df)}")

df = df.merge(vals_df, on='Name', how='left')
print(f"Missing transfer_value: {df['transfer_value'].isna().sum()} of {len(df)}")

# ---------------------------------
# 3) Parse transfer_value to numeric
# ---------------------------------
def parse_val(v):
    if pd.isna(v): return np.nan
    s = str(v).replace('€','').replace(' ','').upper()
    m = re.match(r'([\d\.]+)([MK]?)', s)
    if not m: return np.nan
    num, suf = float(m.group(1)), m.group(2)
    return num if suf in ('M','') else num/1000

df['value_num'] = df['transfer_value'].apply(parse_val)

# ---------------------------------
# 4) Parse Age
# ---------------------------------
def parse_age(val):
    if pd.isna(val): return np.nan
    parts = str(val).split('-')
    if len(parts)==2:
        y,d = map(int, parts)
        return y + d/365
    try: return float(val)
    except: return np.nan

if 'Age' in df.columns:
    df['Age'] = df['Age'].apply(parse_age)

# ---------------------------------
# 5) Feature engineering
# ---------------------------------
# minutes in 90s
df['90s_played'] = df[minute_col] / 90
# per 90 metrics
df['goals_per90'] = df['Goals'] / df['90s_played']
df['assists_per90'] = df['Assists'] / df['90s_played']
# xG vs actual diff
df['xG_diff'] = df['xG'] - df['Goals']
# clean infinite and NaN
for col in ['goals_per90','assists_per90']:
    df[col].replace([np.inf, -np.inf], np.nan, inplace=True)

# ---------------------------------
# 6) Select features & clean
# ---------------------------------
base_feats = ['Age','90s_played','goals_per90','assists_per90','xG','xAG','xG_diff',
               'PrgP_x','GCA90','SCA90','Touches','Pass completion','Shoots on target percentage']
features = [f for f in base_feats if f in df.columns]
df_model = df.dropna(subset=features + ['value_num']).copy()
print(f"Modeling rows: {len(df_model)}")
X = df_model[features]
y = df_model['value_num']

# ---------------------------------
# 7) Split & log-transform y
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
y_train_log = np.log1p(y_train)

# ---------------------------------
# 8) GridSearchCV RandomForest
# ---------------------------------
param_grid = {'n_estimators':[50,100],'max_depth':[3,5],'min_samples_split':[10,20]}
rf = RandomForestRegressor(random_state=42,n_jobs=-1)
grid = GridSearchCV(rf,param_grid,cv=5,scoring='r2',n_jobs=-1,verbose=1)
grid.fit(X_train,y_train_log)
best=grid.best_estimator_
print("Best params:",grid.best_params_)

# Predict and inverse log
y_pred_log=best.predict(X_test)
y_pred=np.expm1(y_pred_log)

# Metrics
def eval_metrics(y_true,y_pred_log,y_pred):
    mse = mean_squared_error(y_true,y_pred)
    r2o = r2_score(y_true,y_pred)
    r2l = r2_score(np.log1p(y_true), y_pred_log)
    print(f"MSE(orig):{mse:.2f}, R2(orig):{r2o:.3f}, R2(log):{r2l:.3f}")

print(eval_metrics(y_test,y_pred_log,y_pred))

# Feature importance
print("Feature importances:")
for f,i in sorted(zip(features,best.feature_importances_), key=lambda x:-x[1]): print(f,round(i,3))

# ---------------------------------
# 9) Save preds
# ---------------------------------
out=X_test.copy()
out['true']=y_test
out['pred']=y_pred
out.to_csv('SourceCode/rf_tuned_preds_eng.csv',index=False)
print("Saved rf_tuned_preds_eng.csv")
