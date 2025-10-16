import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb

from google.colab import drive
drive.mount('/content/drive')
base_path = '/content/drive/My Drive/'
train_df = pd.read_csv(base_path + 'train.csv')
test_df  = pd.read_csv(base_path + 'test.csv')

data_all = pd.concat((train_df, test_df),sort=False).reset_index(drop=True) #639999 train + 160000 test

data_all.shape #(799999, 35)

display(data_all.describe(include="all").transpose())

data_all.head()

data_all.dtypes

#畫曲線圖
plt.rcParams['figure.figsize'] = (4, 3)   # (寬, 高) 4x3 吋
plt.rcParams['figure.dpi'] = 100          # 解析度適中即可
plt.rcParams['axes.titlesize'] = 10       # 標題字體小一點

num_data = data_all.select_dtypes(['int64', 'float64'])
for col in num_data.columns:
    fig, ax = plt.subplots()                      # 自己控管 fig
    # distplot 已被棄用，改 histplot/kdeplot
    sns.histplot(num_data[col].dropna(), ax=ax)
    ax.set_title(col)
    fig.tight_layout()
    plt.show()

cat_data = data_all.select_dtypes(['object'])
for col in cat_data.columns:
    fig, ax = plt.subplots()
    # 用 value_counts 畫長條，可限制前 N 類避免太擠
    vc = cat_data[col].value_counts(dropna=True)
    ax.bar(vc.index[:20], vc.values[:20])
    ax.set_xticklabels(vc.index[:20], rotation=90, fontsize=7)
    ax.set_title(col)
    fig.tight_layout()
    plt.show()
#-------------------------------------------------------------------

#轉日期格式
sig = (data_all['fl_date']
       .str.lower()
       .str.replace(r"\d", "D", regex=True)       # 數字→D
       .str.replace(r"[a-z]+", "A", regex=True)   # 連續字母→A
       .str.replace(r"\s+", "␠", regex=True))     # 空白可視化
sig.value_counts().head(20)

col = "fl_date"

# 1) 先清理字串（去空白、把空值標記成 NaN）
s = (data_all[col].astype(str)
               .str.strip()
               .replace({"": np.nan, "NA": np.nan, "N/A": np.nan, "null": np.nan, "-": np.nan}))

# 2) 只挑長得像 YYYY/M/D 或 YYYY/MM/DD 的進來轉
mask_ymd = s.str.match(r"^\d{4}/\d{1,2}/\d{1,2}$", na=False)

parsed = pd.to_datetime(s.where(mask_ymd), format="%Y/%m/%d", errors="coerce")

# 3) 寫回欄位（dtype 將會是 datetime64[ns]）
data_all[col] = parsed

# 4) 檢查還有沒有轉不過去的（理論上應該變 0）
bad_left = data_all[col].isna() & s.notna()
print("Unparsed still left:", bad_left.sum())

data_all['fl_date'].dtype #<M8[ns]>
#---------------------------------------------------------------------

missing_columns=data_all.isnull().mean().sort_values(ascending=False)
missing_columns=missing_columns[missing_columns!=0].to_frame().reset_index()

fig,ax=plt.subplots(figsize=(7,7))
sns.barplot(x=0,y='index',data=missing_columns)

missing_columns.columns = ["column", "missing_rate"]
missing_columns["dtype"] = missing_columns["column"].map(data_all.dtypes.astype(str))
display(missing_columns)

data_all=data_all.drop(columns=['cancellation_code'],axis=1)
data_all.shape #(799999, 34)

hhmm = ['crs_dep_time', 'crs_arr_time', 'dep_time', 'arr_time', 'wheels_off', 'wheels_on']
n_train = len(train_df)
is_train = np.arange(len(data_all)) < n_train

# 清理train：丟掉缺arr_delay + HHMM 欄位有缺值的列
need_cols = ['arr_delay'] + [c for c in hhmm if c in data_all.columns]  
train_clean = (data_all.loc[is_train]
               .dropna(subset=need_cols)
               .copy())

# test保留原樣
test_keep = data_all.loc[~is_train].copy()
if 'arr_delay' in test_keep.columns:
    test_keep = test_keep.drop(columns=['arr_delay'])

print("train before:", is_train.sum(), " -> after:", len(train_clean)) #train before: 639999  -> after: 620218
print("test rows:", len(test_keep)) #test rows: 160000

data_all_clean = pd.concat([train_clean, test_keep], ignore_index=True)
print(len(data_all), '→', len(data_all_clean)) # 799999 → 780218
print('test keep has arr_delay col? ', 'arr_delay' in test_keep.columns) #F

missing_num=['actual_elapsed_time','air_time','taxi_in', 'taxi_out','dep_delay','crs_elapsed_time']
#缺失旗標
for c in missing_num:
    if c in data_all_clean.columns:
        data_all_clean[c + '_missing'] = data_all_clean[c].isna().astype(int)

#補0或中位數
for i in missing_num:
  if i == 'dep_delay':
    data_all_clean[i]=data_all_clean[i].fillna(0)
  else:
    data_all_clean[i]=data_all_clean[i].fillna(train_clean[i].median())

MINS_PER_DAY = 1440

# --- 1) HHMM → 分鐘（含合法性檢查） ---
def hhmm_to_minutes(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    hh = (x // 100)
    mm = (x % 100)
    bad = (hh < 0) | (hh > 23) | (mm < 0) | (mm > 59)
    out = pd.Series(np.nan, index=s.index, dtype="float64")
    ok = (~x.isna()) & (~bad)
    out.loc[ok] = (hh.loc[ok] * 60 + mm.loc[ok]).astype(float)
    return out

# --- 2) 延誤：最近循環差 ∈ [-720, 720)（允許提早為負） ---
def delay_diff(actual_min: pd.Series, sched_min: pd.Series) -> pd.Series:
    return (actual_min - sched_min + MINS_PER_DAY/2) % MINS_PER_DAY - MINS_PER_DAY/2

# --- 3) 時長：跨日取非負 ∈ [0, 1440) ---
def duration_diff(end_min: pd.Series, start_min: pd.Series) -> pd.Series:
    return (end_min - start_min) % MINS_PER_DAY

# === 對 data_all_clean 一次性處理（確定性轉換，無洩漏） ===
hhmm_cols = ["dep_time","arr_time","wheels_off","wheels_on","crs_dep_time","crs_arr_time"]
for c in hhmm_cols:
    if c in data_all_clean.columns:
        data_all_clean[c + "_min"] = hhmm_to_minutes(data_all_clean[c])

# 延誤特徵（允許負值）
if {"dep_time_min","crs_dep_time_min"} <= set(data_all_clean.columns):
    data_all_clean["dep_delay_min"] = delay_diff(
        data_all_clean["dep_time_min"], data_all_clean["crs_dep_time_min"]
    )
if {"arr_time_min","crs_arr_time_min"} <= set(data_all_clean.columns):
    data_all_clean["arr_delay_min"] = delay_diff(
        data_all_clean["arr_time_min"], data_all_clean["crs_arr_time_min"]
    )

# 時長特徵（非負）
if {"wheels_off_min","dep_time_min"} <= set(data_all_clean.columns):
    data_all_clean["taxi_out_min"] = duration_diff(
        data_all_clean["wheels_off_min"], data_all_clean["dep_time_min"]
    )
if {"wheels_on_min","wheels_off_min"} <= set(data_all_clean.columns):
    data_all_clean["air_time_min"] = duration_diff(
        data_all_clean["wheels_on_min"], data_all_clean["wheels_off_min"]
    )
if {"arr_time_min","wheels_on_min"} <= set(data_all_clean.columns):
    data_all_clean["taxi_in_min"] = duration_diff(
        data_all_clean["arr_time_min"], data_all_clean["wheels_on_min"]
    )

cross_sched = data_all_clean["crs_arr_time_min"] < data_all_clean["crs_dep_time_min"]

# 安全的時長（僅在應跨日時 +1440；不該跨日卻為負 → NaN）
def safe_duration(end_min, start_min, cross_flag):
    d = end_min - start_min
    d = d.where(~(d < 0), d + 1440)                # 先假設跨日補 1440
    # 如果本來是負而 cross_flag 為 False，視為異常 → NaN
    bad = (end_min - start_min < 0) & (~cross_flag)
    return d.mask(bad, np.nan)

data_all_clean["air_time_min"] = safe_duration(
    data_all_clean["wheels_on_min"], data_all_clean["wheels_off_min"], cross_sched
)

#把HHMM原本欄位刪除
for i in hhmm_cols:
  data_all_clean=data_all_clean.drop(columns=[i],axis=1)

n_train = len(train_clean) 
trainX = data_all_clean.iloc[:n_train].copy()
testX  = data_all_clean.iloc[n_train:].copy()

# 注意：從這裡開始的「擬合型」步驟（中位數/標準化/分箱/IQR…）
# 只用 trainX 擬合，再 transform 到 testX。

data_all_clean.shape #(780218, 45)

# correlation heatmap
num_train=trainX.select_dtypes(['int64','float64'])
num_corr=num_train.corr()
fig,ax=plt.subplots(figsize=(15,1))
sns.heatmap(num_corr.sort_values(by=['arr_delay'], ascending=False).head(1), cmap='Reds')
plt.title("Correlation Matrix", weight='bold', fontsize=18)
plt.xticks(weight='bold')
plt.yticks(weight='bold', color='dodgerblue', rotation=0)
# top 10
print (num_corr['arr_delay'].sort_values(ascending=False).iloc[1:11])

