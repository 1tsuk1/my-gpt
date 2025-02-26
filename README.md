```
import pandas as pd

data = pd.DataFrame({
    '1': [50],
    '2': [30],
    '3': [20]
}, index=['ユーザ数'])

# 前月減少率の計算
reduction_rates = [None] + [data.iloc[0, i] / data.iloc[0, i - 1] for i in range(1, data.shape[1])]
data.loc['前月減少率'] = reduction_rates

# 減少率を適用したデータAの計算（初期値を設定）
data.loc['減少率を適用したデータA'] = 0  # 初期化
data.loc['減少率を適用したデータA', '1'] = 100  # 初期値設定

for i in range(2, data.shape[1] + 1):
    prev_col = str(i - 1)
    curr_col = str(i)
    data.loc['減少率を適用したデータA', curr_col] = data.loc['減少率を適用したデータA', prev_col] * data.loc['前月減少率', curr_col]

# 歩留まり率の計算（初期値はNone）
yield_rates = [None] + [data.loc['減少率を適用したデータA', str(i)] / 100 for i in range(2, data.shape[1] + 1)]
data.loc['歩留まり率'] = yield_rates

# 結果の表示
print(data)
```
