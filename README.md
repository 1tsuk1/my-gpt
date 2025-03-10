```import pandas as pd

# サンプルデータ
data = {
    "期間": ["A_202405", "A_202406", "A_202407", "B_202405"],
    "hoge": [5, 15, 25, 35],
}

df = pd.DataFrame(data)

# "期間" を分割して "グループ"（A, B）と "年月"（YYYYMM）を抽出
df["グループ"] = df["期間"].str.split("_").str[0]  # "A" または "B"
df["年月"] = df["期間"].str.extract(r"(\d{6})")[0]  # "YYYYMM"

# "YYYYMM" を日付型に変換（集計しやすくするため）
df["年月"] = pd.to_datetime(df["年月"], format="%Y%m")

# 各グループごとに最小・最大の年月を求める
grouped = df.groupby("グループ")["年月"].agg([min, max]).reset_index()

# "YYYY/MM〜YYYY/MM" の形式に変換
grouped["期間"] = grouped["min"].dt.strftime("%Y/%m") + "〜" + grouped["max"].dt.strftime("%Y/%m")

# 必要なカラムを選択
result_df = grouped[["グループ", "期間"]].rename(columns={"グループ": "A"})

# 結果表示
print(grouped)
print(result_df)
```
