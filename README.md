
```
import pandas as pd
import streamlit as st

# データフレームの作成
df = pd.DataFrame({
    "項目": ["A", "B", "C", "D"],
    "パーセント": [10.34, 9.57, 15.2, 7.45],  # 数値型
})

# Streamlit アプリ
st.dataframe(
    df,
    column_config={
        "パーセント": st.column_config.NumberColumn(
            "パーセント (%)",
            format="%.2f%%"
        )
    }
)


```
<img width="796" alt="スクリーンショット 2025-03-19 11 27 25" src="https://github.com/user-attachments/assets/e0e821c8-61f4-4f4a-b63e-d569b59b27ec" />

![Uploading スクリーンショット 2025-03-19 11.27.25.png…]()
