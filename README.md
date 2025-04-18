```
port pandas as pd
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
