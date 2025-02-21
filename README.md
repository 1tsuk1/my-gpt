```
import streamlit as st

# 初期オプション
options = ["%Y%m", "%Y/%m/%d", "その他"]

# ラジオボタンで選択
choice = st.radio("選択してください", options, horizontal=True)

# 「カスタム」が選択された場合はテキスト入力を表示
if choice == "その他":
    custom_value = st.text_input("カスタム設定を入力してください")
    final_value = custom_value
else:
    final_value = choice

st.write(f"選択された値: {final_value}")
```
