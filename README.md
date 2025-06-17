```
import pandas as pd

def transform_dataframe(df, column_name):
    """
    データフレームの指定列を階層ルールに基づいて整形する
    
    ルール:
    - 「L」は階層を表す
    - 「L」が1つもないものを「グループ」、「L」が1つのものを「サブグループ」とする
    - グループ配下のサブグループが1つだけの時:
      * グループ名をサブグループ名にする
      * サブグループ行を削除する
      * 「L」が2つの行は「L」を1つにする
    """
    data = df[column_name].tolist()
    result = []
    
    i = 0
    while i < len(data):
        current = data[i]
        
        # グループ（Lで始まらない）の場合
        if not current.startswith('L'):
            group_name = current
            i += 1
            
            # このグループに属する項目を収集
            sub_items = []
            while i < len(data) and data[i].startswith('L'):
                sub_items.append(data[i])
                i += 1
            
            # グループとその配下を処理
            processed = process_group(group_name, sub_items)
            result.extend(processed)
        else:
            # 独立したL/LL項目（通常はグループから処理するためここには来ない）
            result.append(current)
            i += 1
    
    return pd.DataFrame({column_name: result})

def process_group(group_name, sub_items):
    """
    グループとその配下の項目を処理する
    """
    if not sub_items:
        return [group_name]
    
    # サブグループ（L で始まり、LL でない）を抽出
    subgroups = [item for item in sub_items if item.startswith('L ') and not item.startswith('LL ')]
    # LL項目を抽出
    ll_items = [item for item in sub_items if item.startswith('LL ')]
    
    # サブグループが1つだけの場合
    if len(subgroups) == 1:
        subgroup = subgroups[0]
        # サブグループ名を取得（"L " を除去）
        subgroup_name = subgroup[2:]  # "L web広告e_fuga" → "web広告e_fuga"
        
        # グループ名をサブグループ名に変更
        result = [subgroup_name]
        
        # LL項目を処理（LLをLに変換）
        for ll_item in ll_items:
            # "LL " を "L " に置換
            l_item = ll_item.replace('LL ', 'L ', 1)
            result.append(l_item)
        
        return result
    
    # サブグループが複数ある場合、または0個の場合は変更なし
    result = [group_name]
    result.extend(sub_items)
    return result

# テスト関数
def test_transformation():
    """変換のテスト"""
    # テストデータ
    test_data = {
        'category': [
            "web広告",
            "L web広告",
            "LL web広告_202404",
            "LL web広告_202405",
            "web広告_hoge",
            "L web広告_hoge",
            "LL web広告_hoge_202404",
            "LL web広告_hoge_202405",
            "L web広告_piyo",
            "LL web広告_piyo_202404",
            "LL web広告_piyo_202405",
            "web広告e",
            "L web広告e_fuga",
            "LL web広告e_fuga_202404",
            "LL web広告e_fuga_202405"
        ]
    }
    
    # 期待される結果
    expected = [
        "web広告",
        "L web広告_202404",
        "L web広告_202405",
        "web広告_hoge",
        "L web広告_hoge",
        "LL web広告_hoge_202404",
        "LL web広告_hoge_202405",
        "L web広告_piyo",
        "LL web広告_piyo_202404",
        "LL web広告_piyo_202405",
        "web広告e_fuga",
        "L web広告e_fuga_202404",
        "L web広告e_fuga_202405"
    ]
    
    # DataFrameを作成して変換
    df = pd.DataFrame(test_data)
    result_df = transform_dataframe(df, 'category')
    
    # 結果を表示
    print("=== 変換前 ===")
    for item in test_data['category']:
        print(item)
    
    print("\n=== 変換後 ===")
    for item in result_df['category']:
        print(item)
    
    print("\n=== 期待される結果 ===")
    for item in expected:
        print(item)
    
    # 検証
    print("\n=== 検証結果 ===")
    result_list = list(result_df['category'])
    if result_list == expected:
        print("✓ 成功: 期待通りの結果です！")
    else:
        print("✗ 失敗: 結果が異なります")
        print("\n差分:")
        max_len = max(len(result_list), len(expected))
        for i in range(max_len):
            if i < len(result_list) and i < len(expected):
                if result_list[i] != expected[i]:
                    print(f"  行{i+1}: '{result_list[i]}' ≠ '{expected[i]}'")
            elif i >= len(result_list):
                print(f"  行{i+1}: (なし) ≠ '{expected[i]}'")
            else:
                print(f"  行{i+1}: '{result_list[i]}' ≠ (なし)")

# 使用例
def main():
    """メイン実行関数"""
    # テストを実行
    test_transformation()
    
    print("\n" + "="*60)
    print("実際の使用方法:")
    print("="*60)
    print("""
# データフレームの読み込み
import pandas as pd
df = pd.read_csv('your_file.csv')  # または既存のDataFrame

# 変換の実行
df_transformed = transform_dataframe(df, 'category')  # 'category'は列名

# 結果の確認
print(df_transformed)

# 結果の保存
df_transformed.to_csv('transformed_file.csv', index=False)
""")

if __name__ == "__main__":
    main()
```
