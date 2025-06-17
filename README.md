```
def transform_vertical_dataframe(df, column_name='category', value_columns=None):
    """
    縦持ちデータフレーム（同じカテゴリが複数行に存在）を階層ルールに基づいて整形する
    
    Parameters:
    - df: 変換対象のDataFrame
    - column_name: カテゴリ列の名前
    - value_columns: 保持する値列のリスト（Noneの場合は全列を保持）
    
    Returns:
    - 変換後のDataFrame
    """
    # カテゴリのユニークな順序を保持
    unique_categories = []
    seen = set()
    for cat in df[column_name]:
        if cat not in seen:
            unique_categories.append(cat)
            seen.add(cat)
    
    # 各カテゴリのデータを収集
    category_data = {}
    for cat in unique_categories:
        category_data[cat] = df[df[column_name] == cat].copy()
    
    # 横持ちと同じロジックで変換ルールを決定
    result_categories = []
    i = 0
    while i < len(unique_categories):
        current = unique_categories[i]
        
        # グループ（Lで始まらない）の場合
        if not current.startswith('L'):
            group_name = current
            i += 1
            
            # このグループに属する項目を収集
            sub_items = []
            while i < len(unique_categories) and unique_categories[i].startswith('L'):
                sub_items.append(unique_categories[i])
                i += 1
            
            # グループとその配下を処理
            processed = process_vertical_group(group_name, sub_items)
            result_categories.extend(processed)
        else:
            # 独立したL/LL項目
            result_categories.append(current)
            i += 1
    
    # 新しいDataFrameを構築
    result_dfs = []
    for cat in result_categories:
        if cat in category_data:
            # 既存のカテゴリデータをそのまま使用
            result_dfs.append(category_data[cat])
        else:
            # 変換されたカテゴリ名の場合、元のデータを探す
            # LL -> L の変換を確認
            if cat.startswith('L ') and not cat.startswith('LL '):
                # 元のLL版を探す
                original_ll = 'LL ' + cat[2:]
                if original_ll in category_data:
                    df_copy = category_data[original_ll].copy()
                    df_copy[column_name] = cat
                    result_dfs.append(df_copy)
                    continue
            
            # グループ名の変換を確認（サブグループ名への変換）
            # 例: "web広告e_fuga" <- "web広告e"
            for original_cat in category_data:
                if not original_cat.startswith('L'):
                    # 対応するサブグループを探す
                    expected_subgroup = f"L {cat}"
                    if expected_subgroup in unique_categories:
                        # このグループのデータを使用
                        df_copy = category_data[original_cat].copy()
                        df_copy[column_name] = cat
                        result_dfs.append(df_copy)
                        break
    
    # 結果を結合
    if result_dfs:
        result_df = pd.concat(result_dfs, ignore_index=True)
        return result_df
    else:
        return pd.DataFrame()

def process_vertical_group(group_name, sub_items):
    """
    縦持ちデータ用のグループ処理
    """
    if not sub_items:
        return [group_name]
    
    # サブグループとLL項目を分離
    subgroups = [item for item in sub_items if item.startswith('L ') and not item.startswith('LL ')]
    ll_items = [item for item in sub_items if item.startswith('LL ')]
    
    # サブグループが1つだけの場合
    if len(subgroups) == 1:
        subgroup = subgroups[0]
        # サブグループ名を取得
        subgroup_name = subgroup[2:]
        
        # 結果
        result = [subgroup_name]
        
        # LL項目をLに変換
        for ll_item in ll_items:
            l_item = ll_item.replace('LL ', 'L ', 1)
            result.append(l_item)
        
        return result
    
    # その他の場合は変更なし
    result = [group_name]
    result.extend(sub_items)
    return result
```
