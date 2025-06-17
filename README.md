```
import pandas as pd

def transform_dataframe(df, column_name='category'):
    """
    データフレームの指定列を階層ルールに基づいて整形する
    他の列も保持する
    
    ルール:
    - 「L」は階層を表す
    - 「L」が1つもないものを「グループ」、「L」が1つのものを「サブグループ」とする
    - グループ配下のサブグループが1つだけの時:
      * グループ名をサブグループ名にする
      * サブグループ行を削除する
      * 「L」が2つの行は「L」を1つにする
    """
    # 元のデータフレームのコピーを作成
    df_copy = df.copy()
    
    # 処理対象の列データ
    data = df_copy[column_name].tolist()
    
    # 結果を格納するリスト（インデックスも保持）
    result_indices = []  # 元のDataFrameのインデックス
    result_values = []   # 変換後の値
    
    i = 0
    while i < len(data):
        current = data[i]
        current_idx = df_copy.index[i]
        
        # グループ（Lで始まらない）の場合
        if not current.startswith('L'):
            group_name = current
            group_idx = current_idx
            i += 1
            
            # このグループに属する項目を収集
            sub_items = []
            sub_indices = []
            while i < len(data) and data[i].startswith('L'):
                sub_items.append(data[i])
                sub_indices.append(df_copy.index[i])
                i += 1
            
            # グループとその配下を処理
            processed_items, processed_indices = process_group(
                group_name, group_idx, sub_items, sub_indices
            )
            result_indices.extend(processed_indices)
            result_values.extend(processed_items)
        else:
            # 独立したL/LL項目（通常はグループから処理するためここには来ない）
            result_indices.append(current_idx)
            result_values.append(current)
            i += 1
    
    # 新しいDataFrameを作成（保持するインデックスのみ）
    result_df = df_copy.loc[result_indices].copy()
    result_df[column_name] = result_values
    
    # インデックスをリセット（必要に応じて）
    result_df = result_df.reset_index(drop=True)
    
    return result_df

def process_group(group_name, group_idx, sub_items, sub_indices):
    """
    グループとその配下の項目を処理する
    Returns: (処理後の値のリスト, 対応するインデックスのリスト)
    """
    if not sub_items:
        return [group_name], [group_idx]
    
    # サブグループ（L で始まり、LL でない）を抽出
    subgroups = []
    subgroup_indices = []
    ll_items = []
    ll_indices = []
    
    for item, idx in zip(sub_items, sub_indices):
        if item.startswith('L ') and not item.startswith('LL '):
            subgroups.append(item)
            subgroup_indices.append(idx)
        elif item.startswith('LL '):
            ll_items.append(item)
            ll_indices.append(idx)
    
    # サブグループが1つだけの場合
    if len(subgroups) == 1:
        subgroup = subgroups[0]
        # サブグループ名を取得（"L " を除去）
        subgroup_name = subgroup[2:]  # "L web広告e_fuga" → "web広告e_fuga"
        
        # 結果の値とインデックス
        result_values = [subgroup_name]
        result_indices = [group_idx]  # グループのインデックスを使用
        
        # LL項目を処理（LLをLに変換）
        for ll_item, ll_idx in zip(ll_items, ll_indices):
            # "LL " を "L " に置換
            l_item = ll_item.replace('LL ', 'L ', 1)
            result_values.append(l_item)
            result_indices.append(ll_idx)
        
        return result_values, result_indices
    
    # サブグループが複数ある場合、または0個の場合は変更なし
    result_values = [group_name]
    result_indices = [group_idx]
    
    # すべてのサブ項目を追加
    result_values.extend(sub_items)
    result_indices.extend(sub_indices)
    
    return result_values, result_indices```
    
