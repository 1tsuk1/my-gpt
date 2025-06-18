```
def apply_transformation_rules(group_name, sub_items):
    """
    共通の変換ルールを適用
    
    ルール:
    - グループ配下のサブグループが1つだけの場合：
      * L行をLなしの行に変換（サブグループをグループに昇格）
      * グループの全行を削除
      * LL行をL行に変換
    
    Returns:
        (transformed_categories, indices_to_keep)
        - transformed_categories: 変換後のカテゴリ名のリスト
        - indices_to_keep: 保持するインデックスのリスト（0=グループ, 1以降=サブ項目）
    """
    if not sub_items:
        return [group_name], [0]
    
    # サブグループとLL項目を分離
    subgroups = []
    subgroup_indices = []
    for i, item in enumerate(sub_items, 1):
        if is_subgroup(item):
            subgroups.append(item)
            subgroup_indices.append(i)
    
    # サブグループが1つだけの場合
    if len(subgroups) == 1:
        subgroup = subgroups[0]
        subgroup_idx = subgroup_indices[0]
        subgroup_name = extract_subgroup_name(subgroup)
        
        # 変換後のカテゴリ（グループは削除、サブグループがグループに昇格）
        result_categories = [subgroup_name]
        indices_to_keep = [subgroup_idx]  # サブグループのインデックスを使用
        
        # LL項目を処理
        for i, item in enumerate(sub_items, 1):
            if is_ll_item(item):
                result_categories.append(convert_ll_to_l(item))
                indices_to_keep.append(i)
        
        return result_categories, indices_to_keep```
