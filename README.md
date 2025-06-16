
def validate_group_dependency(group_selection, sub_group_selection, group_not_select_name):
    """
    分計用グループが未選択で分計用サブグループが選択されている場合にエラーを発生させる関数
    
    Args:
        group_selection: 分計用グループの選択
        sub_group_selection: 分計用サブグループの選択
        group_not_select_name: 未選択を表す選択肢名
    """
    if group_selection == group_not_select_name and sub_group_selection != group_not_select_name:
        st.error(
            "分計用グループが未選択の場合、分計用サブグループは選択できません。"
            "分計用グループを選択するか、分計用サブグループを未選択にしてください。"
        )
        st.stop()



    validate_group_dependency(
        st.session_state.multi_service_group_selection,
        st.session_state.multi_service_sub_group_selection,
        group_not_select_name
    )
