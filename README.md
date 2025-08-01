
### SQL(Snowflake)に関するレビュー項目

#### **パフォーマンス・効率性**
- 必要なカラムのみをSELECTしているか（SELECT * の使用を避ける）
- WHERE句で適切な絞り込みを行っているか
- JOINの条件が明確に指定されているか（デカルト積を避ける）
- サブクエリよりCTEを使用しているか
- 同じサブクエリが複数回記述されていないか（CTEで共通化できる）
- 同じテーブルに対する類似した条件での複数回JOINを避けているか
- UNION ALLで十分な場面でUNIONを使っていないか
- 不必要なDISTINCTやGROUP BYを使用していないか

#### **可読性・保守性**
- 適切なインデントと改行が行われているか
- WITH句の各中間テーブルで何の処理をしているかコメントが記述されているか
- テーブルエイリアスが一貫性のある命名規則に従っているか（t1, t2, t3... または意味のある短縮名）
- JOIN時に全てのカラムがテーブルエイリアスで修飾されているか（例: t1.column_name）
- 命名規則が統一されているか（大文字・小文字の使い分け）
- 不要なコードやコメントアウトされたコードが残っていないか
- 複雑なロジックに適切なコメントが付けられているか


## SQL
```
-- 売上分析クエリ
-- 作成日: 2024-01-01
-- 作成者: テスト太郎

-- TODO: 後で修正する
-- SELECT * FROM orders WHERE status = 'cancelled';

SELECT * 
FROM sales_data
WHERE 1=1
;

-- 顧客ごとの売上集計
SELECT 
    customer_id,
    SUM(amount) as total_amount
FROM (
    SELECT 
        customer_id,
        amount
    FROM sales_data
    WHERE created_at >= '2024-01-01'
) subquery1
GROUP BY customer_id;

-- 重複を含む商品リスト
SELECT product_id, product_name
FROM products
WHERE category = 'Electronics'
UNION
SELECT product_id, product_name  
FROM products
WHERE category = 'Computers';

-- 売上と在庫の結合
SELECT 
    sales.order_id,
    sales.product_id,
    inventory.quantity,
    sales.amount
FROM sales_data sales, inventory inventory
WHERE sales.product_id = inventory.product_id;

-- 同じ条件で複数回結合
SELECT 
    o.order_id,
    o.customer_id,
    c1.customer_name,
    c2.email,
    c3.phone
FROM orders o
LEFT JOIN customers c1 ON o.customer_id = c1.id AND c1.is_active = TRUE
LEFT JOIN customers c2 ON o.customer_id = c2.id AND c2.is_active = TRUE  
LEFT JOIN customers c3 ON o.customer_id = c3.id AND c3.is_active = TRUE;

-- 不要なDISTINCT
SELECT DISTINCT 
    order_id,
    customer_id,
    order_date
FROM orders
WHERE order_id IS NOT NULL;

-- 重複したサブクエリ
SELECT 
    (SELECT COUNT(*) FROM orders WHERE status = 'completed') as completed_orders,
    (SELECT COUNT(*) FROM orders WHERE status = 'pending') as pending_orders,
    (SELECT COUNT(*) FROM orders WHERE status = 'completed') / 
    (SELECT COUNT(*) FROM orders) * 100 as completion_rate;```

## Python
```
#!/usr/bin/env python
# データ処理スクリプト

import pandas as pd
import snowflake.connector
from datetime import datetime

# グローバル変数
CONNECTION_PARAMS = {
    'user': 'admin',
    'password': 'password123',  # ハードコードされたパスワード
    'account': 'myaccount'
}

def process_sales_data():
    # データベース接続
    conn = snowflake.connector.connect(**CONNECTION_PARAMS)
    cursor = conn.cursor()
    
    # 全データを一度に読み込む
    query = "SELECT * FROM large_sales_table"
    cursor.execute(query)
    all_data = cursor.fetchall()
    
    # データフレームに変換
    df = pd.DataFrame(all_data)
    
    # 処理1: 売上集計
    total_sales = 0
    for index,row in df.iterrows():
        total_sales = total_sales + row[3]  # マジックナンバー
    
    print("合計売上: " + str(total_sales))
    
    # 処理2: カテゴリ別集計（同じ処理の繰り返し）
    electronics_sales = 0
    for index,row in df.iterrows():
        if row[5] == 'Electronics':
            electronics_sales = electronics_sales + row[3]
            
    clothing_sales = 0
    for index,row in df.iterrows():
        if row[5] == 'Clothing':
            clothing_sales = clothing_sales + row[3]
    
    # ユーザー入力を受け付ける
    category = input("検索するカテゴリを入力してください: ")
    
    # SQLインジェクションの脆弱性
    query2 = "SELECT * FROM products WHERE category = '" + category + "'"
    cursor.execute(query2)
    
    # エラーハンドリングなし
    results = cursor.fetchall()
    
    # 不要なコード
    # for r in results:
    #     print(r)
    
    # TODO: リファクタリング必要
    
    return df

# メイン処理
if __name__ == "__main__":
    # PEP8違反：関数名が不適切
    def LoadDataFromCSV(filepath):
        data=pd.read_csv(filepath)
        return data
    
    # 大量データを一度にメモリに読み込む
    huge_df = pd.read_csv("huge_file.csv")
    
    # 非効率なループ処理
    result_list = []
    for i in range(len(huge_df)):
        if huge_df.iloc[i]['amount'] > 1000:
            result_list.append(huge_df.iloc[i])
    
    # データ処理実行
    processed_data = process_sales_data()
    
    # 未使用の変数
    temp_var = 100
    another_temp = "test"
    
    print("処理完了")```
