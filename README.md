## GitHub Copilot カスタムインストラクションの効果的な書き方ガイド

### 📚 参考資料
- **公式ドキュメント**: [https://copilot-instructions.md/](https://copilot-instructions.md/)
  - GitHub Copilotのカスタムインストラクション機能の公式説明
  - 現在パブリックプレビュー中の機能

### 📌 基本原則

#### ✅ **DO - 推奨される書き方**

1. **短く自己完結した文章**
   ```markdown
   ❌ 悪い例：コーディング規約についてはstyleguide.mdを参照してください
   ✅ 良い例：変数名はcamelCase、クラス名はPascalCase、定数はUPPER_SNAKE_CASEを使用
   ```

2. **プロジェクト固有の事実を簡潔に**
   ```markdown
   ✅ Snowflakeを使用したデータ分析プロジェクト、Python 3.9以上が必須
   ✅ Next.js 14のApp Router使用、React Server Componentsを優先
   ✅ 医療系SaaSアプリケーション、HIPAA準拠が必要
   ```

3. **技術スタックと依存関係を明確に**
   ```markdown
   ✅ 主要技術：TypeScript、Prisma、PostgreSQL、TailwindCSS
   ✅ テストはJest使用、E2EテストはPlaywright使用
   ✅ パッケージ管理はpnpm、ビルドツールはVite使用
   ```

#### ❌ **DON'T - 避けるべき書き方**

1. **外部参照の要求**
   ```markdown
   ❌ coding-standards.mdに従ってコードを書いて
   ❌ 会社のWikiを参照して実装して
   ```

2. **回答スタイルの指定**
   ```markdown
   ❌ フレンドリーな口調で回答して
   ❌ 1000文字以内で説明して
   ❌ 初心者にもわかるように説明して
   ```

3. **Copilotの動作方法の指定**
   ```markdown
   ❌ 必ず@terminalを使ってGitの質問に答えて
   ❌ 常に3つ以上の選択肢を提示して
   ```

### 🎯 効果的な構成パターン

#### 1. **カテゴリ別整理法**
```markdown
## 技術スタック
TypeScript 5.0+、React 18、Next.js 14 App Router使用
状態管理はZustand、フォーム処理はreact-hook-form使用

## コーディング規約
ESLint設定に準拠、Prettierで自動フォーマット適用
命名規則：変数・関数はcamelCase、型・インターフェースはPascalCase

## アーキテクチャ
Clean Architecture採用、ドメイン層は外部依存なし
APIはtRPC使用、認証はNextAuth実装
```

#### 2. **優先度別記載法**
```markdown
## 必須要件
Snowflake SQL使用、すべてのクエリで実行計画確認
TypeScriptのstrictモード有効、型定義必須

## 推奨事項
React Server Components優先、クライアントコンポーネントは最小限
パフォーマンス最適化：メモ化、遅延読み込み活用

## 補足情報
開発環境：Node.js 20+、pnpm 8+使用
CI/CD：GitHub Actions使用、mainブランチ自動デプロイ
```

#### 3. **ドメイン特化型**
```markdown
## プロジェクト概要
医療データ分析システム、HIPAA準拠必須、PHI取り扱い注意

## データ処理
大規模データセット対応、バッチ処理でchunksize設定必須
Snowflakeパーティション活用、クラスタリングキー最適化

## セキュリティ
すべてのAPIエンドポイントで認証必須
個人情報は必ず暗号化、ログに個人情報出力禁止
```

### 💡 実践的なテクニック

#### 1. **具体的な数値や設定値を含める**
```markdown
✅ レスポンスタイムは3秒以内、バンドルサイズは500KB以下
✅ 1関数は50行以内、循環的複雑度は10以下
```

#### 2. **「なぜ」ではなく「何を」を記載**
```markdown
❌ パフォーマンスのためにメモ化を使用する
✅ React.memo、useMemo、useCallbackを適切に使用
```

#### 3. **ツール固有の設定を明記**
```markdown
✅ ESLintはairbnb-typescript設定使用
✅ テストカバレッジ80%以上維持
✅ Snowflakeウェアハウスサイズは自動スケーリング有効
```

### 📝 テンプレート例

```markdown
## プロジェクト基本情報
[プロジェクトタイプ]、[主要技術スタック]使用
[特殊な要件や制約]

## 技術要件
[言語/フレームワーク]バージョン[X.X]使用、[主要ライブラリ]で実装
[コーディング規約]準拠、[テストフレームワーク]でテスト実装

## 重要な慣例
[命名規則]、[ファイル構成]、[その他のプロジェクト固有ルール]
```

### 🔄 メンテナンスのコツ

1. **定期的な見直し**
   - 四半期ごとに内容を更新
   - 新しい技術導入時は即座に追記

2. **チームでの合意形成**
   - PRレビューで指示内容も確認
   - 新メンバー参加時に説明

3. **効果測定**
   - Copilotの提案品質を定期的に評価
   - 不要な指示は削除

### 🔗 関連リソース
- **設定ファイルの場所**: `.github/copilot-instructions.md`
- **対応エディタ**: VS Code、Visual Studio
- **設定項目**: 
  - VS Code: "Code Generation: Use Instruction Files"
  - Visual Studio: "(Preview) Enable custom instructions"

このガイドラインに従うことで、Copilotがプロジェクトに最適化された高品質なコード提案を行えるようになります。
