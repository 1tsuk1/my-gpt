# GitHub Copilot PRレビュー活用ガイド

## 目次
1. [GitHub Copilotの基本情報](#github-copilotの基本情報)
2. [PRレビューを依頼する方法](#prレビューを依頼する方法)
3. [Copilotに指示を与える方法](#copilotに指示を与える方法)
4. [再レビューを依頼する方法](#再レビューを依頼する方法)
5. [制限事項と注意点](#制限事項と注意点)

## GitHub Copilotの基本情報

### 概要
GitHub Copilotは、AIを活用したコーディングアシスタントです。コードの自動補完、チャット機能、そしてPull Request（PR）のレビュー機能を提供しています。

**参考リンク：**
- [GitHub Copilot公式ドキュメント](https://docs.github.com/ja/copilot)
- [GitHub Copilotコードレビューについて](https://docs.github.com/ja/copilot/concepts/code-review/code-review)

### PRレビュー機能の特徴
- **自動コードレビュー**: PRに追加されたコードを分析し、改善提案を提供
- **コメント形式のレビュー**: 常に「Comment」としてレビューを残し、承認やリクエスト変更は行わない（[公式ドキュメント](https://docs.github.com/ja/copilot/how-tos/use-copilot-agents/request-a-code-review/use-code-review)）
- **提案の適用**: 可能な場合、ワンクリックで適用できる修正提案を含む
- **多言語対応**: 日本語でのレビューコメントにも対応

### 利用可能なプラン
- GitHub Copilot Free: 月50件のチャットメッセージ
- GitHub Copilot Pro/Pro+: より多くの機能とリクエスト数
- GitHub Copilot Business/Enterprise: 組織向けの高度な機能

**詳細情報：**
- [GitHub Copilotの料金プラン](https://github.com/features/copilot)

## PRレビューを依頼する方法

### レビュアーとしてCopilotを追加する

公式ドキュメント「[Using GitHub Copilot code review](https://docs.github.com/ja/copilot/how-tos/use-copilot-agents/request-a-code-review/use-code-review)」に基づく手順：

1. **PRページを開く**
   - GitHub.comで対象のPull Requestページにアクセス

2. **Reviewersメニューを開く**
   - PR画面右側の「Reviewers」セクションをクリック

3. **Copilotを選択**
   - レビュアーリストから「Copilot」を選択
   - 選択後、自動的にレビューが開始される

4. **レビュー完了を待つ**
   - 通常30秒以内にレビューが完了
   - コメントがPR上に表示される

**実例紹介記事：**
- [【GitHub Copilot】最大限活用するためのTips紹介（Qiita）](https://qiita.com/hirokiii27/items/3d0696333df3af3ac8e9)

### 自動レビューの設定
組織やリポジトリレベルで、すべてのPRに対して自動的にCopilotレビューを有効化することも可能です。詳細は「[Configuring automatic code review by Copilot](https://docs.github.com/ja/copilot/how-tos/agents/copilot-code-review/automatic-code-review)」を参照してください。

## Copilotに指示を与える方法

### 方法1: PR本文に直接指示を記述

**最も確実な方法**として、PR本文に直接Copilot向けの指示を含めることができます（[SIOS Tech. Lab記事](https://tech-lab.sios.jp/archives/47820)参照）。

```markdown
## 変更内容
このPRでは、ユーザー認証機能を実装しました。

<!-- for GitHub Copilot review rule -->
以下の観点でレビューしてください：
- セキュリティの脆弱性がないか確認
- パフォーマンスの問題がないか確認
- 日本語でコメントを記載
<!-- for GitHub Copilot review rule -->
```

**HTMLコメント形式のメリット**:
- 人間のレビュアーには表示されない
- Copilotは認識して指示に従う

### 方法2: `.github/copilot-instructions.md`ファイルの作成

リポジトリ全体に適用される基本的なレビュールールを設定できます（[公式ドキュメント](https://docs.github.com/ja/copilot/how-tos/use-copilot-agents/request-a-code-review/use-code-review#customizing-copilots-reviews-with-custom-instructions)）。

**ファイルパス**: `.github/copilot-instructions.md`

**記述例**:
```markdown
# Copilot Code Review Instructions

When performing a code review:
- レビューコメントは日本語で記載してください
- セキュリティの観点を最優先で確認してください
- 可読性とメンテナンス性を重視してください
- ネストした三項演算子の使用は避けるよう提案してください
- 社内のコーディング規約に準拠しているか確認してください
```

**注意事項**:
- このファイルはリポジトリ全体のレビューに影響
- 個別のPRでの即時反映は保証されない（[SIOS Tech. Lab](https://tech-lab.sios.jp/archives/47820)）
- Copilot Chatでも同じファイルが使用される

**関連ドキュメント：**
- [GitHub Copilotのリポジトリカスタム命令を追加する](https://docs.github.com/ja/copilot/customizing-copilot/adding-repository-custom-instructions-for-github-copilot)

### 方法3: PRテンプレートの活用

一貫したレビュー指示を提供するため、PRテンプレートを使用できます（[SIOS Tech. Lab](https://tech-lab.sios.jp/archives/47820)推奨）。

**ファイルパス**: `.github/pull_request_template.md`

**テンプレート例**:
```markdown
## 概要
<!-- PRの概要を記載 -->

## 変更内容
<!-- 主な変更点を箇条書きで記載 -->

## テスト
<!-- 実施したテストについて記載 -->

<!-- Copilot Review Instructions -->
<!-- 
以下の観点でレビューをお願いします：
1. コードの品質とベストプラクティスの遵守
2. 潜在的なバグやエッジケースの検出
3. パフォーマンスの最適化の余地
4. セキュリティ上の懸念事項
-->
```

## 再レビューを依頼する方法

### 手動での再レビューリクエスト

[公式ドキュメント](https://docs.github.com/ja/copilot/how-tos/use-copilot-agents/request-a-code-review/use-code-review#requesting-a-re-review-from-copilot)に基づく手順：

1. **変更をプッシュ**
   - Copilotは自動的に再レビューを行わない

2. **再レビューをリクエスト**
   - Reviewersセクションで、Copilotの名前の横にある再リクエストボタンをクリック
   - または、Copilotを一度削除してから再度追加

**参考記事：**
- [Pull Requestレビューをリクエストする](https://docs.github.com/ja/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review)

### 再レビュー時の注意点
- 以前のコメントが解決済みでも、同じ指摘が繰り返される可能性がある
- 👎ボタンでフィードバックしても、次回のレビューには反映されない場合がある

## 制限事項と注意点

### ❌ できないこと

1. **PR上でのメンション機能**
   - `@copilot`のようなメンションで直接指示を出すことはできない（公式ドキュメントで未記載）
   - コメントへの返信でCopilotを再起動することもできない

2. **承認や変更リクエスト**
   - Copilotは常に「Comment」レビューのみ（[公式ドキュメント](https://docs.github.com/ja/copilot/how-tos/use-copilot-agents/request-a-code-review/use-code-review)）
   - 必須レビューの承認者としてカウントされない

3. **リアルタイムの対話**
   - PR上でのチャット形式での連続的なやり取りは不可
   - 各レビューは独立したセッション

### ⚠️ 注意すべき点

1. **コードの最終責任**
   - Copilotの提案は必ず人間がレビューする
   - 自動的に適用される変更はない

2. **機密情報の取り扱い**
   - PRに機密情報を含めない
   - 指示にも機密情報を含めない

3. **レビューの品質**
   - Copilotは完璧ではない
   - 重要な問題を見逃す可能性もある

## ベストプラクティス

### 効果的な活用のために

1. **明確な指示を与える**
   ```markdown
   <!-- Copilot向け指示 -->
   - TypeScriptの型安全性を確認
   - async/awaitの適切な使用を確認
   - エラーハンドリングの実装を確認
   ```

2. **段階的なレビュー**
   - まずCopilotでの自動レビュー
   - その後、人間のレビュアーによる詳細確認

3. **フィードバックの活用**
   - 有用なコメントには👍
   - 不適切なコメントには👎とその理由（[公式ドキュメント](https://docs.github.com/ja/copilot/how-tos/use-copilot-agents/request-a-code-review/use-code-review#providing-feedback-on-copilots-reviews)）

4. **定期的な指示の見直し**
   - プロジェクトの成長に合わせて`copilot-instructions.md`を更新
   - チームのフィードバックを反映

**参考になる実践例：**
- [GitHub Copilot導入後、初めて使う時（Qiita）](https://qiita.com/masakinihirota/items/0e58a6b921e4420a2882)

## トラブルシューティング

### よくある問題と解決方法

| 問題 | 原因 | 解決方法 |
|------|------|----------|
| Copilotが表示されない | 権限不足またはプラン未加入 | 組織の管理者に確認 |
| レビューが開始されない | 一時的な問題 | Copilotを削除して再追加 |
| 日本語で表示されない | 指示が不明確 | PR本文に明示的に日本語指定 |
| 同じコメントの繰り返し | 仕様上の制限 | 解決済みとマークして無視 |

**サポート情報：**
- [GitHub Copilotサポート](https://support.anthropic.com)
- [GitHubコミュニティフォーラム](https://github.com/community)

## 今後の展望

GitHub Copilotは急速に進化しており、以下の機能が将来的に追加される可能性があります：

- PR上でのインタラクティブな対話機能（[Qiita記事](https://qiita.com/hirokiii27/items/3d0696333df3af3ac8e9)で言及）
- メンション機能のサポート
- より高度なコンテキスト理解
- チーム固有のレビュー基準の学習

**最新情報の確認先：**
- [GitHub公式ドキュメント](https://docs.github.com/ja/copilot)
- [GitHubブログ（日本語）](https://github.blog/jp/)
- [GitHub Copilot新機能（コーディングエージェント）](https://github.blog/jp/2025-05-20-github-copilot-meet-the-new-coding-agent/)

---

**最終更新日**: 2025年7月31日  
**ドキュメントバージョン**: 1.1
