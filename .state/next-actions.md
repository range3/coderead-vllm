# 次にやるべきこと

## 最優先

- [ ] Phase 1 を開始: 垂直スライスの追跡（reading-guide.mdのユーザー優先度に基づきスライスを選択）
  - 候補1: リクエスト受信→Prefill→KVキャッシュ書込→Decode→応答返却のフルパス
  - 候補2: 画像付きリクエスト→マルチモーダル処理→推論のフルパス
  - ユーザーにスライス選択を確認する

## 次点

- [ ] KV Transfer / LMCache のコンポーネントsummary.md作成（Phase 2で詳細調査）
- [ ] マルチモーダル（`vllm/multimodal/`）のコンポーネントsummary.md作成
- [ ] プラグインシステム（`vllm/plugins/`）の仕組み把握

## いつかやる

- [ ] C++/CUDAカーネル（`csrc/`）のAPI一覧整理
- [ ] 分散推論（Tensor/Pipeline並列）の仕組み把握
- [ ] Speculative Decodingの実装詳細
