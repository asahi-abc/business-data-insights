# Business Data Insights App

📊 **Streamlitを用いたビジネスデータ分析ダッシュボード**

このプロジェクトは、売上・広告費・顧客データをもとに、データの可視化や機械学習を用いた売上予測を行うStreamlitアプリです。

## 📝 入力データ形式

### CSVファイルのカラム構成

アプリケーションにアップロードするCSVファイルには、以下のカラムが必要です：

#### 必須カラム
- `date`: 日付データ（例：2024-01-01）
- `sales`: 売上データ（数値）
- `advertising`: 広告費データ（数値）
- `customers`: 顧客数データ（数値）

#### オプショナルカラム
- `season_factor`: 季節要因データ（数値）

### サンプルデータ
```csv
date,sales,advertising,customers,season_factor
2024-01-01,120000,3000,50,250
2024-01-02,135000,3500,55,245
2024-01-03,115000,2800,45,240
...
```

### 注意事項
- ファイルがアップロードされない場合は、サンプルデータが自動生成されます
- データは日次データであることが想定されています
- 季節性分析には少なくとも2年分（24ヶ月）のデータが推奨されます
- 日付は`YYYY-MM-DD`形式で入力してください

## 🏆 主な機能

### 📊 データ概要
- **データアップロード:** CSVファイルをアップロードし、データを可視化
- **統計情報の表示:** データの基本統計量や分布を簡単に確認
- **相関分析:** 数値データ間の相関関係をヒートマップで表示

### 📈 データ可視化
- **時系列チャート:** 日別売上の推移を表示
- **散布図:** 広告費・顧客数と売上の関係を可視化
- **月別集計:** 月次での売上データの集計と表示

### 🤖 機械学習モデル
- **複数のモデル選択:**
  - 線形回帰
  - Ridge回帰
  - Lasso回帰
  - ランダムフォレスト
  - 勾配ブースティング
- **モデル評価:** MSE, RMSE, MAE, R²などの評価指標を表示
- **クロスバリデーション:** モデルの安定性を評価
- **SHAP値による説明可能性:** モデルの予測根拠を可視化

### 🔮 予測シミュレーション
- **インタラクティブ予測:** 広告費と顧客数を入力して売上を予測
- **信頼区間の表示:** 予測値の不確実性を表示
- **感度分析:** 入力値の変化に対する売上予測の変化を可視化

### 🔍 高度な分析
- **季節性分析:** 時系列データの季節性を分解して表示
- **異常値検知:** IsolationForestを使用した異常値の検出と可視化
- **時系列分解:** トレンド、季節性、残差成分の分析

### ⏱️ 時系列予測
- **ARIMAモデル:** 時系列データの特性を考慮した予測
- **自己相関分析:** ACFとPACFによる時系列特性の分析
- **予測期間の設定:** 柔軟な予測期間の選択

## 🚀 デモ

ローカル環境でアプリを実行できます。

### ローカル環境で実行する

1. **リポジトリをクローン**
    ```bash
    git clone git@github.com:asahi-abc/business-data-insights.git
    cd business-data-insights
    ```

2. **必要なライブラリをインストール**
    ```bash
    pip install -r requirements.txt
    ```

3. **Streamlit アプリを起動**
    ```bash
    streamlit run app.py
    ```

## 📂 ファイル構成
```
📂 business-data-insights
 ├── app.py              # メインのStreamlitアプリ
 ├── requirements.txt    # 必要なライブラリ
 ├── README.md           # 本ファイル
 └── models/            # 保存されたモデル（実行時に生成）
```

## 🛠 使用技術

- **Python**
- **Streamlit**
- **Pandas, NumPy**
- **Plotly, Matplotlib, Seaborn**
- **Scikit-learn**
- **Statsmodels**
- **SHAP**
- **Joblib**

## 🎯 今後の改善点

- より高度な時系列モデル（Prophet, LSTMなど）の実装
- モデルの自動選択機能の追加
- データの前処理オプションの拡充
- ユーザ認証機能の追加
- 予測結果のエクスポート機能の追加

