# Business Data Insights App

📊 **Streamlitを用いたビジネスデータ分析ダッシュボード**

このプロジェクトは、売上・広告費・顧客データをもとに、データの可視化や機械学習を用いた売上予測を行うStreamlitアプリです。 
ビジネスデータに関するスキルをアピールするポートフォリオとして最適です。

## 🏆 主な機能

- **データアップロード:** CSVファイルをアップロードし、データを可視化
- **統計情報の表示:** データの基本統計量や分布を簡単に確認
- **インタラクティブなグラフ:** Plotlyを用いた時系列チャートやAltairの散布図を提供
- **機械学習による売上予測:** 線形回帰を用いて広告費と顧客数から売上を予測
- **シミュレーション:** ユーザが入力した値に基づき売上を即時予測

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
```

## 🛠 使用技術

- **Python**
- **Streamlit**
- **Pandas, NumPy**
- **Plotly, Altair**
- **Scikit-learn**

## 🎯 今後の改善点

- より高度な予測モデル（時系列分析など）の実装
- ダッシュボードのUI強化
- ユーザ認証機能の追加

