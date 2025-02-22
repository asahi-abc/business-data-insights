import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# タイトルと概要の表示
st.title("ビジネスデータ分析ダッシュボード")
st.markdown("このアプリは、ビジネスデータの可視化と機械学習による売上予測モデルを構築するデモです。")

# サイドバーでファイルアップロード
st.sidebar.header("データのアップロード")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

# データ読み込みまたはサンプルデータ生成
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("アップロードされたデータを使用しています。")
else:
    st.info("ファイルがアップロードされていないため、サンプルデータを使用します。")
    # サンプルの時系列データ生成
    date_rng = pd.date_range(start='2021-01-01', end='2021-12-31', freq='D')
    data = pd.DataFrame(date_rng, columns=['date'])
    data['sales'] = np.random.randint(100, 500, size=(len(date_rng)))
    data['customers'] = np.random.randint(20, 100, size=(len(date_rng)))
    data['advertising'] = np.random.randint(1000, 5000, size=(len(date_rng)))

# 日付列の型変換（必要に応じて）
data['date'] = pd.to_datetime(data['date'])

# データ概要の表示
st.header("データの概要")
st.write(data.head())
st.subheader("統計情報")
st.write(data.describe())

# Plotlyを用いた時系列チャート
st.header("時系列チャート：日別売上")
fig = px.line(data, x='date', y='sales', title='日別売上の推移', labels={'sales': '売上', 'date': '日付'})
st.plotly_chart(fig, use_container_width=True)

# Altairを用いた散布図：広告費と売上の関係
st.header("散布図：広告費と売上の関係")
scatter = alt.Chart(data).mark_circle(size=60).encode(
    x='advertising',
    y='sales',
    tooltip=['date', 'sales', 'advertising']
).interactive()
st.altair_chart(scatter, use_container_width=True)

# 機械学習による売上予測
st.header("機械学習: 売上予測モデル")
st.markdown("**広告費** と **顧客数** を説明変数とし、**売上** を目的変数とした線形回帰モデルを構築します。")

# 特徴量とターゲットの選択
features = data[['advertising', 'customers']]
target = data['sales']

# データの分割
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# モデルの学習
model = LinearRegression()
model.fit(X_train, y_train)

# テストデータでの予測と評価
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write("モデルの平均二乗誤差 (MSE):", mse)

st.subheader("予測結果と実際の売上の比較")
result_df = pd.DataFrame({
    '実際の売上': y_test,
    '予測された売上': y_pred
})
st.write(result_df.head())

# インタラクティブな予測シミュレーション
st.header("インタラクティブ予測シミュレーション")
advertising_input = st.number_input("広告費を入力してください", min_value=0, value=3000, step=100)
customers_input = st.number_input("顧客数を入力してください", min_value=0, value=50, step=1)

input_df = pd.DataFrame({'advertising': [advertising_input], 'customers': [customers_input]})
predicted_sales = model.predict(input_df)[0]
st.write("予測された売上:", int(predicted_sales))

# まとめ
st.markdown("### まとめ")
st.markdown("このアプリでは、データの概要把握、可視化、統計解析、そして機械学習による売上予測まで、ビジネスデータに関するスキルを総合的に実演しました。")
