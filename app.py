import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
# 日本語フォントのサポート（インポートするだけで日本語フォントが有効になる）
import japanize_matplotlib  # noqa: F401
from sklearn.ensemble import IsolationForest
import seaborn as sns
import joblib
import os
import datetime
import io
import zipfile
import shap
# Prophetモデル用
from prophet import Prophet

# タイトルと概要の表示
st.set_page_config(page_title="ビジネスデータ分析ダッシュボード", layout="wide")
st.title("ビジネスデータ分析ダッシュボード")
st.markdown("このアプリは、ビジネスデータの可視化と機械学習による売上予測モデルを構築するデモです。")

# サイドバーでファイルアップロード
st.sidebar.header("データのアップロード")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロードしてください", type=["csv"])

# データ読み込みまたはサンプルデータ生成
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    # 日付列がある場合は日付型に変換
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
    st.success("アップロードされたデータを使用しています。")
else:
    st.info("ファイルがアップロードされていないため、サンプルデータを使用します。")
    # サンプルの時系列データ生成
    date_rng = pd.date_range(start='2021-01-01', end='2022-12-31', freq='D')
    data = pd.DataFrame(date_rng, columns=['date'])

    # トレンド成分
    trend = np.linspace(100, 300, len(date_rng))

    # 季節性成分（年間の季節性）
    yearly_seasonality = 50 * \
        np.sin(2 * np.pi * np.arange(len(date_rng)) / 365)

    # 週次の季節性（週末に売上増加）
    weekly_seasonality = 30 * (data['date'].dt.dayofweek >= 5).astype(int)

    # ランダム成分
    noise = np.random.normal(0, 20, size=len(date_rng))

    # 売上データの生成
    data['sales'] = trend + yearly_seasonality + weekly_seasonality + noise
    data['sales'] = data['sales'].clip(lower=50)  # 最低売上を設定

    # 異常値を追加
    anomaly_indices = np.random.choice(len(date_rng), 10, replace=False)
    data.loc[anomaly_indices, 'sales'] = data.loc[anomaly_indices, 'sales'] * 2

    # 広告費と顧客数の生成（売上と相関を持たせる）
    base_advertising = np.random.randint(1000, 5000, size=len(date_rng))
    data['advertising'] = base_advertising + data['sales'] * 2 \
        + np.random.normal(0, 500, size=len(date_rng))

    base_customers = np.random.randint(20, 100, size=len(date_rng))
    data['customers'] = base_customers + data['sales'] * 0.1 \
        + np.random.normal(0, 10, size=len(date_rng))

    # 季節性を持つデータを追加
    data['season_factor'] = yearly_seasonality + 200

    # 日付列の型変換
    data['date'] = pd.to_datetime(data['date'])

# データフレームの表示前に日付列を文字列に変換（PyArrowの互換性問題を解決）


def prepare_df_for_display(df):
    """
    Streamlitで表示するためにデータフレームを準備します。
    特に日付列をPyArrowと互換性のある形式に変換します。

    Parameters:
    -----------
    df : pandas.DataFrame
        表示するデータフレーム

    Returns:
    --------
    pandas.DataFrame
        表示用に処理されたデータフレームのコピー
    """
    df_display = df.copy()
    for col in df_display.columns:
        # 日付型の列を文字列に変換
        if pd.api.types.is_datetime64_any_dtype(df_display[col]):
            df_display[col] = df_display[col].dt.strftime('%Y-%m-%d')
    return df_display


# タブを作成
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 データ概要",
    "📈 データ可視化",
    "🤖 機械学習モデル",
    "🔮 予測シミュレーション",
    "🔍 高度な分析",
    "⏱️ 時系列予測"
])

with tab1:
    # データ概要の表示
    st.header("データの概要")
    st.write(prepare_df_for_display(data.head()))

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("統計情報")
        st.write(data.describe())

    with col2:
        st.subheader("データ型情報")
        # データ型情報をテキスト形式で表示
        data_types = pd.DataFrame({
            'データ型': data.dtypes.astype(str),
            '欠損値数': data.isnull().sum(),
            '欠損率(%)': (data.isnull().sum() / len(data) * 100).round(2)
        })
        st.write(data_types)

    # 相関行列のヒートマップ
    st.subheader("相関行列")
    numeric_data = data.select_dtypes(include=[np.number])
    corr = numeric_data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

with tab2:
    # データ可視化
    st.header("データ可視化")

    # 時系列チャート（日付を文字列に変換）
    plot_data = prepare_df_for_display(data.copy())
    st.subheader("時系列チャート：日別売上")
    fig = px.line(plot_data, x='date', y='sales', title='日別売上の推移')
    fig.update_layout(xaxis_title='日付', yaxis_title='売上')
    st.plotly_chart(fig, use_container_width=True)

    # 散布図：広告費と売上の関係
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("散布図：広告費と売上の関係")
        scatter1 = px.scatter(
            data, x='advertising', y='sales',
            trendline='ols',
            title='広告費と売上の関係'
        )
        scatter1.update_layout(xaxis_title='広告費', yaxis_title='売上')
        st.plotly_chart(scatter1, use_container_width=True)

    with col2:
        st.subheader("散布図：顧客数と売上の関係")
        scatter2 = px.scatter(
            data, x='customers', y='sales',
            trendline='ols',
            title='顧客数と売上の関係'
        )
        scatter2.update_layout(xaxis_title='顧客数', yaxis_title='売上')
        st.plotly_chart(scatter2, use_container_width=True)

    # 月別の集計データ
    st.subheader("月別売上集計")
    data['month'] = data['date'].dt.month
    monthly_data = data.groupby('month')['sales'].agg(
        ['mean', 'min', 'max', 'sum'])
    monthly_data.columns = ['平均売上', '最小売上', '最大売上', '合計売上']

    fig = px.bar(monthly_data, y='合計売上', title='月別合計売上')
    fig.update_layout(xaxis_title='月', yaxis_title='合計売上')
    st.plotly_chart(fig, use_container_width=True)

    # 月次データを表示する前に日付を文字列に変換
    st.write("月次データの表示:")
    st.write(prepare_df_for_display(monthly_data))

with tab3:
    # 機械学習による売上予測
    st.header("機械学習: 売上予測モデル")
    st.markdown(
        "**広告費** と **顧客数** 、**季節性（オプション）** を説明変数とし、\
        **売上** を目的変数とした回帰モデルを構築します。")

    # 特徴量とターゲットの選択
    feature_cols = ['advertising', 'customers']
    if 'season_factor' in data.columns:
        feature_cols.append('season_factor')

    features = data[feature_cols]
    target = data['sales']

    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42)

    # 特徴量のスケーリング
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # モデル選択
    st.subheader("モデル選択")
    model_option = st.selectbox(
        "使用するモデルを選択してください",
        ["線形回帰", "Ridge回帰", "Lasso回帰", "ランダムフォレスト", "勾配ブースティング"]
    )

    # ハイパーパラメータ設定
    col1, col2 = st.columns(2)

    with col1:
        if model_option in ["Ridge回帰", "Lasso回帰"]:
            alpha = st.slider("正則化パラメータ (alpha)", 0.01, 10.0, 1.0, 0.01)

        if model_option in ["ランダムフォレスト", "勾配ブースティング"]:
            n_estimators = st.slider("決定木の数", 10, 200, 100, 10)
            max_depth = st.slider("決定木の最大深さ", 1, 20, 5, 1)

    # モデルのインスタンス化
    if model_option == "線形回帰":
        model = LinearRegression()
    elif model_option == "Ridge回帰":
        model = Ridge(alpha=alpha)
    elif model_option == "Lasso回帰":
        model = Lasso(alpha=alpha)
    elif model_option == "ランダムフォレスト":
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
    elif model_option == "勾配ブースティング":
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

    # モデルの学習
    with st.spinner('モデルを学習中...'):
        model.fit(X_train_scaled, y_train)

    # クロスバリデーション
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, cv=5,
        scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)

    # テストデータでの予測と評価
    y_pred = model.predict(X_test_scaled)

    # 評価指標の計算
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 評価指標の表示
    st.subheader("モデル評価指標")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("平均二乗誤差 (MSE)", f"{mse:.2f}")
    col2.metric("平方根平均二乗誤差 (RMSE)", f"{rmse:.2f}")
    col3.metric("平均絶対誤差 (MAE)", f"{mae:.2f}")
    col4.metric("決定係数 (R²)", f"{r2:.4f}")
    col5.metric("クロスバリデーション RMSE",
                f"{cv_rmse.mean():.2f} ± {cv_rmse.std():.2f}")

    # 予測結果と実際の売上の比較
    st.subheader("予測結果と実際の売上の比較")
    result_df = pd.DataFrame({
        '実際の売上': y_test,
        '予測された売上': y_pred
    })

    # 予測結果の散布図
    fig = px.scatter(result_df, x='実際の売上', y='予測された売上',
                     title='実際の売上 vs 予測された売上')
    fig.add_trace(
        go.Scatter(x=[min(y_test), max(y_test)],
                   y=[min(y_test), max(y_test)],
                   mode='lines',
                   name='完全一致線',
                   line=dict(color='red', dash='dash'))
    )
    st.plotly_chart(fig, use_container_width=True)

    # 特徴量の重要度（ツリーベースのモデルの場合）
    if model_option in ["ランダムフォレスト", "勾配ブースティング"]:
        st.subheader("特徴量の重要度")
        importance_df = pd.DataFrame({
            '特徴量': features.columns,
            '重要度': model.feature_importances_
        }).sort_values('重要度', ascending=False)
        st.write(prepare_df_for_display(importance_df))

        fig = px.bar(importance_df, x='特徴量', y='重要度',
                     title='特徴量の重要度')
        st.plotly_chart(fig, use_container_width=True)

    # 線形モデルの場合は係数を表示
    if model_option in ["線形回帰", "Ridge回帰", "Lasso回帰"]:
        st.subheader("モデル係数")
        coef_df = pd.DataFrame({
            '特徴量': features.columns,
            '係数': model.coef_
        })
        st.write(prepare_df_for_display(coef_df))

        fig = px.bar(coef_df, x='特徴量', y='係数',
                     title='モデル係数')
        st.plotly_chart(fig, use_container_width=True)

    # モデルとスケーラーの保存
    if st.button("モデルを保存"):
        # モデル保存用のディレクトリを作成
        if not os.path.exists('models'):
            os.makedirs('models')

        # 現在の日時を含むファイル名を生成
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"sales_prediction_{model_option}_{now}"
        model_filename = f"models/{model_name}.joblib"
        scaler_filename = f"models/scaler_{model_name}.joblib"

        # モデルとスケーラーを保存
        joblib.dump(model, model_filename)
        joblib.dump(scaler, scaler_filename)

        st.success(f"モデルとスケーラーを保存しました: {model_filename}")

        # ZIPファイルを作成してモデルとスケーラーを格納
        zip_filename = f"models/model_and_scaler_{model_name}.zip"

        # メモリ上でZIPファイルを作成
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            # モデルファイルを追加
            with open(model_filename, 'rb') as model_file:
                zf.writestr(os.path.basename(
                    model_filename), model_file.read())
            # スケーラーファイルを追加
            with open(scaler_filename, 'rb') as scaler_file:
                zf.writestr(os.path.basename(
                    scaler_filename), scaler_file.read())

        # ポインタをファイルの先頭に戻す
        zip_buffer.seek(0)

        # ダウンロードボタンを表示
        st.download_button(
            label="モデルとスケーラーをダウンロード",
            data=zip_buffer,
            file_name=f"model_and_scaler_{model_name}.zip",
            mime="application/zip"
        )

    # SHAP値による説明可能性の追加
    if model_option in ["ランダムフォレスト", "勾配ブースティング",
                        "線形回帰", "Ridge回帰", "Lasso回帰"]:
        st.subheader("モデルの説明可能性（SHAP値）")

        try:
            # サンプルデータの選択
            st.write("SHAP値を計算するためのサンプルデータを選択")
            sample_size = min(100, len(X_test))
            X_sample = X_test_scaled[:sample_size]

            with st.spinner('SHAP値を計算中...'):
                # SHAP値の計算
                explainer = shap.Explainer(model, X_train_scaled)
                shap_values = explainer(X_sample)

                # SHAP値のサマリープロット
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(
                    shap_values,
                    X_sample,
                    feature_names=features.columns,
                    show=False
                )
                plt.title("特徴量の重要度（SHAP値）")
                st.pyplot(fig)

                # 特定のデータポイントに対する詳細な説明
                st.subheader("個別予測の説明")

                # ランダムなデータポイントを選択
                sample_idx = np.random.randint(0, sample_size)

                # 特定のデータポイントに対するSHAP値の可視化
                fig, ax = plt.subplots(figsize=(10, 4))
                shap.plots.waterfall(shap_values[sample_idx], show=False)
                plt.title("個別予測の説明（SHAP値）")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"SHAP値の計算中にエラーが発生しました: {e}")
            st.info("モデルやデータによってはSHAP値の計算ができない場合があります。")

with tab4:
    # インタラクティブな予測シミュレーション
    st.header("インタラクティブ予測シミュレーション")

    col1, col2 = st.columns(2)

    with col1:
        advertising_input = st.number_input(
            "広告費を入力してください",
            min_value=0,
            value=3000,
            step=100
        )

        customers_input = st.number_input(
            "顧客数を入力してください",
            min_value=0,
            value=50,
            step=1
        )

        if 'season_factor' in data.columns:
            season_input = st.slider(
                "季節要因 (オプション)",
                float(data['season_factor'].min()),
                float(data['season_factor'].max()),
                float(data['season_factor'].mean())
            )
            input_data = {
                'advertising': [advertising_input],
                'customers': [customers_input],
                'season_factor': [season_input]
            }
        else:
            input_data = {
                'advertising': [advertising_input],
                'customers': [customers_input]
            }

        input_df = pd.DataFrame(input_data)

        # 入力データをスケーリング
        input_scaled = scaler.transform(input_df)

        # 予測
        predicted_sales = model.predict(input_scaled)[0]

    with col2:
        st.subheader("予測結果")
        st.metric(
            "予測された売上",
            f"{int(predicted_sales):,} 円",
            delta=int(predicted_sales - data['sales'].mean())
        )

        # 信頼区間の計算（ランダムフォレストの場合）
        if model_option == "ランダムフォレスト":
            predictions = []
            for estimator in model.estimators_:
                predictions.append(estimator.predict(input_scaled)[0])

            lower_bound = np.percentile(predictions, 2.5)
            upper_bound = np.percentile(predictions, 97.5)

            st.write(
                f"95%信頼区間: {int(lower_bound):,} 円 〜 {int(upper_bound):,} 円")

    # 感度分析
    st.subheader("感度分析")
    st.write("広告費または顧客数を変化させた場合の売上予測の変化を確認できます。")

    analysis_type = st.radio(
        "分析する変数を選択してください",
        ["広告費", "顧客数"]
    )

    if analysis_type == "広告費":
        # 広告費の範囲を設定
        min_adv = max(0, advertising_input - 2000)
        max_adv = advertising_input + 2000
        adv_range = np.linspace(min_adv, max_adv, 20)

        # 各広告費に対する予測を計算
        sensitivity_data = []
        for adv in adv_range:
            if 'season_factor' in data.columns:
                test_input = pd.DataFrame({
                    'advertising': [adv],
                    'customers': [customers_input],
                    'season_factor': [season_input]
                })
            else:
                test_input = pd.DataFrame({
                    'advertising': [adv],
                    'customers': [customers_input]
                })

            test_scaled = scaler.transform(test_input)
            pred = model.predict(test_scaled)[0]
            sensitivity_data.append({'広告費': adv, '予測売上': pred})

        sensitivity_df = pd.DataFrame(sensitivity_data)

        fig = px.line(sensitivity_df, x='広告費', y='予測売上',
                      title='広告費の変化に対する売上予測の感度分析')
        fig.add_vline(x=advertising_input, line_dash="dash",
                      annotation_text="現在の値")
        st.plotly_chart(fig, use_container_width=True)

    else:  # 顧客数の感度分析
        # 顧客数の範囲を設定
        min_cust = max(0, customers_input - 30)
        max_cust = customers_input + 30
        cust_range = np.linspace(min_cust, max_cust, 20)

        # 各顧客数に対する予測を計算
        sensitivity_data = []
        for cust in cust_range:
            if 'season_factor' in data.columns:
                test_input = pd.DataFrame({
                    'advertising': [advertising_input],
                    'customers': [cust],
                    'season_factor': [season_input]
                })
            else:
                test_input = pd.DataFrame({
                    'advertising': [advertising_input],
                    'customers': [cust]
                })

            test_scaled = scaler.transform(test_input)
            pred = model.predict(test_scaled)[0]
            sensitivity_data.append({'顧客数': cust, '予測売上': pred})

        sensitivity_df = pd.DataFrame(sensitivity_data)

        fig = px.line(sensitivity_df, x='顧客数', y='予測売上',
                      title='顧客数の変化に対する売上予測の感度分析')
        fig.add_vline(x=customers_input, line_dash="dash",
                      annotation_text="現在の値")
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    # 高度な分析
    st.header("高度な分析")

    # 季節性分析
    st.subheader("時系列データの季節性分析")

    # 日付でソート
    ts_data = data.sort_values('date')

    # 日次データを月次データに集計
    monthly_data = ts_data.set_index('date').resample('ME')[
        'sales'].mean().reset_index()

    # 季節分解
    st.write("月次データの表示:")
    st.write(prepare_df_for_display(monthly_data))

    # 季節性分解のためのデータ生成（2年分のデータを生成）
    if len(monthly_data) < 24:  # 2年分のデータが必要
        st.warning("季節性分解には少なくとも2年分（24ヶ月）のデータが必要です。サンプルデータを拡張します。")

        # 既存のデータを複製して2年分に拡張
        if len(monthly_data) > 0:
            last_date = monthly_data['date'].max()
            months_to_add = 24 - len(monthly_data)

            new_dates = [
                last_date + pd.DateOffset(months=i+1)
                for i in range(months_to_add)
            ]
            new_sales = np.random.normal(
                monthly_data['sales'].mean(),
                monthly_data['sales'].std(),
                months_to_add
            )

            extension_df = pd.DataFrame({
                'date': new_dates,
                'sales': new_sales
            })

            monthly_data = pd.concat(
                [monthly_data, extension_df]).reset_index(drop=True)
            st.write("拡張されたデータ:")
            st.write(prepare_df_for_display(monthly_data))

    # 季節分解を実行
    try:
        monthly_series = monthly_data.set_index('date')['sales']
        decomposition = seasonal_decompose(
            monthly_series, model='additive', period=12)

        # 結果をプロット
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
        decomposition.observed.plot(ax=ax1)
        ax1.set_title('観測データ')
        decomposition.trend.plot(ax=ax2)
        ax2.set_title('トレンド成分')
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_title('季節成分')
        decomposition.resid.plot(ax=ax4)
        ax4.set_title('残差成分')
        plt.tight_layout()
        st.pyplot(fig)

        # 季節性パターンの詳細分析
        st.subheader("季節性パターンの詳細")
        seasonal_pattern = pd.DataFrame({
            '月': range(1, 13),
            '季節性効果': decomposition.seasonal.iloc[:12].values
        })

        fig = px.bar(seasonal_pattern, x='月', y='季節性効果',
                     title='月別の季節性効果')
        st.plotly_chart(fig, use_container_width=True,
                        key="seasonal_pattern_plot")

    except Exception as e:
        st.error(f"季節性分解中にエラーが発生しました: {e}")

    # 異常検知
    st.subheader("異常値検知")

    # IsolationForestを使用した異常検知
    st.write("IsolationForestを使用した異常値検知:")

    # 異常検知のための特徴量を選択
    anomaly_features = data[['sales', 'advertising', 'customers']]

    # モデルの設定
    contamination = st.slider(
        "異常値の予想割合",
        min_value=0.01,
        max_value=0.2,
        value=0.05,
        step=0.01
    )

    # 異常検知の実行
    isolation_forest = IsolationForest(
        contamination=contamination,
        random_state=42
    )

    data['anomaly'] = isolation_forest.fit_predict(anomaly_features)
    data['anomaly_score'] = isolation_forest.decision_function(
        anomaly_features)

    # 異常値を-1、正常値を1としてラベル付け
    anomalies = data[data['anomaly'] == -1]

    # 異常値の可視化
    # 日付列を文字列に変換してからプロットに使用
    plot_data = prepare_df_for_display(data.copy())
    fig = px.scatter(plot_data, x='date', y='sales',
                     color='anomaly',
                     color_discrete_map={-1: 'red', 1: 'blue'},
                     title='売上の異常値検知')

    st.plotly_chart(fig, use_container_width=True, key="anomaly_plot")

    # 異常値の詳細
    if not anomalies.empty:
        st.write(f"検出された異常値: {len(anomalies)}件")
        # 表示する前に日付列を文字列に変換
        st.write(prepare_df_for_display(
            anomalies[['date', 'sales', 'advertising',
                       'customers', 'anomaly_score']]
        ))
    else:
        st.write("異常値は検出されませんでした。")

with tab6:
    # 時系列予測
    st.header("時系列予測分析")
    st.markdown("""
    時系列データの特性を分析し、将来の売上を予測します。
    この分析では、過去のパターンに基づいて将来のトレンドを予測します。
    """)

    # 日付でソート
    ts_data = data.sort_values('date')

    # 時系列データの可視化
    st.subheader("時系列データの可視化")

    # 日次データを表示（日付を文字列に変換）
    plot_ts_data = prepare_df_for_display(ts_data.copy())
    fig = px.line(plot_ts_data, x='date', y='sales', title='日次売上データ')
    st.plotly_chart(fig, use_container_width=True, key="daily_sales_plot")

    # 時系列の集計期間を選択
    agg_period = st.selectbox(
        "集計期間を選択",
        ["日次", "週次", "月次"]
    )

    if agg_period == "日次":
        ts_agg = ts_data.copy()
    elif agg_period == "週次":
        ts_agg = ts_data.set_index('date').resample(
            'W')['sales'].mean().reset_index()
    else:  # 月次
        ts_agg = ts_data.set_index('date').resample(
            'ME')['sales'].mean().reset_index()

    # 集計データの可視化（日付を文字列に変換）
    plot_ts_agg = prepare_df_for_display(ts_agg.copy())
    fig = px.line(plot_ts_agg, x='date', y='sales', title=f'{agg_period}売上データ')
    st.plotly_chart(fig, use_container_width=True, key="agg_sales_plot")

    # 自己相関と偏自己相関の分析
    st.subheader("自己相関分析")

    col1, col2 = st.columns(2)

    with col1:
        # 自己相関関数（ACF）
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_acf(ts_agg['sales'], ax=ax)
        plt.title("自己相関関数（ACF）")
        st.pyplot(fig)

    with col2:
        # 偏自己相関関数（PACF）
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_pacf(ts_agg['sales'], ax=ax)
        plt.title("偏自己相関関数（PACF）")
        st.pyplot(fig)

    # 予測モデルの選択
    st.subheader("時系列予測モデル")
    model_type = st.selectbox(
        "使用するモデルを選択してください",
        ["ARIMA", "SARIMA", "Prophet"]
    )

    # 予測期間の設定
    forecast_periods = st.slider("予測期間", 1, 30, 7)

    # モデル固有のパラメータ設定
    if model_type == "ARIMA":
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.slider("自己回帰次数 (p)", 0, 5, 1)
        with col2:
            d = st.slider("差分次数 (d)", 0, 2, 1)
        with col3:
            q = st.slider("移動平均次数 (q)", 0, 5, 1)

    elif model_type == "SARIMA":
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.slider("自己回帰次数 (p)", 0, 5, 1)
        with col2:
            d = st.slider("差分次数 (d)", 0, 2, 1)
        with col3:
            q = st.slider("移動平均次数 (q)", 0, 5, 1)

        st.markdown("季節性パラメータ:")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            P = st.slider("季節自己回帰次数 (P)", 0, 2, 1)
        with col2:
            D = st.slider("季節差分次数 (D)", 0, 1, 1)
        with col3:
            Q = st.slider("季節移動平均次数 (Q)", 0, 2, 1)
        with col4:
            m = st.slider("季節周期 (m)", 2, 12, 12)

    elif model_type == "Prophet":
        col1, col2 = st.columns(2)
        with col1:
            yearly_seasonality = st.selectbox("年次季節性", [True, False], index=0)
        with col2:
            weekly_seasonality = st.selectbox("週次季節性", [True, False], index=0)

        col1, col2 = st.columns(2)
        with col1:
            seasonality_mode = st.selectbox(
                "季節性モード", ["additive", "multiplicative"], index=0)
        with col2:
            changepoint_prior_scale = st.slider(
                "変化点の柔軟性", 0.001, 0.5, 0.05, step=0.001)

    # モデルの構築と予測
    try:
        if model_type == "ARIMA":
            with st.spinner('ARIMAモデルを構築中...'):
                # モデルの構築
                model = ARIMA(ts_agg['sales'], order=(p, d, q))
                model_fit = model.fit()

                # モデルの要約
                st.write("ARIMAモデルの要約:")
                st.text(str(model_fit.summary()))

                # 予測
                forecast = model_fit.forecast(steps=forecast_periods)

        elif model_type == "SARIMA":
            with st.spinner('SARIMAモデルを構築中...'):
                # モデルの構築
                model = SARIMAX(ts_agg['sales'],
                                order=(p, d, q),
                                seasonal_order=(P, D, Q, m))
                model_fit = model.fit(disp=False)

                # モデルの要約
                st.write("SARIMAモデルの要約:")
                st.text(str(model_fit.summary()))

                # 予測
                forecast = model_fit.forecast(steps=forecast_periods)

        elif model_type == "Prophet":
            with st.spinner('Prophetモデルを構築中...'):
                # Prophetのデータフォーマットに変換
                prophet_data = ts_agg.rename(
                    columns={'date': 'ds', 'sales': 'y'})

                # モデルの初期化と学習
                model = Prophet(
                    yearly_seasonality=yearly_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    seasonality_mode=seasonality_mode,
                    changepoint_prior_scale=changepoint_prior_scale
                )
                model.fit(prophet_data)

                # 将来のデータフレームを作成
                future = model.make_future_dataframe(
                    periods=forecast_periods, freq='D')

                # 予測
                forecast_result = model.predict(future)

                # Prophetコンポーネントのプロット
                fig_components = model.plot_components(forecast_result)
                st.write("Prophetモデルの季節性成分:")
                st.pyplot(fig_components)

                # 最後のforecast_periods分のデータを取得
                forecast = forecast_result.tail(
                    forecast_periods)['yhat'].values

        # 予測結果の可視化
        last_date = ts_agg['date'].iloc[-1]

        if agg_period == "日次":
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_periods
            )
        elif agg_period == "週次":
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(weeks=1),
                periods=forecast_periods,
                freq='W'
            )
        else:  # 月次
            forecast_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=forecast_periods,
                freq='ME'
            )

        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'sales': forecast
        })

        # 実績値と予測値の結合
        combined_df = pd.concat([
            ts_agg[['date', 'sales']].tail(30),
            forecast_df
        ])

        # 可視化（日付を文字列に変換）
        plot_combined_df = prepare_df_for_display(combined_df.copy())
        fig = px.line(plot_combined_df, x='date', y='sales',
                      title=f'{model_type}モデルによる売上予測')

        # 最後の実績日付を文字列に変換
        last_date_str = pd.to_datetime(last_date).strftime('%Y-%m-%d')
        max_date_str = pd.to_datetime(
            combined_df['date'].max()).strftime('%Y-%m-%d')

        fig.add_vrect(
            x0=last_date_str, x1=max_date_str,
            fillcolor="lightgray", opacity=0.3,
            layer="below", line_width=0,
            annotation_text="予測期間",
            annotation_position="top left"
        )
        st.plotly_chart(fig, use_container_width=True, key="forecast_plot")

        # 予測値の表示
        st.subheader("予測結果")
        st.write(prepare_df_for_display(forecast_df))

    except Exception as e:
        st.error(f"{model_type}予測中にエラーが発生しました: {e}")
        st.info("データやパラメータによってはモデルが収束しない場合があります。パラメータを調整してみてください。")

    # 時系列分解の可視化
    st.subheader("時系列分解")

    try:
        # 時系列分解
        decomposition = seasonal_decompose(ts_agg.set_index(
            'date')['sales'], model='additive', period=12)

        # 結果をプロット
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
        decomposition.observed.plot(ax=ax1)
        ax1.set_title('観測データ')
        decomposition.trend.plot(ax=ax2)
        ax2.set_title('トレンド成分')
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_title('季節成分')
        decomposition.resid.plot(ax=ax4)
        ax4.set_title('残差成分')
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.error(f"時系列分解中にエラーが発生しました: {e}")

# フッター
st.markdown("---")
st.markdown("© 2025 データサイエンスポートフォリオ | 作成者: Asahi Ito")
