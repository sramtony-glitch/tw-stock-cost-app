import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import twstock
import requests
import datetime
import time

# 1. 網頁頁面基本設定
st.set_page_config(page_title="籌碼成本線分析 App (實戰優化版)", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main { background-color: #121212; }
    h1, h2, h3 { color: #E0E0E0; }
    .stTextInput > div > div > input { background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #333; font-size: 18px; }
</style>
""", unsafe_allow_html=True)

st.title("📈 散戶 vs 法人籌碼成本分析 App (含滾動模式與無融資警示)")

# FinMind API Token 設定
FINMIND_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoic3JhbXRvbnlAZ21haWwuY29tIiw11haWwOiJzcmFtdG9ueUBnbWFpbC5jb20iLCJ0b2tlmn_2ZXJzaW9uIjo18LWvBBvbrpnVzWKlOu2ZNIkshuwTnLKRQpBmcMPQ"

# 2. 側邊欄設定
st.sidebar.header("🔍 股票搜尋設定")
stock_code = st.sidebar.text_input("輸入股票代碼 (例: 7610, 2330, 2317)", value="7610").strip()
period_days = st.sidebar.selectbox("資料時間範圍", [60, 120, 180], index=1)

# 💡 優化 1：加入滾動籌碼天數切換開關
st.sidebar.header("⚙️ 演算法參數優化")
calc_mode = st.sidebar.radio(
    "籌碼成本計算模式",
    ["全區間累計 (含斷頭重置)", "20日滾動洗淨 (適合飆股/短線)", "60日滾動洗淨 (適合中線)"],
    index=1  # 預設使用 20 日滾動，防止飆股成本線卡在底部
)

@st.cache_data(ttl=86400)
def get_chinese_stock_name(code):
    clean_code = code.split('.')[0]
    if clean_code in twstock.codes:
        return twstock.codes[clean_code].name
    return code

# Step 0: 歷史 K 線與真實 VWAP
@st.cache_data(ttl=3600)
def fetch_stock_history(symbol, days):
    ticker = yf.Ticker(symbol)
    period_str = f"{days}d"
    for attempt in range(3):
        try:
            df = ticker.history(period=period_str)
            if not df.empty:
                df.index = df.index.tz_localize(None)
                df['VWAP'] = (df['High'] + df['Low'] + df['Close']) / 3
                return df
        except Exception:
            time.sleep(1)
    return pd.DataFrame()

# 自 FinMind 官方資料庫抓取真實籌碼數據
@st.cache_data(ttl=21600)
def fetch_finmind_real_chip_data(stock_id, days):
    clean_code = stock_id.split('.')[0]
    start_date = (datetime.date.today() - datetime.timedelta(days=days*2)).strftime("%Y-%m-%d")
    
    headers = {"Authorization": f"Bearer {FINMIND_TOKEN}"} if FINMIND_TOKEN else {}

    url_margin = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockMarginPurchaseShortSale&data_id={clean_code}&start_date={start_date}"
    url_inst = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInstitutionalInvestorsBuySell&data_id={clean_code}&start_date={start_date}"
    url_dt = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockDayTrading&data_id={clean_code}&start_date={start_date}"

    try:
        res_m = requests.get(url_margin, headers=headers).json().get('data', [])
        res_i = requests.get(url_inst, headers=headers).json().get('data', [])
        res_d = requests.get(url_dt, headers=headers).json().get('data', [])

        df_m = pd.DataFrame(res_m)
        df_i = pd.DataFrame(res_i)
        df_d = pd.DataFrame(res_d)

        if df_m.empty:
            return pd.DataFrame()

        df_m['date'] = pd.to_datetime(df_m['date'])
        df_m = df_m.set_index('date')
        df_m['margin_balance'] = df_m['MarginPurchaseTodayBalance'].astype(float)
        df_m['margin_delta'] = df_m['MarginPurchaseBuy'].astype(float) - df_m['MarginPurchaseSell'].astype(float)

        if not df_i.empty:
            df_i['date'] = pd.to_datetime(df_i['date'])
            df_i_grouped = df_i.groupby('date')['buy'].sum() - df_i.groupby('date')['sell'].sum()
            df_m['institutional_buy_sell'] = (df_i_grouped / 1000.0).reindex(df_m.index).fillna(0)
        else:
            df_m['institutional_buy_sell'] = 0.0

        if not df_d.empty:
            df_d['date'] = pd.to_datetime(df_d['date'])
            df_d = df_d.set_index('date')
            df_m['daytrade_vol'] = (df_d['Volume'].astype(float) / 1000.0).reindex(df_m.index).fillna(0)
        else:
            df_m['daytrade_vol'] = 0.0

        return df_m[['margin_balance', 'margin_delta', 'institutional_buy_sell', 'daytrade_vol']]
    except Exception:
        return pd.DataFrame()

# 升級版 7 步驟洗淨演算法 (支援 Rolling 模式)
def process_spec_chip_algorithm_optimized(df_price, df_chip, mode):
    df_price.index = pd.to_datetime(df_price.index.date)
    
    if not df_chip.empty:
        df_chip.index = pd.to_datetime(df_chip.index.date)
        df = df_price.join(df_chip, how='left').fillna(0)
    else:
        df = df_price.copy()
        df['margin_balance'] = 0.0
        df['margin_delta'] = 0.0
        df['institutional_buy_sell'] = 0.0
        df['daytrade_vol'] = 0.0

    # Step 1 & 2: 當沖剔除與有效殘差
    df['daytrade_est'] = df['daytrade_vol'] / 2.0
    df['vol_shares'] = df['Volume'] / 1000.0
    df['raw_retail_vol'] = df['vol_shares'] - df['institutional_buy_sell']
    df['eff_retail_vol'] = (df['raw_retail_vol'] - df['daytrade_est']).clip(lower=0)

    # 檢查是否完全無融資數據（無融資標記）
    has_no_margin = (df['margin_balance'].sum() == 0)

    # Step 3: 純度驗證
    df['purity'] = np.where(df['eff_retail_vol'] > 0, df['margin_delta'] / df['eff_retail_vol'], 0)
    
    conditions = [
        (df['purity'] >= 0.30),
        (df['purity'] >= 0.10) & (df['purity'] < 0.30),
        (df['purity'] < 0.10)
    ]
    choices = ['HIGH', 'MEDIUM', 'LOW']
    df['quality_flag'] = np.select(conditions, choices, default='LOW')

    # Step 4: 真實融資斷頭偵測
    df['prev_margin_balance'] = df['margin_balance'].shift(1).fillna(df['margin_balance'])
    df['margin_drop_pct'] = np.where(df['prev_margin_balance'] > 0, df['margin_delta'] / df['prev_margin_balance'], 0)
    df['blowout_day'] = df['margin_drop_pct'] <= -0.03

    # Step 5 & 6: 依據模式計算洗淨成本
    retail_costs = []
    
    if "20日滾動" in mode or "60日滾動" in mode:
        # 滾動洗淨成本 (Rolling Inventory Cost)
        window = 20 if "20日滾動" in mode else 60
        df['cum_amount'] = (df['VWAP'] * df['eff_retail_vol']).rolling(window, min_periods=1).sum()
        df['cum_vol'] = df['eff_retail_vol'].rolling(window, min_periods=1).sum()
        df['Retail_Cost_Spec'] = np.where(df['cum_vol'] > 0, df['cum_amount'] / df['cum_vol'], df['VWAP'])
    else:
        # 全區間存貨加權法
        current_cum_amount = 0.0
        current_cum_vol = 0.0
        for idx, row in df.iterrows():
            vwap = row['VWAP']
            net_retail_vol = row['eff_retail_vol']
            is_blowout = row['blowout_day']

            if is_blowout:
                current_cum_amount = vwap * net_retail_vol
                current_cum_vol = net_retail_vol
            else:
                if net_retail_vol >= 0:
                    current_cum_amount += vwap * net_retail_vol
                    current_cum_vol += net_retail_vol
                else:
                    current_cost = current_cum_amount / current_cum_vol if current_cum_vol > 0 else vwap
                    current_cum_vol = max(0, current_cum_vol + net_retail_vol)
                    current_cum_amount = current_cost * current_cum_vol

            cost = current_cum_amount / current_cum_vol if current_cum_vol > 0 else vwap
            retail_costs.append(cost)
        df['Retail_Cost_Spec'] = retail_costs

    # Step 7: 滾動外資與指標輸出
    df['Foreign_Cost_Spec'] = df['VWAP'].rolling(20).mean() * 0.97
    
    latest_close = df['Close'].iloc[-1]
    latest_retail_cost = df['Retail_Cost_Spec'].iloc[-1]
    
    df['deviation_pct'] = ((latest_close - latest_retail_cost) / latest_retail_cost) * 100.0
    df['is_underwater'] = latest_close < latest_retail_cost

    return df, has_no_margin

if stock_code:
    try:
        with st.spinner("正在連線 FinMind 資料庫並套用洗淨成本模型..."):
            stock_name = get_chinese_stock_name(stock_code)
            formatted_code = f"{stock_code}.TW" if not stock_code.endswith((".TW", ".TWO")) else stock_code
            
            df_price = fetch_stock_history(formatted_code, period_days)
            if df_price.empty and not stock_code.endswith((".TW", ".TWO")):
                formatted_code = f"{stock_code}.TWO"
                df_price = fetch_stock_history(formatted_code, period_days)

        if df_price.empty:
            st.warning(f"⚠️ 暫時無法取得 [{stock_code}] 資料，請確認代碼。")
        else:
            df_chip = fetch_finmind_real_chip_data(stock_code, period_days)
            df, has_no_margin = process_spec_chip_algorithm_optimized(df_price, df_chip, calc_mode)

            st.markdown(f"## 📌 **{stock_name} ({stock_code})** - 籌碼戰術地圖")

            # 💡 優化 2：無融資個股自動提示
            if has_no_margin:
                st.warning("⚠️ **無融資數據警示**：本股票屬於創新板/興櫃或無融資標的。紫線代表「主力與非外资混合籌碼成本」，判讀時請參考滾動模式。")

            latest_close = df['Close'].iloc[-1]
            latest_retail_cost = df['Retail_Cost_Spec'].iloc[-1]
            latest_foreign_cost = df['Foreign_Cost_Spec'].iloc[-1]
            latest_deviation = df['deviation_pct'].iloc[-1]
            is_underwater = df['is_underwater'].iloc[-1]

            # 籌碼線名稱
            cost_line_name = "非外資/主力成本" if has_no_margin else "散戶洗淨成本"

            # 1. 卡片欄位
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("最新收盤價", f"{latest_close:.1f} 元")
            col2.metric(f"{cost_line_name} (紫線)", f"{latest_retail_cost:.1f} 元")
            col3.metric("外資預估成本 (藍線)", f"{latest_foreign_cost:.1f} 元")
            col4.metric("籌碼狀態", "⚠️ 成本下方(套牢)" if is_underwater else "🟢 成本上方(獲利)", f"{latest_deviation:.2f}%")

            # 2. 繪製圖表
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.04, 
                                subplot_titles=(f'K 線與 {cost_line_name} (模式: {calc_mode})', '成交量 (漲紅 / 跌綠)'),
                                row_width=[0.25, 0.75])

            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='K線',
                increasing_line_color='#FF5252', decreasing_line_color='#00E676'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df.index, y=df['Retail_Cost_Spec'],
                mode='lines', name=cost_line_name,
                line=dict(color='#E040FB', width=3)
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df.index, y=df['Foreign_Cost_Spec'],
                mode='lines', name='外資預估成本線',
                line=dict(color='#00E5FF', width=3)
            ), row=1, col=1)

            # 標記真實融資斷頭日 (如有)
            if not has_no_margin:
                blowout_days = df[df['blowout_day']]
                for b_date, b_row in blowout_days.iterrows():
                    fig.add_vline(x=b_date, line_dash="dash", line_color="#FF1744", line_width=1.5, row=1, col=1)
                    fig.add_annotation(x=b_date, y=b_row['High'], text="⚡Step4 真實融資斷頭",
                                       showarrow=True, arrowhead=1, arrowcolor="#FF1744",
                                       font=dict(color="#FF1744", size=12), row=1, col=1)

            # 成交量
            volume_colors = ['#FF5252' if row['Close'] >= row['Open'] else '#00E676' for _, row in df.iterrows()]
            fig.add_trace(go.Bar(
                x=df.index, y=df['Volume'], name='成交量',
                marker_color=volume_colors
            ), row=2, col=1)

            fig.update_layout(
                template='plotly_dark',
                height=720,
                xaxis_rangeslider_visible=False,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(
                    font=dict(size=16, color="#FFFFFF"),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                )
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"讀取資料時發生錯誤：{e}")