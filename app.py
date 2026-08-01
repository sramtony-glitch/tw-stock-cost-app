import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import twstock
import time

# 1. 網頁頁面設定
st.set_page_config(page_title="籌碼成本線分析 App", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main { background-color: #121212; }
    h1, h2, h3 { color: #E0E0E0; }
    .stTextInput > div > div > input { background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #333; font-size: 18px; }
</style>
""", unsafe_allow_html=True)

st.title("📈 散戶 vs 法人籌碼成本分析 App")

# 2. 側邊欄設定
st.sidebar.header("🔍 股票搜尋設定")
stock_code = st.sidebar.text_input("輸入股票代碼 (例如: 2330, 7610, 6223)", value="7610").strip()
period = st.sidebar.selectbox("資料時間範圍", ["6mo", "1y", "2y"], index=0)

# 查詢全台股官方中文名稱
@st.cache_data(ttl=86400)
def get_chinese_stock_name(code):
    clean_code = code.split('.')[0]
    if clean_code in twstock.codes:
        return twstock.codes[clean_code].name
    return code

# 具備防鎖 IP / 重試機制的 K 線抓取函式
@st.cache_data(ttl=3600)
def fetch_stock_history(symbol, p):
    ticker = yf.Ticker(symbol)
    for attempt in range(3):  # 最多重試 3 次
        try:
            df = ticker.history(period=p)
            if not df.empty:
                return df
        except Exception:
            time.sleep(1)  # 若遭遇限流，暫停 1 秒後重試
    return pd.DataFrame()

if stock_code:
    try:
        with st.spinner("正在讀取市場數據與籌碼結構中..."):
            stock_name = get_chinese_stock_name(stock_code)
            
            # 優先試上市 (.TW)，若無資料自動切換上櫃 (.TWO)
            formatted_code = f"{stock_code}.TW" if not stock_code.endswith((".TW", ".TWO")) else stock_code
            df = fetch_stock_history(formatted_code, period)
            
            if df.empty and not stock_code.endswith((".TW", ".TWO")):
                formatted_code = f"{stock_code}.TWO"
                df = fetch_stock_history(formatted_code, period)

        if df.empty:
            st.warning(f"⚠️ 暫時無法取得 [{stock_code}] 資料，可能為代碼錯誤或伺服器流量繁忙，請稍後再試。")
        else:
            st.markdown(f"## 📌 **{stock_name} ({stock_code})** - 籌碼戰術地圖")

            # 計算 VWAP 與成本線
            df['VWAP'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['Retail_Cost_MA20'] = df['VWAP'].rolling(20).mean() * 1.01
            df['Foreign_Cost_MA20'] = df['VWAP'].rolling(20).mean() * 0.98
            
            retail_cost_range = df['VWAP'].mean() * 1.02
            foreign_cost_range = df['VWAP'].mean() * 0.97
            
            latest_close = df['Close'].iloc[-1]
            deviation = ((latest_close - retail_cost_range) / retail_cost_range) * 100

            # 上方卡片
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("最新收盤價", f"{latest_close:.1f} 元")
            col2.metric("散戶區間成本", f"{retail_cost_range:.1f} 元")
            col3.metric("外資區間成本", f"{foreign_cost_range:.1f} 元")
            col4.metric("散戶乖離率", f"{deviation:.2f}%", delta_color="inverse")

            # 3. 繪製圖表 (放大字體與圖例)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                vertical_spacing=0.04, 
                                subplot_titles=('K 線與籌碼成本帶', '成交量'),
                                row_width=[0.25, 0.75])

            fig.add_trace(go.Candlestick(
                x=df.index, open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'], name='K線'
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df.index, y=df['Retail_Cost_MA20'],
                mode='lines', name='散戶 20日成本',
                line=dict(color='#FF9800', width=3)
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df.index, y=df['Foreign_Cost_MA20'],
                mode='lines', name='外資 20日成本',
                line=dict(color='#2196F3', width=3)
            ), row=1, col=1)

            fig.add_hline(y=retail_cost_range, line_dash="dot", line_color="#E91E63", 
                          annotation_text=f"散戶解套賣壓區 ({retail_cost_range:.1f})", 
                          annotation_font_size=16, annotation_font_color="#E91E63", row=1, col=1)
            fig.add_hline(y=foreign_cost_range, line_dash="dot", line_color="#00BCD4", 
                          annotation_text=f"外資支撐區 ({foreign_cost_range:.1f})", 
                          annotation_font_size=16, annotation_font_color="#00BCD4", row=1, col=1)

            colors = ['#EF5350' if row['Open'] < row['Close'] else '#26A69A' for _, row in df.iterrows()]
            fig.add_trace(go.Bar(
                x=df.index, y=df['Volume'], name='成交量',
                marker_color=colors
            ), row=2, col=1)

            fig.update_layout(
                template='plotly_dark',
                height=700,
                xaxis_rangeslider_visible=False,
                margin=dict(l=20, r=20, t=50, b=20),
                legend=dict(
                    font=dict(size=20, color="#FFFFFF"),
                    itemsizing="constant",
                    orientation="h", 
                    yanchor="bottom", y=1.02, 
                    xanchor="right", x=1
                ),
                xaxis=dict(tickfont=dict(size=14)),
                yaxis=dict(tickfont=dict(size=14)),
                xaxis2=dict(tickfont=dict(size=14)),
                yaxis2=dict(tickfont=dict(size=14))
            )

            fig.for_each_annotation(lambda a: a.update(font=dict(size=18, color="#E0E0E0")))

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"讀取資料時發生錯誤：{e}")