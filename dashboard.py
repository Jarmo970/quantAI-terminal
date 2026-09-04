import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Page Configuration
st.set_page_config(page_title="QuantAI Terminal Pro", page_icon="📈", layout="wide")

# Custom CSS for Professional Styling & Custom Timer Font
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #f3f4f6;
    }

    code, pre, [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }

    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        background: linear-gradient(135deg, rgba(17, 24, 39, 0.8) 0%, rgba(11, 15, 25, 0.9) 100%);
        border: 1px solid rgba(55, 65, 81, 0.4);
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        margin-bottom: 1rem;
        transition: all 0.3s ease-in-out;
    }

    div[data-testid="stVerticalBlock"] > div[style*="border"]:hover {
        border-color: rgba(59, 130, 246, 0.6);
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.15);
    }

    [data-testid="stMetric"] {
        background-color: rgba(31, 41, 55, 0.4);
        border: 1px solid rgba(75, 85, 99, 0.2);
        padding: 10px 15px;
        border-radius: 10px;
    }

    /* Custom Styling for the Live Ticking Clock */
    .live-clock-container {
        background-color: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(59, 130, 246, 0.3);
        padding: 10px 16px;
        border-radius: 10px;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.95rem;
        font-weight: 500;
        color: #60a5fa;
        text-align: right;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    </style>
""", unsafe_allow_html=True)


# Helper function to check US Stock Market Status in Eastern Time
def get_market_status():
    et_now = datetime.now(ZoneInfo("America/New_York"))
    weekday = et_now.weekday()  # 0 = Monday, 6 = Sunday
    current_time = et_now.time()

    if weekday >= 5:
        return "🔴 MARKET CLOSED (Weekend)", False

    market_open = datetime.strptime("09:30:00", "%H:%M:%S").time()
    market_close = datetime.strptime("16:00:00", "%H:%M:%S").time()
    pre_market_start = datetime.strptime("04:00:00", "%H:%M:%S").time()
    after_hours_end = datetime.strptime("20:00:00", "%H:%M:%S").time()

    if market_open <= current_time <= market_close:
        return "🟢 MARKET OPEN (Regular Trading)", True
    elif pre_market_start <= current_time < market_open:
        return "🟡 PRE-MARKET SESSION", False
    elif market_close < current_time <= after_hours_end:
        return "🟠 AFTER-HOURS SESSION", False
    else:
        return "🔴 MARKET CLOSED", False


# 2. Dynamically Fetch All S&P 500 Tickers from Wikipedia
@st.cache_data
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df_tickers = tables[0]
        tickers = df_tickers['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        return sorted(tickers)
    except Exception:
        return ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "NFLX", "AMD"]


all_stocks = get_sp500_tickers()

# 3. Sidebar Controls & Search Multi-Select
st.sidebar.markdown("### ⚙️ Terminal Settings")
years_back = st.sidebar.slider("Training Horizon (Years)", min_value=1, max_value=10, value=5)
start_date = pd.to_datetime(date.today() - timedelta(days=365 * years_back))
end_date = pd.to_datetime(date.today())

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Asset Filter")
default_selection = ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN"]
selected_stocks = st.sidebar.multiselect("Select Assets to Analyze", all_stocks, default=default_selection)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Live Terminal:** Ticking clock runs locally every second. Data pulls reliably via Stooq backend engine.")

# 4. Main Header Section & Live Market Status Banner
st.title("⚡ QuantAI Market Terminal")
status_text, is_market_open = get_market_status()

col_banner1, col_banner2 = st.columns([2, 1])
with col_banner1:
    st.markdown(f"**Market Status:** {status_text}")

with col_banner2:
    # Isolated 1-Second Ticking Clock Fragment with custom styling class
    @st.fragment(run_every=1)
    def live_clock():
        et_time_str = datetime.now(ZoneInfo("America/New_York")).strftime("%B %d, %Y - %I:%M:%S %p ET")
        st.markdown(f'<div class="live-clock-container">🕒 <b>ET:</b> {et_time_str}</div>', unsafe_allow_html=True)


    live_clock()

st.markdown("---")


# Robust Stooq Data Loader (Bypasses cloud IP blocking completely)
@st.cache_data(ttl=3600)
def load_stock_data(ticker_symbol, start, end):
    try:
        # Format symbol for Stooq (e.g. AAPL -> aapl.us)
        clean_symbol = ticker_symbol.replace('-', '').lower()
        stooq_url = f"https://stooq.com/q/d/l/?s={clean_symbol}.us&i=d"

        df = pd.read_csv(stooq_url)
        if df is not None and not df.empty and 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)

            # Ensure proper numeric columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Filter strictly by user-selected date range
            df = df.loc[start:end]
            return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker_symbol}: {e}")

    return pd.DataFrame()


# 5. Live Dashboard Render Fragment (Auto-refreshes every 60 seconds only if market is open)
refresh_interval = 60 if is_market_open else None


@st.fragment(run_every=refresh_interval)
def render_market_terminal():
    if not selected_stocks:
        st.warning("⚠️ Please select at least one stock ticker from the sidebar filter.")
        return

    for stock in selected_stocks:
        with st.container(border=True):
            st.subheader(f"📊 {stock}")

            stock_df = load_stock_data(stock, start_date, end_date)

            if stock_df is None or stock_df.empty or len(stock_df) < 30:
                st.warning(f"⚠️ Insufficient market data available for {stock}.")
                continue

            stock_df['SMA_20'] = stock_df['Close'].rolling(window=20).mean()
            stock_df['SMA_50'] = stock_df['Close'].rolling(window=50).mean()
            stock_df['Price_Change'] = stock_df['Close'].pct_change()
            stock_df['Volume_Change'] = stock_df['Volume'].pct_change()

            stock_df['Target'] = (stock_df['Close'].shift(-1) > stock_df['Close']).astype(int)
            ml_df = stock_df.dropna()

            if ml_df is None or len(ml_df) < 15:
                st.warning(f"⚠️ Not enough clean records for ML training on {stock}.")
                continue

            features = ['SMA_20', 'SMA_50', 'Price_Change', 'Volume_Change']
            X = ml_df[features]
            y = ml_df['Target']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            latest_features = X.iloc[[-1]]
            prediction = model.predict(latest_features)[0]
            prediction_proba = model.predict_proba(latest_features)[0][1] * 100

            col_sig, col_met1, col_met2, col_met3 = st.columns([1.3, 1, 1, 1])

            with col_sig:
                confidence = prediction_proba if prediction == 1 else (100 - prediction_proba)
                if prediction == 1:
                    st.success(f"**AI SIGNAL: BUY**\n\nConfidence: `{confidence:.1f}%`")
                else:
                    st.warning(f"**AI SIGNAL: HOLD**\n\nConfidence: `{confidence:.1f}%`")

            latest_close = float(stock_df['Close'].iloc[-1])
            prev_close = float(stock_df['Close'].iloc[-2])
            daily_pct = ((latest_close - prev_close) / prev_close) * 100
            period_return = ((latest_close - float(stock_df['Close'].iloc[0])) / float(stock_df['Close'].iloc[0])) * 100

            col_met1.metric("Close Price", f"${latest_close:.2f}", f"{daily_pct:.2f}%")
            col_met2.metric(f"{years_back}-Yr Return", f"{period_return:.2f}%")
            col_met3.metric("20 / 50 SMA",
                            f"${float(stock_df['SMA_20'].iloc[-1]):.2f} / ${float(stock_df['SMA_50'].iloc[-1]):.2f}")

            with st.expander(f"📈 Expand {stock} Price Action & Technical Chart"):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Close'], mode='lines', name='Close Price',
                                         line=dict(color='#60a5fa', width=2)))
                fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['SMA_20'], mode='lines', name='20-Day SMA',
                                         line=dict(color='#fbbf24', width=1.5)))
                fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['SMA_50'], mode='lines', name='50-Day SMA',
                                         line=dict(color='#f87171', width=1.5)))

                fig.update_layout(
                    xaxis_title="",
                    yaxis_title="Price (USD)",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=380,
                    margin=dict(l=10, r=10, t=10, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)


# Execute the live terminal block
render_market_terminal()