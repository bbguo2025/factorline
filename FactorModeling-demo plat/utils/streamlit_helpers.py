import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time # Added for initialize_session_state

def initialize_session_state():
    """Centralized session state initialization"""
    defaults = {
        'data_uploaded': False,
        'features_computed': False,
        'portfolio_built': False,
        'last_activity': time.time(),
        'performance_metrics': {},
        'uploaded_data_info': None,
        'upload_tab_active': False,
        'current_table_actions': {},
        'file_validation_cache': {},
        'df': None  # Initialize df as None
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def create_progress_ui():
    """Create consistent progress UI components"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    return progress_bar, status_text

def update_progress(progress_bar, status_text, progress: int, message: str):
    """Update progress with consistent messaging"""
    progress_bar.progress(progress)
    status_text.text(message)

def create_upload_progress_ui():
    """Create consistent upload progress UI components"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    return progress_bar, status_text

def update_upload_progress(progress_bar, status_text, progress: int, message: str):
    """Update upload progress with consistent messaging"""
    progress_bar.progress(progress)
    status_text.text(message)

def validate_data_format(df):
    """
    Validate the uploaded data format and determine if it's wide or long format
    """
    try:
        # Check for required columns
        if 'date' not in df.columns:
            return False, "Data must contain a 'date' column", None

        # Try to parse dates
        try:
            pd.to_datetime(df['date'])
        except:
            return False, "Date column cannot be parsed as datetime", None

        # Check if it's long format (has symbol and price columns)
        if 'symbol' in df.columns and any(col in df.columns for col in ['close', 'price']):
            # Long format validation
            required_long_cols = ['date', 'symbol']
            price_cols = [col for col in ['close', 'price', 'adjClose'] if col in df.columns]

            if not price_cols:
                return False, "Long format data must contain at least one price column (close, price, or adjClose)", None

            return True, f"Valid long format data with {len(df)} records and {df['symbol'].nunique()} unique symbols", "Long Format"

        # Check if it's wide format (date + multiple symbol columns)
        elif len(df.columns) > 2:
            # Wide format validation
            symbol_cols = [col for col in df.columns if col != 'date']

            if len(symbol_cols) < 1:
                return False, "Wide format data must contain at least one symbol column", None

            # Check if the data is numeric (except for date column)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < len(symbol_cols) * 0.8:  # At least 80% should be numeric
                return False, "Wide format data should be mostly numeric (price data)", None

            return True, f"Valid wide format data with {len(df)} records and {len(symbol_cols)} symbols", "Wide Format"

        else:
            return False, "Unable to determine data format. Please check your data structure.", None

    except Exception as e:
        return False, f"Error validating data: {str(e)}", None

def create_sample_data(sample_type="Russell 3000 Sample"):
    """
    Create sample data for demonstration purposes
    """
    try:
        if sample_type == "Russell 3000 Sample":
            # Create sample data similar to Russell 3000
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'V', 
                      'WMT', 'PG', 'UNH', 'HD', 'MA', 'DIS', 'PYPL', 'ADBE', 'NFLX', 'CRM']
            start_date = datetime(2020, 1, 1)
            end_date = datetime(2025, 6, 16)

        elif sample_type == "Tech Stocks Sample":
            # Tech-focused sample
            symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'ADBE', 'CRM',
                      'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM', 'AVGO', 'TXN', 'MU', 'LRCX', 'KLAC']
            start_date = datetime(2020, 1, 1)
            end_date = datetime(2025, 6, 16)

        else:  # Random Walk Sample
            symbols = [f"STOCK_{i:03d}" for i in range(1, 51)]  # 50 synthetic stocks
            start_date = datetime(2020, 1, 1)
            end_date = datetime(2025, 6, 16)

        # Generate date range (business days only)
        date_range = pd.bdate_range(start=start_date, end=end_date)

        data_list = []

        for symbol in symbols:
            # Set seed for reproducible data per symbol
            np.random.seed(hash(symbol) % 10000)

            # Generate realistic price data with some correlation structure
            n_days = len(date_range)

            if sample_type == "Random Walk Sample":
                # Pure random walk
                returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
                initial_price = np.random.uniform(20, 200)
            else:
                # More realistic returns with some momentum and mean reversion
                returns = np.random.normal(0.0008, 0.015, n_days)  # Slightly positive drift
                returns += 0.1 * np.random.normal(0, 0.01, n_days)  # Add some noise
                initial_price = np.random.uniform(50, 500)

            # Generate price series
            log_prices = np.log(initial_price) + np.cumsum(returns)
            prices = np.exp(log_prices)

            # Add some realistic OHLV data
            for i, date in enumerate(date_range):
                close_price = prices[i]

                # Generate OHLV with some realistic constraints
                daily_vol = abs(returns[i]) * 2  # Intraday volatility
                high = close_price * (1 + daily_vol * np.random.uniform(0.2, 0.8))
                low = close_price * (1 - daily_vol * np.random.uniform(0.2, 0.8))
                open_price = close_price * (1 + np.random.uniform(-daily_vol/2, daily_vol/2))

                # Ensure OHLC constraints
                high = max(high, open_price, close_price)
                low = min(low, open_price, close_price)

                # Generate volume (higher volume on higher volatility days)
                base_volume = np.random.uniform(100000, 2000000)
                volume = int(base_volume * (1 + abs(returns[i]) * 10))

                data_list.append({
                    'date': date,
                    'symbol': symbol,
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close_price, 2),
                    'adjClose': round(close_price, 2),  # Assume no adjustments for sample data
                    'volume': volume,
                    'unadjustedVolume': volume,
                    'change': round(close_price - open_price, 2),
                    'changePercent': round((close_price - open_price) / open_price * 100, 2),
                    'vwap': round((high + low + close_price) / 3, 2),  # Simplified VWAP
                    'changeOverTime': round(returns[i], 6)
                })

        sample_df = pd.DataFrame(data_list)
        sample_df = sample_df.sort_values(['symbol', 'date']).reset_index(drop=True)

        return sample_df

    except Exception as e:
        st.error(f"Error creating sample data: {str(e)}")
        return None

def format_number(num, format_type="default"):
    """
    Format numbers for display in the UI
    """
    try:
        if pd.isna(num):
            return "N/A"

        if format_type == "percentage":
            return f"{num:.2%}"
        elif format_type == "currency":
            return f"${num:,.2f}"
        elif format_type == "basis_points":
            return f"{num * 10000:.0f} bps"
        elif format_type == "large_number":
            if abs(num) >= 1e9:
                return f"{num/1e9:.2f}B"
            elif abs(num) >= 1e6:
                return f"{num/1e6:.2f}M"
            elif abs(num) >= 1e3:
                return f"{num/1e3:.2f}K"
            else:
                return f"{num:.2f}"
        else:  # default
            if abs(num) < 0.01:
                return f"{num:.4f}"
            else:
                return f"{num:.2f}"

    except Exception:
        return str(num)

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """
    Create a custom metric card with styling
    """
    delta_html = ""
    if delta is not None:
        # Try to interpret delta as a number for color logic
        delta_numeric = None
        if isinstance(delta, (int, float)):
            delta_numeric = delta
            delta_display = f"{delta_numeric:+.2f}%" if abs(delta_numeric) < 100 else f"{delta_numeric:+.2f}"
        else:
            # Attempt to strip %, commas, etc.
            try:
                delta_numeric = float(str(delta).replace("%", "").replace(",", ""))
            except ValueError:
                pass
            delta_display = str(delta)

        # Determine color
        if delta_color == "normal" and delta_numeric is not None:
            color = "green" if delta_numeric > 0 else "red" if delta_numeric < 0 else "inherit"
        else:
            color = delta_color

        delta_html = f'<div style="color: {color}; font-size: 0.8rem;">Δ {delta_display}</div>'

    card_html = f"""
    <div style="
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    ">
        <div style="font-size: 0.9rem; color: #666; margin-bottom: 0.5rem;">{title}</div>
        <div style="font-size: 1.5rem; font-weight: bold; color: #333;">{value}</div>
        {delta_html}
    </div>
    """
    return card_html

def add_sidebar_info(title, content):
    """
    Add informational content to sidebar
    """
    with st.sidebar.expander(f"ℹ️ {title}"):
        st.markdown(content)

def show_data_quality_summary(df):
    """
    Display a comprehensive data quality summary
    """
    st.subheader("📊 Data Quality Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Records", f"{len(df):,}")

    with col2:
        if 'symbol' in df.columns:
            st.metric("Unique Symbols", f"{df['symbol'].nunique():,}")
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Columns", len(numeric_cols))

    with col3:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing Data %", f"{missing_pct:.1f}%")

    with col4:
        if 'date' in df.columns:
            date_range = pd.to_datetime(df['date']).max() - pd.to_datetime(df['date']).min()
            st.metric("Date Range", f"{date_range.days} days")
        else:
            st.metric("Columns", len(df.columns))

def create_download_link(df, filename, link_text):
    """
    Create a download link for a dataframe
    """
    csv = df.to_csv(index=False)
    st.download_button(
        label=link_text,
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

@st.cache_data
def load_and_cache_data(file_path):
    """
    Load and cache data for better performance
    """
    try:
        return pd.read_csv(file_path, parse_dates=['date'])
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None
