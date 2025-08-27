"""
Factor Analysis Notebook
Streamlined workflow: Load data → Build custom factors → Configure simulation → Run analysis
Data already includes technical features - focus on factor combination and strategy testing
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, Union
import traceback
import re

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database_manager import DatabaseManager
from operations import *
from portfolio_constructer import (
    simulation_settings, daily_trade_list, daily_portfolio_returns, 
    metrics_calculator, get_rebalance_dates
)
from portfolio_analyzer import PortfolioAnalyzer
from error_handler import ErrorHandler, safe_operation
from performance_monitor import PerformanceMonitor
from utils.simulation_storage import save_simulation_record

from styles.design_system import DesignSystem
from utils.streamlit_helpers import initialize_session_state, create_progress_ui, update_progress

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def validate_custom_feature(custom_feature: pd.Series, df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate custom feature for simulation

    Returns:
        (is_valid, error_message)
    """
    if custom_feature is None:
        return False, "❌ Custom feature is None. Please check your factor code."

    if not isinstance(custom_feature, pd.Series):
        return False, "❌ Custom feature must be a pandas Series. Please check your factor code."

    if custom_feature.empty:
        return False, "❌ Custom feature is empty. Please check your factor code."

    # Check for non-unique MultiIndex
    if custom_feature.index.duplicated().any():
        return False, "❌ Custom feature has duplicate indices. This can cause 'non-unique multi-index' errors. Please check your data for duplicate (date, symbol) combinations."

    # Check if custom_feature has the same index as df
    if not custom_feature.index.equals(df.index):
        return False, f"❌ Custom feature index doesn't match data index. Expected {df.index.names}, got {custom_feature.index.names}. This usually happens when there are duplicate indices in your data."

    return True, "✅ Custom feature validation passed"

def create_execution_environment(df: pd.DataFrame) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Create execution environment with all necessary variables"""
    # Check for required columns
    if "log_return" not in df.columns and "return" not in df.columns:
        raise ValueError("Data must contain either 'log_return' or 'return' column")

    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()

    exec_globals = {
        **globals(),
        'df': df_clean,
        # Add all columns as individual variables for easier access
        **{
            col: df_clean[col]
            for col in df_clean.columns
        },
        # Add common data access patterns
        'returns': df_clean["log_return"] if "log_return" in df_clean.columns else df_clean["return"],
        'industry': df_clean["industry"] if "industry" in df_clean.columns else None,
        'subindustry': df_clean["subindustry"] if "subindustry" in df_clean.columns else None,
        # Add numpy and pandas for calculations
        'np': np,
        'pd': pd,
        # Add common functions
        'ts_decay': ts_decay,
        'group_neutralize': group_neutralize,
        'ts_sum': ts_sum
    }

    return exec_globals, df_clean

def execute_factor_code(feature_code: str, df: pd.DataFrame) -> Tuple[bool, Union[pd.Series, str]]:
    """
    Execute factor code safely

    Returns:
        (success, result_or_error_message)
    """
    try:
        exec_globals, df_clean = create_execution_environment(df)

        # Execute user's factor code
        exec(feature_code, exec_globals)

        if 'custom_feature' not in exec_globals:
            return False, "❌ Code must create a variable named 'custom_feature'"

        custom_feature = exec_globals['custom_feature']

        # Validate the custom_feature against the cleaned dataframe
        is_valid, error_msg = validate_custom_feature(custom_feature, df_clean)
        if not is_valid:
            return False, error_msg

        return True, custom_feature

    except Exception as e:
        error_type = type(e).__name__
        error_traceback = traceback.format_exc()
        line_number = None
        for line in error_traceback.split('\n'):
            if '<string>' in line:
                match = re.search(r'line (\d+)', line)
                if match:
                    line_number = match.group(1)
                    break

        error_message = f"❌ {error_type} at line {line_number}: {str(e)}" if line_number else f"❌ {error_type}: {str(e)}"
        return False, error_message



def handle_loading_error(error: Exception, estimated_rows: int, estimated_cols: int) -> None:
    """Enhanced error handling for loading operations"""
    error_msg = str(error).lower()

    if '502' in error_msg or 'timeout' in error_msg:
        st.error("❌ **Connection timeout (502 error)**")
        st.markdown(f"""
        **This happens with very large datasets:**

        🔧 **Solutions:**
        1. **Try again** - server may be temporarily overloaded
        2. **Use smaller dataset** - split your data into smaller tables
        3. **Close other applications** - free up memory
        4. **Wait 30 seconds** and try again

        📊 **Current dataset:** {estimated_rows:,} rows, {estimated_cols} columns
        """)
    else:
        st.error(f"❌ **Loading error:** {error}")
        if estimated_rows > 1000000:
            st.info("💡 For large datasets, try using the 'Chunked Load' button instead.")

def load_data_with_progress(db_manager: DatabaseManager, selected_table: str, 
                          loading_method: str = "standard") -> Tuple[bool, Union[pd.DataFrame, str]]:
    """
    Load data with progress tracking and error handling

    Returns:
        (success, result_or_error_message)
    """
    try:
        # Get table info to estimate size
        table_info = db_manager.get_table_info(selected_table)
        estimated_rows = table_info.get('row_count', 0)
        estimated_cols = len(table_info.get('columns', []))

        # Create progress tracking
        progress_bar, status_text = create_progress_ui()

        # Simple loading approach - use chunked loading for large datasets
        if loading_method == "chunked" or estimated_rows > 1000000:
            st.info(f"📊 Loading large dataset: {estimated_rows:,} rows × {estimated_cols} columns")

            def update_progress_callback(message):
                status_text.text(message)
                if "%" in message:
                    try:
                        pct = float(message.split("(")[1].split("%")[0])
                        progress_bar.progress(int(pct))
                    except:
                        pass

            df = db_manager.load_table_data_simple(selected_table, progress_callback=update_progress_callback)
        else:
            # Standard loading for smaller datasets
            status_text.text(f"Loading {estimated_rows:,} rows with {estimated_cols} columns...")
            progress_bar.progress(50)
            df = db_manager.load_table_data(selected_table)
            progress_bar.progress(90)

        if df.empty:
            return False, "❌ No data loaded - check your database connection"

        status_text.text("Processing data structure...")
        progress_bar.progress(95)

        # Convert to MultiIndex (date, symbol) format
        if 'date' in df.columns and 'symbol' in df.columns:
            # Ensure date is datetime
            if df['date'].dtype == 'object':
                df['date'] = pd.to_datetime(df['date'])

            df = df.set_index(['date', 'symbol']).sort_index()

            # Clean duplicate indices before returning
            if df.index.duplicated().any():
                duplicate_count = df.index.duplicated().sum()
                df = df[~df.index.duplicated(keep='first')]
                st.warning(f"⚠️ {duplicate_count:,} duplicate indices detected and removed (keeping first occurrence). This ensures your factor analysis will work correctly.")

            progress_bar.progress(100)
            status_text.text("Data loaded successfully!")

            return True, df
        else:
            return False, "Data must contain 'date' and 'symbol' columns"

    except Exception as e:
        return False, str(e)

def create_code_cell_styling():
    """Create enhanced code cell styling"""
    st.markdown("""
    <style>
    .code-cell-container {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        border-radius: 8px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .code-cell-header {
        background: #e9ecef;
        padding: 0.5rem 1rem;
        border-bottom: 1px solid #dee2e6;
        font-family: monospace;
        font-size: 0.9rem;
        color: #495057;
    }
    .stTextArea textarea {
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', 'Consolas', monospace;
        font-size: 14px;
        line-height: 1.4;
        background: #ffffff;
        border: none;
    }
    .factor-stats {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .factor-stats h4 {
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

def display_factor_statistics(custom_feature: pd.Series) -> None:
    """Display comprehensive factor statistics"""
    st.markdown('<div class="factor-stats">', unsafe_allow_html=True)
    st.markdown("**📋 Feature Statistics:**")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Mean", f"{custom_feature.mean():.4f}")
        st.metric("Std", f"{custom_feature.std():.4f}")
        st.metric("Coverage", f"{(~custom_feature.isna()).mean():.2%}")

    with col2:
        st.metric("Min", f"{custom_feature.min():.4f}")
        st.metric("Max", f"{custom_feature.max():.4f}")
        st.metric("Skewness", f"{custom_feature.skew():.4f}")

    with col3:
        st.metric("Data Type", str(custom_feature.dtype))
        st.metric("Shape", str(custom_feature.shape))
        st.metric("Index Levels", str(custom_feature.index.names))

    st.markdown('</div>', unsafe_allow_html=True)

def simulation(custom_feature: pd.Series, settings: Dict[str, Any], contributor: bool = True) -> Optional[Dict[str, Any]]:
    """Run portfolio simulation with settings - implementing your exact workflow"""
    try:
        # Get the returns data from session state
        if 'df' not in st.session_state or st.session_state.df is None:
            st.error("No data loaded. Please load data first.")
            return None

        df = st.session_state.df

        # Validate custom_feature
        is_valid, error_msg = validate_custom_feature(custom_feature, df)
        if not is_valid:
            st.error(error_msg)
            return None

        returns = df["log_return"] if "log_return" in df.columns else df["return"]

        # Step 1: Generate portfolio weights
        st.info("🔄 Step 1: Generating portfolio weights...")

        # Add progress indicator for monthly rebalancing
        if settings['rebalance_period'] == "monthly":
            progress_placeholder = st.empty()
            progress_placeholder.info(
                "🚀 Processing monthly rebalancing with optimized algorithm...")

        # Generate portfolio weights (without signal-based timeout)
        try:
            weights, counts = daily_trade_list(
                equal_weight=settings['equal_weight'],
                custom_feature=custom_feature,
                pct=settings['pct'],
                min_universe=settings['min_universe'],
                max_weight=settings['max_weight'],
                rebalance_period=settings['rebalance_period'])
        except Exception as e:
            st.error(f"❌ Portfolio weight generation failed: {str(e)}")
            st.info(
                "💡 Try reducing the dataset size or using daily rebalancing instead."
            )
            return None
        finally:
            # Clear progress indicator
            if settings['rebalance_period'] == "monthly":
                progress_placeholder.empty()

        # Step 2: Compute daily portfolio returns (log return, long/short breakdown, turnover)
        st.info("🔄 Step 2: Computing portfolio returns...")
        returns_df, top_longs, top_shorts = daily_portfolio_returns(
            weights, returns, settings, contributor=contributor)

        # Step 3: Analyze performance using PortfolioAnalyzer
        st.info("🔄 Step 3: Analyzing performance...")
        analyzer = PortfolioAnalyzer(returns_df)

        # Display results
        st.success("✅ Simulation completed successfully!")

        # Show performance visualization
        st.markdown("### 📈 Portfolio Performance Analysis")

        # Create matplotlib figure
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend for Streamlit

        # Generate the performance plot
        fig = analyzer.plot_full_performance(counts_df=counts)

        # Display in Streamlit
        if fig:
            st.pyplot(fig)

        # Performance Summary
        with st.expander("📊 Performance Summary", expanded=True):
            summary_dict = analyzer.summary()

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Risk & Return Metrics:**")
                for key in [
                        'Annualized Return', 'Yearly Volatility',
                        'Sharpe Ratio', 'Sortino Ratio'
                ]:
                    if key in summary_dict:
                        st.write(f"• {key}: {summary_dict[key]}")

            with col2:
                st.markdown("**Drawdown & Extremes:**")
                for key in [
                        'Max Drawdown', 'Max Daily Return', 'Min Daily Return'
                ]:
                    if key in summary_dict:
                        st.write(f"• {key}: {summary_dict[key]}")

        # Top Contributors (if enabled)
        if contributor and top_longs is not None and top_shorts is not None:
            with st.expander("🏆 Top Contributors", expanded=False):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Top 10 Long Leg Contributors:**")
                    for symbol, pnl in top_longs.items():
                        st.write(f"• {symbol}: {pnl:.4f}")

                with col2:
                    st.markdown("**Top 10 Short Leg Contributors:**")
                    for symbol, pnl in top_shorts.items():
                        st.write(f"• {symbol}: {pnl:.4f}")

        # Settings Summary
        with st.expander("⚙️ Simulation Settings", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Portfolio Settings:**")
                st.write(f"• Coverage (pct): {settings['pct']}")
                st.write(f"• Min Universe: {settings['min_universe']}")
                st.write(f"• Equal Weight: {settings['equal_weight']}")
                st.write(f"• Rebalance Period: {settings['rebalance_period']}")

            with col2:
                st.markdown("**Risk & Cost Settings:**")
                st.write(f"• Max Weight: {settings['max_weight']}")
                st.write(f"• Transaction Cost: {settings['transaction_cost']}")
                st.write(
                    f"• Cost per Turnover: {settings['cost_per_turnover']}")

        # Feature Statistics
        with st.expander("📋 Feature Statistics", expanded=False):
            display_factor_statistics(custom_feature)

        return {
            "status": "completed",
            "settings": settings,
            "returns_df": returns_df,
            "analyzer": analyzer,
            "weights": weights,
            "counts": counts
        }

    except Exception as e:
        st.error(f"Simulation failed: {str(e)}")
        st.code(f"Error details: {e}")
        return None

def main():
    """Main application logic"""
    # Page config
    st.set_page_config(page_title="Factor Analysis Notebook",
                       page_icon="📝",
                       layout="wide")

    # Apply global design system
    DesignSystem.inject_global_styles()

    # Create professional page header using design system
    DesignSystem.create_page_header(
        title="Factor Analysis Notebook",
        description=
        "Streamlined workflow: Load data → Build custom factors → Configure simulation → Run analysis",
        icon="📝")

    # Display system monitoring
    PerformanceMonitor.display_system_status()

    # Initialize session state
    initialize_session_state()

    # Initialize database manager
    @st.cache_resource
    def get_db_manager():
        return DatabaseManager()

    db_manager = get_db_manager()

    # Step 1: Load Data
    st.markdown("## 1. Load Data")

    # Get available tables
    try:
        tables = db_manager.get_table_list()
    except Exception as e:
        st.error(f"Database connection error: {e}")
        tables = None

    if tables:
        selected_table = st.selectbox(
            "Select dataset from database:",
            tables,
            help="Choose a table to load for analysis")

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("📥 Load Data", type="primary"):
                with st.spinner("Loading data..."):
                    success, result = load_data_with_progress(db_manager, selected_table, "standard")

                    if success:
                        st.session_state.df = result
                        st.success(f"✅ Loaded {len(result):,} rows from '{selected_table}'")
                        st.rerun()
                    else:
                        handle_loading_error(Exception(result), 0, 0)

        with col2:
            if st.button("🔄 Chunked Load", type="secondary", help="Load large datasets in chunks"):
                with st.spinner("Loading in chunks..."):
                    success, result = load_data_with_progress(db_manager, selected_table, "chunked")

                    if success:
                        st.session_state.df = result
                        st.success(f"✅ Chunked loaded {len(result):,} rows from '{selected_table}'")
                        st.rerun()
                    else:
                        handle_loading_error(Exception(result), 0, 0)

        # Show loaded data info if available
        if st.session_state.df is not None:
            st.info(
                f"**Loaded:** {len(st.session_state.df):,} rows | {st.session_state.df.index.get_level_values('symbol').nunique():,} symbols | {len(st.session_state.df.columns)} columns"
            )

        # Loading method guidance
        with st.expander("📊 **Loading Method Guide**", expanded=False):
            st.markdown("""
            **Choose the right loading method for your dataset:**

            ### 🔍 **Dataset Size Analysis**
            - **Small datasets** (< 1M rows): Use Load Data
            - **Large datasets** (> 1M rows): Use Chunked Load

            ### 📥 **Load Data** (Standard)
            - **Best for:** Small to medium datasets
            - **Memory:** Loads entire dataset at once
            - **Speed:** Fastest for small files
            - **Limit:** May fail with very large datasets

            ### 🔄 **Chunked Load** (Reliable)
            - **Best for:** Large datasets with many rows or columns
            - **Memory:** Loads data in 50K row chunks
            - **Speed:** Slower but more reliable for large datasets
            - **Limit:** Handles most datasets up to 10M+ rows

            ### 💡 **Memory Limits**
            - **Available Memory:** ~36 GB on this system
            - **Estimated Safe Limits:**
              - Standard: ~5M rows × 100 columns
              - Chunked: ~10M+ rows × 500+ columns

            ### 🔧 **If Loading Still Fails:**
            1. **Split your data** into smaller tables by date ranges
            2. **Reduce columns** by selecting only needed features
            3. **Sample your data** for initial analysis
            4. **Wait 30 seconds** and try again
            """)


    else:
        st.warning(
            "No data tables found. Please upload data in the Data Management page first."
        )

    # Step 2: Build Custom Factors
    st.markdown("## 2. Build Custom Factors")

    if st.session_state.df is not None:
        # Show available columns
        with st.expander("📋 Available Columns", expanded=False):
            cols = st.session_state.df.columns.tolist()
            st.write(", ".join(cols))

        # Enhanced code cell with notebook styling
        create_code_cell_styling()

        st.markdown("**📝 Factor Generation Code Cell**")

        # Show available functions and variables
        with st.expander("🔧 Available Functions & Variables", expanded=False):
            st.markdown("""
            To be completed
            """)

        # Default factor code based on your example
        default_factor_code = """# Example: Momentum factor with industry neutralization
long_period = 120
short_period = 10

# Calculate momentum (long-term minus short-term returns)
momentum = ts_sum(returns, long_period) - ts_sum(returns, short_period)

# Neutralize by subindustry (if available)
if subindustry is not None:
    momentum_neutralized = group_neutralize(momentum, subindustry)
else:
    momentum_neutralized = momentum

# Apply time decay smoothing
custom_feature = ts_decay(momentum_neutralized, 3).rename("custom_feature")"""

        # Enhanced code cell
        st.markdown('<div class="code-cell-container">', unsafe_allow_html=True)
        st.markdown(
            '<div class="code-cell-header">In [1]: # Factor Generation Cell</div>',
            unsafe_allow_html=True)

        feature_code = st.text_area(
            "Factor Code:",
            value=default_factor_code,
            height=200,
            label_visibility="collapsed",
            placeholder=
            "# Write your factor generation code here...\n# Must create 'custom_feature' variable at the end"
        )

        st.markdown('</div>', unsafe_allow_html=True)

        # Step 3: Simulation Settings (Dropdown Menu)
        st.markdown("## 3. Simulation Settings")

        # Settings dropdown with your exact format
        with st.expander("⚙️ Simulation Settings Configuration", expanded=True):
            st.markdown("**Configure simulation parameters:**")

            col1, col2 = st.columns(2)

            with col1:
                pct = st.slider("Portfolio Coverage (pct)",
                                0.05,
                                0.20,
                                0.10,
                                0.01,
                                help="Percentage of universe to hold")
                min_universe = st.number_input(
                    "Min Universe Size",
                    500,
                    2000,
                    1000,
                    100,
                    help="Minimum number of stocks in universe")
                equal_weight = st.checkbox(
                    "Equal Weight",
                    value=True,
                    help="Use equal weights vs signal weights")
                rebalance_period = st.selectbox(
                    "Rebalance Period", ["daily", "monthly"],
                    index=0,
                    help="Portfolio rebalancing frequency")

            with col2:
                max_weight = st.slider("Max Position Weight",
                                       0.01,
                                       0.05,
                                       0.025,
                                       0.005,
                                       help="Maximum weight per position")
                transaction_cost = st.checkbox("Transaction Cost",
                                               value=False,
                                               help="Apply transaction costs")
                cost_per_turnover = st.number_input(
                    "Cost per Turnover",
                    0.0001,
                    0.001,
                    0.0003,
                    0.0001,
                    help="Cost per unit of turnover",
                    format="%.4f")

            # Display the settings code format
            st.markdown("**Generated Settings Code:**")
            settings_code = f"""settings = simulation_settings(
    pct={pct}, min_universe={min_universe},
    equal_weight={equal_weight}, max_weight={max_weight},
    transaction_cost={transaction_cost}, cost_per_turnover={cost_per_turnover},
    rebalance_period="{rebalance_period}"
)"""
            st.code(settings_code, language="python")

        # Step 4: Run Simulation Button
        st.markdown("## 4. Run Simulation")

        # Display the simulation execution code
        st.markdown("**Simulation Execution Code:**")
        simulation_code = "simulation(custom_feature, settings, contributor = True)"
        st.code(simulation_code, language="python")

        if st.button("🚀 Run Simulation",
                     type="primary",
                     use_container_width=True,
                     key="main_run_button"):
            if not feature_code.strip():
                st.error("Please write feature code first")
            elif st.session_state.df is None:
                st.error("Please load data first")
            else:
                try:
                    with st.spinner("Running simulation..."):
                        progress_bar, status_text = create_progress_ui()

                        # Step 1: Execute feature code
                        update_progress(progress_bar, status_text, 33, "1/3 Executing feature code...")

                        # Execute factor code
                        success, result = execute_factor_code(feature_code, st.session_state.df)

                        if not success:
                            st.error(result)
                            st.info("💡 Check your factor code syntax and variable names.")
                            st.stop()

                        custom_feature = result

                        # Step 2: Create settings
                        update_progress(progress_bar, status_text, 66, "2/3 Creating simulation settings...")

                        # Create settings using the exact format
                        settings = simulation_settings(
                            pct=pct,
                            min_universe=min_universe,
                            equal_weight=equal_weight,
                            max_weight=max_weight,
                            transaction_cost=transaction_cost,
                            cost_per_turnover=cost_per_turnover,
                            rebalance_period=rebalance_period)

                        # Step 3: Run simulation
                        update_progress(progress_bar, status_text, 100, "3/3 Running simulation...")

                        # Execute simulation with your exact function call
                        results = simulation(custom_feature,
                                             settings,
                                             contributor=True)

                        # Clean up progress display
                        progress_bar.empty()
                        status_text.empty()

                        # Store results in session state
                        if results is not None:
                            st.session_state.simulation_results = {
                                'results': results,
                                'settings': settings,
                                'custom_feature': custom_feature
                            }

                            # -------------------------------------------------
                            # NEW: Auto-save this simulation run
                            # -------------------------------------------------
                            try:
                                summary_metrics = results['analyzer'].summary() if 'analyzer' in results else {}
                                # Recreate (or obtain) performance figure for storage
                                try:
                                    fig_to_store = results['analyzer'].plot_full_performance(counts_df=results.get('counts'))
                                except Exception:
                                    fig_to_store = None

                                save_simulation_record(
                                    code=feature_code,
                                    settings=settings,
                                    summary=summary_metrics,
                                    fig=fig_to_store,
                                )
                                st.info("💾 Simulation automatically saved. View it in the 📂 Simulation Results page.")
                            except Exception as save_err:
                                st.warning(f"Could not auto-save simulation: {save_err}")

                except Exception as e:
                    st.error(f"Simulation error: {str(e)}")
                    st.code(f"Error details: {e}")

    else:
        st.info("👆 Load data first to begin factor analysis")

if __name__ == "__main__":
    main()
