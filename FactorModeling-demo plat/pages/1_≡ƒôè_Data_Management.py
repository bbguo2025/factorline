"""
Data Management Page
Clean interface for CSV upload and database operations
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import io
from typing import Dict, Any, Optional, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database_manager import DatabaseManager
from error_handler import ErrorHandler, safe_operation
from performance_monitor import PerformanceMonitor
from memory_optimizer import MemoryOptimizer
from styles.design_system import DesignSystem
from utils.streamlit_helpers import initialize_session_state, create_upload_progress_ui, update_upload_progress

# =============================================================================
# CONFIGURATION AND INITIALIZATION
# =============================================================================

# Page config
st.set_page_config(
    page_title="Data Management",
    page_icon="📊",
    layout="wide"
)

# Apply global design system
DesignSystem.inject_global_styles()

# Create professional page header using design system
DesignSystem.create_page_header(
    title="Data Management",
    description="Upload large CSV files and manage database storage for quantitative analysis",
    icon="📊"
)

# Display system performance monitoring
PerformanceMonitor.display_system_status()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_resource
def get_database_manager():
    """Get cached database manager instance"""
    return DatabaseManager()

def validate_csv_structure(df: pd.DataFrame) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Validate CSV structure and return detailed analysis

    Returns:
        (is_valid, message, analysis_dict)
    """
    try:
        analysis = {
            'has_required_cols': False,
            'missing_cols': [],
            'data_types': {},
            'sample_dates': [],
            'unique_symbols': 0,
            'date_range': None,
            'warnings': []
        }

        # Check required columns
        required_cols = ['date', 'symbol']
        missing_cols = [col for col in required_cols if col not in df.columns]
        analysis['missing_cols'] = missing_cols
        analysis['has_required_cols'] = len(missing_cols) == 0

        if not analysis['has_required_cols']:
            return False, f"Missing required columns: {missing_cols}", analysis

        # Analyze data types
        analysis['data_types'] = df.dtypes.to_dict()

        # Check date column
        try:
            df['date'] = pd.to_datetime(df['date'])
            analysis['sample_dates'] = df['date'].head(5).dt.strftime('%Y-%m-%d').tolist()
            analysis['date_range'] = f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
        except Exception as e:
            analysis['warnings'].append(f"Date parsing issue: {str(e)}")

        # Check symbol column
        analysis['unique_symbols'] = df['symbol'].nunique()

        # Additional validations
        if len(df) == 0:
            return False, "File contains no data", analysis

        if analysis['unique_symbols'] < 2:
            analysis['warnings'].append("Only one unique symbol found - consider multi-symbol data")

        return True, "File structure is valid", analysis

    except Exception as e:
        return False, f"Validation error: {str(e)}", {}

def get_file_size_category(file_size_mb: float) -> Dict[str, Any]:
    """Categorize file size and provide appropriate guidance"""
    if file_size_mb <= 10:
        return {
            'category': 'small',
            'processing_time': '< 30 seconds',
            'risk_level': 'low',
            'recommendations': ['Direct upload recommended']
        }
    elif file_size_mb <= 50:
        return {
            'category': 'medium',
            'processing_time': '1-2 minutes',
            'risk_level': 'low',
            'recommendations': ['Chunked processing will be used']
        }
    elif file_size_mb <= 100:
        return {
            'category': 'large',
            'processing_time': '2-5 minutes',
            'risk_level': 'medium',
            'recommendations': ['May hit deployment limits', 'Consider splitting if upload fails']
        }
    elif file_size_mb <= 500:
        return {
            'category': 'very_large',
            'processing_time': '5-10 minutes',
            'risk_level': 'high',
            'recommendations': ['High risk of deployment timeout', 'Consider local development']
        }
    else:
        return {
            'category': 'massive',
            'processing_time': '10+ minutes',
            'risk_level': 'very_high',
            'recommendations': ['Very high risk of failure', 'Use local development only']
        }

def display_file_analysis(file_size_mb: float, analysis: Dict[str, Any]):
    """Display comprehensive file analysis"""
    size_info = get_file_size_category(file_size_mb)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("File Size", f"{file_size_mb:.1f} MB", 
                 delta=f"{size_info['category'].title()} file")

    with col2:
        st.metric("Processing Time", size_info['processing_time'])

    with col3:
        st.metric("Risk Level", size_info['risk_level'].title())

    # Show recommendations
    if size_info['recommendations']:
        st.info("💡 **Recommendations:** " + " | ".join(size_info['recommendations']))

def handle_upload_error(error: Exception, file_size_mb: float) -> None:
    """Enhanced error handling with specific guidance"""
    error_message = str(error).lower()

    if "413" in error_message or "request entity too large" in error_message:
        st.error("❌ **Upload Failed: File Too Large for Deployment**")
        st.markdown(f"""
        **This is a 413 error - your file is too large for the deployed server:**

        🔧 **Solutions:**
        1. **Split your file** into smaller chunks (under 100MB each)
        2. **Use local development** for large files (works up to 2GB)
        3. **Compress your CSV** to reduce file size
        4. **Remove unnecessary columns** before upload

        📊 **Current file size:** {file_size_mb:.1f} MB  
        🚀 **Deployment limit:** ~100 MB  
        💻 **Local development limit:** 2,048 MB
        """)
    elif "timeout" in error_message or "502" in error_message:
        st.error("❌ **Upload Timeout - File Processing Taking Too Long**")
        st.info("Large files may time out in deployed environments. Try splitting into smaller files.")
    elif "memory" in error_message:
        st.error("❌ **Memory Error - Insufficient System Resources**")
        st.info("Close other applications and browser tabs, then try again.")
    else:
        st.error(f"❌ **Upload Error:** {error}")
        if file_size_mb > 100:
            st.info("💡 Large file detected. This error might be related to deployment server limits.")

def create_table_action_buttons(table: str) -> None:
    """Create consistent table action buttons"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button(f"👁️ Preview", key=f"preview_{table}"):
            st.session_state[f"show_preview_{table}"] = True
            st.rerun()

    with col2:
        if st.button(f"📥 Load", key=f"load_{table}"):
            st.code(f"load_data('{table}')", language="python")
            st.info("Copy this code to Factor Analysis Notebook")

    with col3:
        if st.button(f"📊 Info", key=f"info_{table}"):
            st.session_state[f"show_info_{table}"] = True
            st.rerun()

    with col4:
        if st.button(f"🗑️ Delete", key=f"delete_{table}", type="secondary"):
            st.session_state[f"confirm_delete_{table}"] = True
            st.rerun()

def display_table_preview(table: str, db_manager: DatabaseManager) -> None:
    """Display table preview with enhanced error handling"""
    try:
        with st.spinner("Loading preview..."):
            sample_data = db_manager.load_table_data(table, limit=10)
            st.subheader(f"Preview: {table}")

            # Fix date column display in preview
            if 'date' in sample_data.columns and sample_data['date'].dtype == 'object':
                try:
                    sample_data['date'] = pd.to_datetime(sample_data['date'])
                except:
                    pass  # Keep as string if conversion fails

            st.dataframe(sample_data, use_container_width=True)
            if st.button("Close Preview", key=f"close_preview_{table}"):
                st.session_state[f"show_preview_{table}"] = False
                st.rerun()
    except Exception as e:
        st.error(f"Error loading preview: {e}")

def display_table_info(table: str, db_manager: DatabaseManager, info: Dict[str, Any]) -> None:
    """Display detailed table information"""
    try:
        with st.spinner("Loading table details..."):
            sample_data = db_manager.load_table_data(table, limit=1000)
            st.subheader(f"Table Details: {table}")

            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Total Rows", f"{info.get('row_count', 'N/A'):,}")
                # Fix date handling - convert to datetime if it's a string
                if 'date' in sample_data.columns:
                    try:
                        # Convert date column to datetime if it's a string
                        if sample_data['date'].dtype == 'object':
                            sample_data['date'] = pd.to_datetime(sample_data['date'])
                        date_range = f"{sample_data['date'].min().date()} to {sample_data['date'].max().date()}"
                        st.metric("Date Range", date_range)
                    except Exception as date_error:
                        st.metric("Date Range", "Date format error")
                else:
                    st.metric("Date Range", "No date column")
            with col_info2:
                st.metric("Unique Symbols", f"{sample_data['symbol'].nunique():,}")
                st.metric("Columns", len(sample_data.columns))

            st.write("**Column List:**")
            st.write(", ".join(sample_data.columns.tolist()))

            if st.button("Close Info", key=f"close_info_{table}"):
                st.session_state[f"show_info_{table}"] = False
                st.rerun()
    except Exception as e:
        st.error(f"Error loading info: {e}")

def handle_table_deletion(table: str, db_manager: DatabaseManager) -> None:
    """Handle table deletion with confirmation"""
    if st.session_state.get(f"confirm_delete_{table}", False):
        st.warning(f"⚠️ Are you sure you want to delete table '{table}'? This action cannot be undone.")
        col_del1, col_del2 = st.columns(2)
        with col_del1:
            if st.button(f"✅ Yes, Delete {table}", key=f"confirm_yes_{table}", type="primary"):
                try:
                    success = db_manager.delete_table(table)
                    if success:
                        st.success(f"Table '{table}' deleted successfully")
                        del st.session_state[f"confirm_delete_{table}"]
                        st.rerun()
                    else:
                        st.error("Failed to delete table")
                except Exception as e:
                    st.error(f"Deletion error: {e}")
        with col_del2:
            if st.button("❌ Cancel", key=f"cancel_delete_{table}"):
                del st.session_state[f"confirm_delete_{table}"]
                st.rerun()

# =============================================================================
# MAIN APPLICATION LOGIC
# =============================================================================

def main():
    """Main application logic"""
    # Initialize session state
    initialize_session_state()

    # Initialize database manager
    db_manager = get_database_manager()

    # Clear cache if needed for updates
    if st.button("🔄 Refresh Connection", help="Clear cache and refresh database connection"):
        st.cache_resource.clear()
        st.rerun()

    # Database connection status
    st.markdown("### 🔌 Database Connection Status")

    database_available = False
    try:
        if db_manager.test_connection():
            DesignSystem.create_status_indicator("success", "Database Connected - Ready for large file uploads")
            database_available = True
        else:
            DesignSystem.create_status_indicator("error", "Database Unavailable - Service currently unavailable")
    except Exception as e:
        DesignSystem.create_status_indicator("error", f"Connection Error: {str(e)}")

    # Main functionality tabs
    tab1, tab2 = st.tabs(["📁 Upload Large CSV", "🗄️ Manage Database"])

    # Tab 1: Upload Large CSV
    with tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #1f77b4; margin-top: 0;">CSV File Upload</h3>', unsafe_allow_html=True)

        # Expected CSV structure information
        with st.expander("📋 CSV Format Requirements", expanded=False):
            st.markdown("""
            **Required columns:**
            - `date`: Date column (YYYY-MM-DD format)
            - `symbol`: Stock/asset symbol
            - `close`: Closing price
            - `volume`: Trading volume

            **Optional columns:**
            - `open, high, low`: OHLC price data  
            - `vwap`: Volume weighted average price
            - `returns, log_return`: Return calculations
            - `industry, subindustry`: Sector classifications

            **Sample format:**
            ```csv
            date,symbol,open,high,low,close,volume
            2023-01-01,AAPL,150.00,155.00,149.00,154.50,1000000
            2023-01-01,MSFT,250.00,255.00,248.00,253.25,800000
            2023-01-02,AAPL,154.50,158.00,153.00,157.25,1200000
            ```

            **File Requirements:**
            - CSV format with UTF-8 encoding
            - Long format (one row per date-symbol pair)
            - Files >100MB processed in chunks for efficiency
            - Maximum size: 2GB (1.2GB files supported)
            - Large files may take several minutes to process
            """)

        if database_available:
            # Enhanced upload zone using design system
            DesignSystem.create_upload_zone("Upload Large CSV File")

            uploaded_file = st.file_uploader(
                "Choose your large CSV file",
                type=['csv'],
                help="Upload CSV file with symbol_features_long.csv structure"
            )

            # Add deployment-specific upload guidance
            st.info("""
            **📋 Upload Guidelines for Deployed Apps:**
            - For files under 100MB: Use the upload button above
            - For larger files: You may encounter upload limits in deployed environments
            - If you get a 413 error: The file is too large for the deployment server
            - Consider splitting large files or using database import tools
            """)

            if uploaded_file is not None:
                # Show file info
                file_size_mb = uploaded_file.size / (1024 * 1024)

                # Check file size limits - different for deployed vs local
                deployment_limit_mb = 100  # Most deployed environments limit to ~100MB

                if file_size_mb > 2048:  # 2GB hard limit
                    st.error("❌ File too large. Maximum size is 2GB.")
                    st.info("💡 For files larger than 2GB, consider using database import tools or splitting the data.")
                    st.stop()
                elif file_size_mb > deployment_limit_mb:  # Deployment warning
                    st.warning(f"⚠️ Large file detected ({file_size_mb:.1f} MB)")
                    st.warning("🚨 **Important:** Files over 100MB may fail in deployed environments due to server limits")
                    st.info("If upload fails with 413 error, try splitting your file into smaller chunks")
                elif file_size_mb > 50:  # Medium file warning
                    st.info(f"📊 File size: {file_size_mb:.1f} MB - Processing may take a few minutes")

                # Display file analysis
                display_file_analysis(file_size_mb, {})

                st.markdown(f"""
                <div class="metric-grid">
                    <div class="metric-box">
                        <div class="metric-value">{uploaded_file.name}</div>
                        <div class="metric-label">Filename</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">{file_size_mb:.1f} MB</div>
                        <div class="metric-label">File Size</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-value">CSV</div>
                        <div class="metric-label">Format</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Table name input
                col1, col2 = st.columns([2, 1])
                with col1:
                    table_name = st.text_input(
                        "Database Table Name", 
                        value="symbol_features_data",
                        help="Name for the table in database"
                    )

                with col2:
                    overwrite_existing = st.checkbox("Overwrite if exists", value=True)

                # Memory status before upload with enhanced monitoring
                MemoryOptimizer.display_memory_warning()
                memory_status = MemoryOptimizer.get_memory_status()

                if not memory_status['is_safe']:
                    st.warning(f"⚠️ High memory usage ({memory_status['percent']:.1f}%). Large file upload may fail.")
                    st.info(f"Available memory: {memory_status['available_gb']:.1f} GB")

                # Upload button
                if st.button("🚀 Upload to Database", type="primary"):
                    if not table_name:
                        st.error("Please provide a table name")
                    else:
                        # Enhanced pre-upload safety checks
                        if not MemoryOptimizer.check_memory_before_operation(file_size_mb * 2):
                            st.stop()

                        # Display warning for large files
                        if file_size_mb > 500:
                            st.warning("⚠️ Large file detected. Processing will use chunked upload to prevent timeouts.")
                            st.info("💡 The system will process your file in small chunks to prevent 502 errors.")

                        try:
                            with st.spinner("Processing large CSV file..."):
                                # Create progress UI
                                progress_bar, status_text = create_upload_progress_ui()

                                # Deployment environment check
                                if file_size_mb > 100:
                                    update_upload_progress(progress_bar, status_text, 5, 
                                                         "⚠️ Large file detected - checking deployment compatibility...")
                                    st.warning("Large files may hit deployment server limits. If this fails, consider splitting your data.")

                                # Read file in chunks for large files
                                update_upload_progress(progress_bar, status_text, 10, "Reading CSV file...")

                                # Reset file pointer to beginning
                                uploaded_file.seek(0)

                                # Memory-efficient processing using DatabaseManager chunked import
                                update_upload_progress(progress_bar, status_text, 20, "Initializing memory-efficient upload...")

                                # Quick validation check - read only first few rows
                                uploaded_file.seek(0)
                                try:
                                    sample_df = pd.read_csv(uploaded_file, nrows=5)
                                    update_upload_progress(progress_bar, status_text, 30, "Validating file structure...")
                                except Exception as e:
                                    st.error(f"❌ Error reading CSV file: {e}")
                                    st.stop()

                                # Validate file structure
                                is_valid, validation_message, analysis = validate_csv_structure(sample_df)
                                if not is_valid:
                                    st.error(f"❌ {validation_message}")
                                    st.stop()

                                update_upload_progress(progress_bar, status_text, 40, "Validation passed. Starting database import...")

                                # Use DatabaseManager's chunked CSV import for memory efficiency
                                uploaded_file.seek(0)  # Reset file pointer

                                # Memory-efficient import using chunked processing
                                update_upload_progress(progress_bar, status_text, 50, 
                                                     "Importing CSV using memory-efficient chunked processing...")

                                success = db_manager._import_csv_chunked(uploaded_file, table_name)

                                if success:
                                    update_upload_progress(progress_bar, status_text, 100, "Upload completed successfully!")

                                    # Get table info for display
                                    try:
                                        table_info = db_manager.get_table_info(table_name)

                                        # Get additional data statistics
                                        try:
                                            # Load a sample to analyze data structure
                                            sample_data = db_manager.load_table_data(table_name, limit=1000)
                                            unique_symbols = sample_data['symbol'].nunique() if 'symbol' in sample_data.columns else 0

                                            # Get date range if available  
                                            date_range = "Unknown"
                                            if 'date' in sample_data.columns:
                                                try:
                                                    sample_data['date'] = pd.to_datetime(sample_data['date'])
                                                    min_date = sample_data['date'].min().strftime('%Y-%m-%d')
                                                    max_date = sample_data['date'].max().strftime('%Y-%m-%d')
                                                    date_range = f"{min_date} to {max_date}"
                                                except:
                                                    date_range = "Date format unknown"
                                        except:
                                            unique_symbols = 0
                                            date_range = "Unknown"

                                        # Store upload info with all required fields
                                        st.session_state.uploaded_data_info = {
                                            'table_name': table_name,
                                            'filename': uploaded_file.name,
                                            'rows': table_info.get('row_count', 0),
                                            'columns': len(table_info.get('columns', [])),
                                            'unique_symbols': unique_symbols,
                                            'date_range': date_range,
                                            'file_size_mb': file_size_mb,
                                            'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                        }
                                        st.session_state.upload_tab_active = True

                                        st.success(f"✅ Successfully uploaded data to table '{table_name}'")
                                        st.balloons()
                                        st.rerun()

                                    except Exception as e:
                                        st.warning(f"Upload completed but could not retrieve table info: {e}")

                                else:
                                    st.error("❌ Failed to import CSV file to database")

                        except Exception as e:
                            handle_upload_error(e, file_size_mb)
        else:
            st.warning("Database connection required for file uploads. Please check connection status above.")

        # Show upload success info within this tab
        if st.session_state.uploaded_data_info:
            info = st.session_state.uploaded_data_info

            st.markdown('<div class="section-card" style="margin-top: 2rem; border-left-color: #2ca02c;">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #2ca02c; margin-top: 0;">✅ Upload Successful</h3>', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rows", f"{info['rows']:,}")
            with col2:
                st.metric("Unique Symbols", f"{info['unique_symbols']:,}")
            with col3:
                st.metric("Columns", info['columns'])
            with col4:
                st.metric("File Size", f"{info['file_size_mb']:.1f} MB")

            st.info(f"**Table:** `{info['table_name']}` | **Date Range:** {info['date_range']} | **File:** {info['filename']}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Tab 2: Manage Database
    with tab2:
        if database_available:
            # Show existing tables
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<h3 style="color: #1f77b4; margin-top: 0;">📋 Database Tables</h3>', unsafe_allow_html=True)

            try:
                tables = db_manager.get_table_list()
                if tables:
                    st.write(f"**Found {len(tables)} table(s):**")

                    for table in tables:
                        try:
                            info = db_manager.get_table_info(table)

                            st.markdown(f"""
                            <div class="table-card">
                                <div style="display: flex; justify-content: between; align-items: center; margin-bottom: 1rem;">
                                    <h4 style="color: #2ca02c; margin: 0;">📊 {table}</h4>
                                </div>
                                <div class="metric-grid">
                                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center;">
                                        <div style="font-size: 1.2rem; font-weight: 600;">{info.get('row_count', 'N/A'):,}</div>
                                        <div style="font-size: 0.9rem; color: #666;">Rows</div>
                                    </div>
                                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center;">
                                        <div style="font-size: 1.2rem; font-weight: 600;">{info.get('column_count', 'N/A')}</div>
                                        <div style="font-size: 0.9rem; color: #666;">Columns</div>
                                    </div>
                                    <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center;">
                                        <div style="font-size: 1.2rem; font-weight: 600;">{info.get('size_mb', 'N/A')}</div>
                                        <div style="font-size: 0.9rem; color: #666;">Size (MB)</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                            # Create table action buttons
                            create_table_action_buttons(table)

                            # Show preview if requested
                            if st.session_state.get(f"show_preview_{table}", False):
                                display_table_preview(table, db_manager)

                            # Show detailed info if requested
                            if st.session_state.get(f"show_info_{table}", False):
                                display_table_info(table, db_manager, info)

                            # Handle table deletion
                            handle_table_deletion(table, db_manager)

                            st.markdown("---")

                        except Exception as e:
                            st.error(f"Error getting info for table {table}: {e}")

                else:
                    st.info("No tables found in database. Upload a CSV file to get started.")

            except Exception as e:
                st.error(f"Error retrieving tables: {e}")

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.warning("Database connection required to manage tables. Please check connection status above.")

if __name__ == "__main__":
    main()