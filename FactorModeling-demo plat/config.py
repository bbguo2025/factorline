"""
Enhanced Configuration for Factor Analysis Platform
Centralized settings with environment-specific optimizations
"""
import os
import time
import traceback
import streamlit as st
from dataclasses import dataclass
from typing import Dict, Any, Optional
from functools import wraps

@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    url: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 300  # Increased from 30 to 300 seconds (5 minutes)
    connect_timeout: int = 300  # Increased from 30 to 300 seconds (5 minutes)
    pool_recycle: int = 600  # Increased to 10 minutes

    @classmethod
    def from_environment(cls):
        return cls(
            url=os.getenv('DATABASE_URL', ''),
            pool_size=int(os.getenv('DB_POOL_SIZE', '10')),
            max_overflow=int(os.getenv('DB_MAX_OVERFLOW', '20')),
            pool_timeout=int(os.getenv('DB_POOL_TIMEOUT', '300')),  # Increased default
            connect_timeout=int(os.getenv('DB_CONNECT_TIMEOUT', '300')),  # Increased default
            pool_recycle=int(os.getenv('DB_POOL_RECYCLE', '600'))  # Increased default
        )

@dataclass 
class PerformanceConfig:
    """Performance and resource management configuration"""
    max_file_size_mb: int = 5000  # Increased from 2000 to 5000 MB (5GB)
    chunk_size_rows: int = 100000  # Increased from 50000 to 100000
    memory_warning_threshold_gb: float = 0.5  # Reduced from 1.0 to 0.5 GB
    max_processing_time_minutes: int = 120  # Increased from 30 to 120 minutes (2 hours)
    enable_optimization: bool = True
    enable_monitoring: bool = True

    @classmethod
    def from_environment(cls):
        return cls(
            max_file_size_mb=int(os.getenv('MAX_FILE_SIZE_MB', '5000')),  # Increased default
            chunk_size_rows=int(os.getenv('CHUNK_SIZE_ROWS', '100000')),  # Increased default
            memory_warning_threshold_gb=float(os.getenv('MEMORY_WARNING_GB', '0.5')),  # Reduced default
            max_processing_time_minutes=int(os.getenv('MAX_PROCESSING_MINUTES', '120')),  # Increased default
            enable_optimization=os.getenv('ENABLE_OPTIMIZATION', 'true').lower() == 'true',
            enable_monitoring=os.getenv('ENABLE_MONITORING', 'true').lower() == 'true'
        )

@dataclass
class UIConfig:
    """User interface configuration"""
    theme: str = "professional"
    show_debug_info: bool = False
    enable_advanced_features: bool = True
    auto_refresh_status: bool = True

    @classmethod
    def from_environment(cls):
        return cls(
            theme=os.getenv('UI_THEME', 'professional'),
            show_debug_info=os.getenv('SHOW_DEBUG', 'false').lower() == 'true',
            enable_advanced_features=os.getenv('ENABLE_ADVANCED', 'true').lower() == 'true',
            auto_refresh_status=os.getenv('AUTO_REFRESH', 'true').lower() == 'true'
        )

class EnhancedConfig:
    """Centralized configuration management"""

    def __init__(self):
        self.database = DatabaseConfig.from_environment()
        self.performance = PerformanceConfig.from_environment()
        self.ui = UIConfig.from_environment()

    def validate(self) -> Dict[str, str]:
        """Validate configuration and return any errors"""
        errors = {}

        if not self.database.url:
            errors['database'] = "DATABASE_URL not configured"

        if self.performance.max_file_size_mb < 1:
            errors['performance'] = "Max file size must be at least 1MB"

        return errors

    def display_status(self):
        """Display configuration status in expander"""
        with st.expander("⚙️ Configuration Status", expanded=False):
            errors = self.validate()

            if not errors:
                st.success("✅ All configurations valid")
            else:
                for component, error in errors.items():
                    st.error(f"❌ {component.title()}: {error}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Database**")
                st.info(f"Pool Size: {self.database.pool_size}")
                st.info(f"Timeout: {self.database.connect_timeout}s")

            with col2:
                st.markdown("**Performance**")
                st.info(f"Max File: {self.performance.max_file_size_mb}MB")
                st.info(f"Chunk Size: {self.performance.chunk_size_rows:,}")

            with col3:
                st.markdown("**Features**")
                st.info(f"Monitoring: {'✅' if self.performance.enable_monitoring else '❌'}")
                st.info(f"Optimization: {'✅' if self.performance.enable_optimization else '❌'}")

# Global configuration instance
config = EnhancedConfig()

# Streamlit page configuration helper
def configure_page(title: str, icon: str = "📊", layout: str = "wide"):
    """Configure Streamlit page with consistent settings"""
    st.set_page_config(
        page_title=f"{title} - Factor Analysis Platform",
        page_icon=icon,
        layout=layout,
        initial_sidebar_state="expanded"
    )

    # Add custom CSS for consistent styling
    st.markdown("""
    <style>
        /* Enhanced professional styling */
        :root {
            --primary-color: #1f77b4;
            --secondary-color: #ff7f0e;
            --success-color: #2ca02c;
            --warning-color: #d62728;
            --info-color: #17a2b8;
            --card-background: #ffffff;
            --border-color: #e1e5e9;
            --text-color: #333333;
            --shadow: 0 4px 16px rgba(0,0,0,0.1);
        }

        /* Card styling */
        .enhanced-card {
            background: var(--card-background);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: var(--shadow);
            transition: all 0.3s ease;
        }

        .enhanced-card:hover {
            box-shadow: 0 6px 24px rgba(31, 119, 180, 0.15);
            transform: translateY(-2px);
        }

        /* Status indicators */
        .status-indicator {
            display: inline-flex;
            align-items: center;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-weight: 500;
            margin: 0.25rem;
        }

        .status-success {
            background-color: rgba(44, 160, 44, 0.1);
            color: var(--success-color);
            border: 1px solid rgba(44, 160, 44, 0.3);
        }

        .status-warning {
            background-color: rgba(214, 39, 40, 0.1);
            color: var(--warning-color);
            border: 1px solid rgba(214, 39, 40, 0.3);
        }

        .status-info {
            background-color: rgba(31, 119, 180, 0.1);
            color: var(--primary-color);
            border: 1px solid rgba(31, 119, 180, 0.3);
        }

        /* Progress enhancements */
        .stProgress > div > div > div {
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        }

        /* Button enhancements */
        .stButton > button {
            border-radius: 8px;
            border: none;
            padding: 0.5rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: var(--shadow);
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(31, 119, 180, 0.3);
        }

        /* Metric styling */
        .metric-container {
            display: flex;
            justify-content: space-around;
            margin: 2rem 0;
        }

        .metric-card {
            background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            min-width: 150px;
            box-shadow: var(--shadow);
        }

        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .metric-label {
            font-size: 0.9rem;
            opacity: 0.9;
        }
    </style>
    """, unsafe_allow_html=True)

# Error handling decorators using configuration
def with_error_handling(operation_name: str):
    """Decorator that uses configuration for error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                if config.performance.enable_monitoring:
                    start_time = time.time()

                result = func(*args, **kwargs)

                if config.performance.enable_monitoring:
                    execution_time = time.time() - start_time
                    if execution_time > 5:  # Log slow operations
                        st.info(f"⏱️ {operation_name}: {execution_time:.2f}s")

                return result

            except Exception as e:
                if config.ui.show_debug_info:
                    st.error(f"❌ {operation_name} failed: {str(e)}")
                    with st.expander("Debug Information"):
                        st.code(traceback.format_exc())
                else:
                    st.error(f"❌ {operation_name} encountered an error. Please try again.")
                raise

        return wrapper
    return decorator