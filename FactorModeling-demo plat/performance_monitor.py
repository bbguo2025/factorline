"""
Performance Monitor for Factor Analysis Platform
Tracks memory usage, execution times, and provides optimization suggestions
"""
import streamlit as st
import psutil
import time
import pandas as pd
from functools import wraps
from typing import Dict, Any, Optional
import threading
import os

class PerformanceMonitor:
    """Monitor and optimize application performance"""

    def __init__(self):
        self.metrics = {}
        self.start_time = time.time()

    @staticmethod
    def get_system_metrics() -> Dict[str, Any]:
        """Get current system resource usage"""
        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()

            return {
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_used_mb': memory_info.rss / 1024 / 1024,
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'available_memory_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024
            }
        except Exception:
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'memory_used_mb': 0,
                'disk_usage_percent': 0,
                'available_memory_gb': 0
            }

    @staticmethod
    def display_system_status():
        """Display system resource status in sidebar"""
        metrics = PerformanceMonitor.get_system_metrics()

        with st.sidebar:
            st.markdown("### 📊 System Status")

            # Memory usage
            mem_color = "red" if metrics['memory_percent'] > 80 else "orange" if metrics['memory_percent'] > 60 else "green"
            st.markdown(f"**Memory:** {metrics['memory_percent']:.1f}% ({metrics['memory_used_mb']:.0f} MB)")
            st.progress(metrics['memory_percent'] / 100)

            # CPU usage
            cpu_color = "red" if metrics['cpu_percent'] > 80 else "orange" if metrics['cpu_percent'] > 60 else "green"
            st.markdown(f"**CPU:** {metrics['cpu_percent']:.1f}%")
            st.progress(metrics['cpu_percent'] / 100)

            # Available memory
            if metrics['available_memory_gb'] < 1:
                st.warning(f"⚠️ Low memory: {metrics['available_memory_gb']:.1f} GB available")
            else:
                st.info(f"💾 Available: {metrics['available_memory_gb']:.1f} GB")

    @staticmethod
    def memory_usage_warning(threshold_gb: float = 1.0):
        """Show memory warning if usage is high"""
        metrics = PerformanceMonitor.get_system_metrics()

        if metrics['available_memory_gb'] < threshold_gb:
            st.warning(f"""
            ⚠️ **Memory Warning**

            Available memory is low ({metrics['available_memory_gb']:.1f} GB). 
            Consider:
            - Processing smaller data chunks
            - Closing other applications
            - Using database storage instead of memory
            """)
            return True
        return False

    @staticmethod
    def time_execution(operation_name: str = "Operation"):
        """Decorator to time function execution"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    execution_time = time.time() - start_time

                    if execution_time > 10:  # Show timing for long operations
                        st.info(f"⏱️ {operation_name} completed in {execution_time:.2f} seconds")

                    return result
                except Exception as e:
                    execution_time = time.time() - start_time
                    st.error(f"❌ {operation_name} failed after {execution_time:.2f} seconds")
                    raise e

            return wrapper
        return decorator

    @staticmethod
    def estimate_file_processing_time(file_size_mb: float) -> str:
        """Estimate processing time based on file size"""
        if file_size_mb < 10:
            return "< 30 seconds"
        elif file_size_mb < 50:
            return "1-2 minutes"
        elif file_size_mb < 200:
            return "2-5 minutes"
        elif file_size_mb < 500:
            return "5-10 minutes"
        else:
            return "10+ minutes"

    @staticmethod
    def show_processing_progress(total_items: int, current_item: int, operation: str = "Processing"):
        """Show progress bar for long operations"""
        progress = current_item / total_items
        progress_bar = st.progress(progress)
        status_text = st.empty()
        status_text.text(f"{operation}: {current_item}/{total_items} ({progress:.1%})")
        return progress_bar, status_text

class DatabasePerformance:
    """Database-specific performance optimizations"""

    @staticmethod
    def get_optimal_chunk_size(file_size_mb: float) -> int:
        """Calculate optimal chunk size based on available memory and file size"""
        metrics = PerformanceMonitor.get_system_metrics()
        available_memory_gb = metrics['available_memory_gb']

        if available_memory_gb > 4:
            if file_size_mb < 100:
                return 10000
            elif file_size_mb < 500:
                return 5000
            else:
                return 2000
        elif available_memory_gb > 2:
            return 5000 if file_size_mb < 200 else 2000
        else:
            return 1000  # Conservative for low memory

    @staticmethod
    def suggest_optimization(df: pd.DataFrame) -> Optional[str]:
        """Suggest optimizations based on data characteristics"""
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024

        suggestions = []

        if memory_usage > 500:  # > 500MB
            suggestions.append("Consider processing data in chunks")

        if len(df) > 1000000:  # > 1M rows
            suggestions.append("Use database storage for better performance")

        object_cols = df.select_dtypes(include=['object']).columns
        if len(object_cols) > 5:
            suggestions.append("Convert repeated text columns to categories")

        if suggestions:
            return "💡 **Performance Tips:**\n" + "\n".join(f"• {tip}" for tip in suggestions)

        return None