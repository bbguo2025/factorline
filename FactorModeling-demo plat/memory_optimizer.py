"""
Memory Optimizer for Large File Processing
Prevents 502 errors by managing memory usage and implementing proper retry logic
"""
import gc
import time
import psutil
import streamlit as st
from typing import Callable, Any, Optional
import pandas as pd
from functools import wraps


class MemoryOptimizer:
    """Memory management and optimization utilities"""
    
    @staticmethod
    def get_memory_status():
        """Get current memory status"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent': memory.percent,
            'is_safe': memory.percent < 85
        }
    
    @staticmethod
    def force_garbage_collection():
        """Force garbage collection to free memory"""
        gc.collect()
        time.sleep(0.1)  # Allow GC to complete
    
    @staticmethod
    def check_memory_before_operation(required_mb: float = 500) -> bool:
        """Check if enough memory is available before operation"""
        status = MemoryOptimizer.get_memory_status()
        available_mb = status['available_gb'] * 1024
        
        if available_mb < required_mb:
            st.error(f"❌ Insufficient memory. Need {required_mb:.0f}MB, have {available_mb:.0f}MB available.")
            st.info("💡 Try closing other applications or browser tabs, then refresh the page.")
            return False
        
        if status['percent'] > 85:
            st.warning(f"⚠️ High memory usage ({status['percent']:.1f}%). Operation may fail.")
            return False
            
        return True
    
    @staticmethod  
    def memory_safe_operation(operation_name: str = "Operation"):
        """Decorator for memory-safe operations with automatic cleanup"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                # Pre-operation memory check
                start_status = MemoryOptimizer.get_memory_status()
                
                try:
                    # Force garbage collection before operation
                    MemoryOptimizer.force_garbage_collection()
                    
                    # Execute operation
                    result = func(*args, **kwargs)
                    
                    return result
                    
                except MemoryError:
                    st.error(f"❌ {operation_name} failed due to insufficient memory.")
                    st.info("💡 Try with a smaller file or close other applications.")
                    return None
                    
                except Exception as e:
                    st.error(f"❌ {operation_name} failed: {str(e)}")
                    return None
                    
                finally:
                    # Post-operation cleanup
                    MemoryOptimizer.force_garbage_collection()
                    
                    # Memory usage report
                    end_status = MemoryOptimizer.get_memory_status()
                    if end_status['percent'] > start_status['percent'] + 10:
                        st.warning(f"⚠️ Memory usage increased during {operation_name}")
                        
            return wrapper
        return decorator
    
    @staticmethod
    def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage"""
        df_optimized = df.copy()
        
        for col in df_optimized.columns:
            col_type = df_optimized[col].dtype
            
            if col_type == 'object':
                # Convert to category if low cardinality
                unique_ratio = df_optimized[col].nunique() / len(df_optimized)
                if unique_ratio < 0.1:
                    df_optimized[col] = df_optimized[col].astype('category')
                    
            elif col_type == 'int64':
                # Downcast integers
                if df_optimized[col].min() >= 0:
                    if df_optimized[col].max() < 255:
                        df_optimized[col] = df_optimized[col].astype('uint8')
                    elif df_optimized[col].max() < 65535:
                        df_optimized[col] = df_optimized[col].astype('uint16')
                    elif df_optimized[col].max() < 4294967295:
                        df_optimized[col] = df_optimized[col].astype('uint32')
                else:
                    # Signed integers
                    if df_optimized[col].min() > -128 and df_optimized[col].max() < 127:
                        df_optimized[col] = df_optimized[col].astype('int8')
                    elif df_optimized[col].min() > -32768 and df_optimized[col].max() < 32767:
                        df_optimized[col] = df_optimized[col].astype('int16')
                        
            elif col_type == 'float64':
                # Downcast floats
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        return df_optimized
    
    @staticmethod
    def display_memory_warning():
        """Display memory usage warning in sidebar"""
        status = MemoryOptimizer.get_memory_status()
        
        if status['percent'] > 85:
            st.sidebar.error(f"🔴 Critical Memory: {status['percent']:.1f}%")
            st.sidebar.info("Close other applications before processing large files")
        elif status['percent'] > 70:
            st.sidebar.warning(f"🟡 High Memory: {status['percent']:.1f}%")
        else:
            st.sidebar.success(f"🟢 Memory OK: {status['percent']:.1f}%")
            
        st.sidebar.info(f"Available: {status['available_gb']:.1f} GB")


class DatabaseRetryHandler:
    """Handles database connection retries and 502 error prevention"""
    
    @staticmethod
    def retry_with_exponential_backoff(max_attempts: int = 3, base_delay: float = 2.0):
        """Decorator for database operations with retry logic"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                last_exception = None
                
                for attempt in range(max_attempts):
                    try:
                        return func(*args, **kwargs)
                        
                    except Exception as e:
                        last_exception = e
                        error_msg = str(e).lower()
                        
                        # Check if it's a retryable error
                        retryable_errors = ['502', 'timeout', 'connection', 'network', 'temporary']
                        is_retryable = any(keyword in error_msg for keyword in retryable_errors)
                        
                        if not is_retryable or attempt == max_attempts - 1:
                            # Not retryable or final attempt
                            if '502' in error_msg:
                                st.error("❌ Server overload (502 error). Wait 30 seconds and try again.")
                                st.info("💡 This often happens with large files. Try processing smaller chunks.")
                            else:
                                st.error(f"❌ Operation failed: {str(e)}")
                            raise last_exception
                        
                        # Wait before retry
                        delay = base_delay * (2 ** attempt)
                        st.warning(f"⚠️ Attempt {attempt + 1} failed. Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        
                        # Force garbage collection between retries
                        MemoryOptimizer.force_garbage_collection()
                
                if last_exception:
                    raise last_exception
                else:
                    raise Exception("Operation failed after all retry attempts")
                
            return wrapper
        return decorator


def safe_database_operation(operation_name: str = "Database operation"):
    """Combined decorator for safe database operations"""
    def decorator(func: Callable) -> Callable:
        @DatabaseRetryHandler.retry_with_exponential_backoff()
        @MemoryOptimizer.memory_safe_operation(operation_name)
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator