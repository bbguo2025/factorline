"""
Enhanced Error Handler for Factor Analysis Platform
Provides robust error handling, retry logic, and user-friendly error messages
"""
import streamlit as st
import time
import logging
from functools import wraps
from typing import Callable, Any, Optional
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorHandler:
    """Enhanced error handling with retry logic and user-friendly messages"""
    
    @staticmethod
    def retry_operation(max_retries: int = 3, delay: float = 1.0):
        """Decorator for retrying operations that might fail temporarily"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                last_exception = None
                
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                        
                        if attempt < max_retries - 1:
                            time.sleep(delay * (attempt + 1))  # Exponential backoff
                        else:
                            # Last attempt failed
                            error_msg = ErrorHandler.get_user_friendly_message(e)
                            st.error(f"Operation failed after {max_retries} attempts: {error_msg}")
                            logger.error(f"Final failure: {str(e)}\n{traceback.format_exc()}")
                            raise last_exception
                            
                return None
            return wrapper
        return decorator
    
    @staticmethod
    def get_user_friendly_message(error: Exception) -> str:
        """Convert technical errors to user-friendly messages"""
        error_str = str(error).lower()
        
        # Database connection errors
        if any(keyword in error_str for keyword in ['connection', 'database', 'postgresql']):
            return "Database connection issue. Please check your internet connection and try again."
        
        # Memory errors
        if any(keyword in error_str for keyword in ['memory', 'memoryerror']):
            return "Not enough memory to process this file. Try uploading a smaller file or close other applications."
        
        # File errors  
        if any(keyword in error_str for keyword in ['file', 'io', 'permission']):
            return "File access error. Please check the file format and permissions."
        
        # Network errors
        if any(keyword in error_str for keyword in ['timeout', 'network', '502', '503', '504']):
            return "Network timeout. Please check your connection and try again."
        
        # Pandas/data processing errors
        if any(keyword in error_str for keyword in ['pandas', 'dataframe', 'parse']):
            return "Data processing error. Please check your data format and column names."
        
        # Default message
        return f"An error occurred: {str(error)[:100]}..."
    
    @staticmethod
    def safe_execute(func: Callable, error_message: str = "Operation failed", 
                    show_details: bool = False) -> Optional[Any]:
        """Safely execute a function with error handling"""
        try:
            return func()
        except Exception as e:
            if show_details:
                st.error(f"{error_message}: {str(e)}")
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
            else:
                user_msg = ErrorHandler.get_user_friendly_message(e)
                st.error(f"{error_message}: {user_msg}")
            
            logger.error(f"{error_message}: {str(e)}\n{traceback.format_exc()}")
            return None

    @staticmethod 
    def connection_status_indicator(test_func: Callable, service_name: str = "Service"):
        """Display connection status with retry option"""
        status_placeholder = st.empty()
        
        def check_status():
            try:
                if test_func():
                    status_placeholder.success(f"✅ {service_name} Connected")
                    return True
                else:
                    status_placeholder.error(f"❌ {service_name} Disconnected")
                    return False
            except Exception as e:
                error_msg = ErrorHandler.get_user_friendly_message(e)
                status_placeholder.error(f"❌ {service_name} Error: {error_msg}")
                return False
        
        # Initial check
        is_connected = check_status()
        
        # Show retry button if disconnected
        if not is_connected:
            if st.button(f"🔄 Retry {service_name} Connection"):
                with st.spinner(f"Reconnecting to {service_name}..."):
                    check_status()
        
        return is_connected

# Context manager for error handling
class safe_operation:
    """Context manager for safe operations with automatic error handling"""
    
    def __init__(self, operation_name: str = "Operation", show_spinner: bool = True):
        self.operation_name = operation_name
        self.show_spinner = show_spinner
        
    def __enter__(self):
        if self.show_spinner:
            self.spinner = st.spinner(f"{self.operation_name} in progress...")
            self.spinner.__enter__()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.show_spinner and hasattr(self, 'spinner'):
            self.spinner.__exit__(exc_type, exc_val, exc_tb)
            
        if exc_type is not None:
            error_msg = ErrorHandler.get_user_friendly_message(exc_val)
            st.error(f"{self.operation_name} failed: {error_msg}")
            logger.error(f"{self.operation_name} failed: {str(exc_val)}\n{traceback.format_exc()}")
            return True  # Suppress the exception
        else:
            st.success(f"{self.operation_name} completed successfully")
        
        return False