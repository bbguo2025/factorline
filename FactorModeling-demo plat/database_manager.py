"""
Database Manager for Excel File Import and Storage
Handles large Excel files and provides database operations
"""
import os
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Float, DateTime, Integer
from sqlalchemy.dialects.postgresql import UUID
import uuid
from datetime import datetime
import streamlit as st
from typing import Optional, List, Dict, Any, Literal
import io
from memory_optimizer import MemoryOptimizer, safe_database_operation

# Constants for configuration
CHUNK_SIZE_LARGE = 2500      # For files > 1M rows
CHUNK_SIZE_MEDIUM = 5000     # For files > 500K rows  
CHUNK_SIZE_SMALL = 10000     # For files > 100K rows
CHUNK_SIZE_DEFAULT = 1000    # Default chunk size
MEMORY_THRESHOLD_MB = 200    # Memory threshold for operations
STATEMENT_TIMEOUT_MS = 600000 # 10 minutes statement timeout


class DatabaseManager:
    """
    Manages database operations for Excel file imports and data storage.
    """

    def __init__(self):
        """Initialize database connection."""
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not found")

        # Configure engine with Replit-optimized settings
        self.engine = create_engine(
            self.database_url,
            pool_pre_ping=True,
            pool_recycle=300,  # Reduced to 5 minutes for Replit
            pool_timeout=60,   # Reduced timeout for Replit (1 minute)
            pool_size=5,       # Reduced pool size for Replit memory constraints
            max_overflow=10,   # Reduced overflow connections
            connect_args={
                "connect_timeout": 60,  # Reduced connection timeout (1 minute)
                "sslmode": "prefer",   # Changed from require to prefer for Replit
                "application_name": "factor_analysis_platform",
                "options": "-c statement_timeout=600000"  # 10 minutes statement timeout (reduced from 30)
            }
        )
        self.metadata = MetaData()

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            error_msg = str(e).lower()
            if '502' in error_msg or 'timeout' in error_msg:
                print(f"Database connection timeout (502 error): {str(e)}")
            else:
                print(f"Database connection failed: {str(e)}")
            return False

    def get_table_list(self) -> List[str]:
        """Get list of all tables in the database."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name
                """))
                return [row[0] for row in result.fetchall()]
        except Exception as e:
            st.error(f"Error fetching table list: {str(e)}")
            return []

    def analyze_excel_structure(self, file) -> Dict[str, Any]:
        """
        Analyze Excel file structure without loading all data.

        Parameters:
        - file: uploaded file object

        Returns:
        - Dictionary with file analysis results
        """
        try:
            # Reset file pointer
            file.seek(0)

            # Try to read Excel file
            if file.name.endswith('.xlsx'):
                excel_file = pd.ExcelFile(file)
                sheets = excel_file.sheet_names

                analysis = {
                    'filename': file.name,
                    'sheets': sheets,
                    'file_size': file.size,
                    'sheet_info': {}
                }

                # Analyze each sheet
                for sheet in sheets:
                    try:
                        # Read just the first few rows to understand structure
                        df_sample = pd.read_excel(file, sheet_name=sheet, nrows=5)

                        analysis['sheet_info'][sheet] = {
                            'columns': list(df_sample.columns),
                            'column_count': len(df_sample.columns),
                            'data_types': df_sample.dtypes.to_dict(),
                            'sample_data': df_sample.head(3).to_dict('records')
                        }

                        # Get total row count
                        df_full = pd.read_excel(file, sheet_name=sheet)
                        analysis['sheet_info'][sheet]['row_count'] = len(df_full)

                    except Exception as e:
                        analysis['sheet_info'][sheet] = {
                            'error': f"Could not analyze sheet: {str(e)}"
                        }

                return analysis

            else:
                # Handle CSV files with memory-efficient row counting
                file.seek(0)
                df_sample = pd.read_csv(file, nrows=5)

                # Memory-efficient row counting without loading entire file
                file.seek(0)
                row_count = sum(1 for _ in file) - 1  # Subtract 1 for header
                file.seek(0)  # Reset for future use

                return {
                    'filename': file.name,
                    'sheets': ['csv_data'],
                    'file_size': file.size,
                    'sheet_info': {
                        'csv_data': {
                            'columns': list(df_sample.columns),
                            'column_count': len(df_sample.columns),
                            'row_count': row_count,
                            'data_types': df_sample.dtypes.to_dict(),
                            'sample_data': df_sample.head(3).to_dict('records')
                        }
                    }
                }

        except Exception as e:
            return {'error': f"Error analyzing file: {str(e)}"}

    def create_table_from_dataframe(self, df: pd.DataFrame, table_name: str, 
                                   if_exists: Literal['replace', 'append', 'fail'] = 'replace') -> bool:
        """
        Create database table from DataFrame.

        Parameters:
        - df: DataFrame to create table from
        - table_name: Name of the table to create
        - if_exists: What to do if table exists ('replace', 'append', 'fail')

        Returns:
        - Boolean indicating success
        """
        try:
            # Clean table name (remove spaces, special characters)
            clean_table_name = self._clean_table_name(table_name)

            # Add metadata columns
            df_with_meta = df.copy()
            df_with_meta['import_id'] = str(uuid.uuid4())
            df_with_meta['import_timestamp'] = datetime.now()

            # Convert data types for better storage
            df_with_meta = self._optimize_dtypes(df_with_meta)

            # Create table in database with optimized chunking for large files
            rows = len(df_with_meta)
            if rows > 1000000:  # For very large files (1M+ rows), use smaller chunks
                chunk_size = 2500
            elif rows > 500000:  # Large files
                chunk_size = 5000
            elif rows > 100000:  # Medium files
                chunk_size = 10000
            else:
                chunk_size = 1000

            df_with_meta.to_sql(
                clean_table_name, 
                self.engine, 
                if_exists=if_exists,  # type: ignore
                index=False,
                method='multi',  # Faster bulk insert
                chunksize=chunk_size   # Dynamic chunk size based on data size
            )

            st.success(f"Successfully created table '{clean_table_name}' with {len(df)} rows")
            return True

        except Exception as e:
            st.error(f"Error creating table: {str(e)}")
            return False

    def import_excel_to_database(self, file, selected_sheets: List[str], 
                                table_prefix: str = "") -> Dict[str, bool]:
        """
        Import selected Excel sheets to database tables.

        Parameters:
        - file: uploaded file object
        - selected_sheets: list of sheet names to import
        - table_prefix: prefix for table names

        Returns:
        - Dictionary with results for each sheet
        """
        results = {}

        try:
            file.seek(0)

            if file.name.endswith('.xlsx'):
                excel_file = pd.ExcelFile(file)

                for sheet in selected_sheets:
                    try:
                        # Read sheet data
                        df = pd.read_excel(file, sheet_name=sheet)

                        # Create table name
                        table_name = f"{table_prefix}{sheet}" if table_prefix else sheet
                        table_name = self._clean_table_name(table_name)

                        # Import to database
                        success = self.create_table_from_dataframe(df, table_name)
                        results[sheet] = success

                    except Exception as e:
                        st.error(f"Error importing sheet '{sheet}': {str(e)}")
                        results[sheet] = False

            else:
                # Handle CSV file with chunked processing for large files
                file.seek(0)
                table_name = f"{table_prefix}csv_data" if table_prefix else "csv_data"
                table_name = self._clean_table_name(table_name)

                success = self._import_csv_chunked(file, table_name)
                results['csv_data'] = success

        except Exception as e:
            st.error(f"Error during import: {str(e)}")

        return results

    def load_table_data(self, table_name: str, limit: Optional[int] = None, offset: Optional[int] = None) -> pd.DataFrame:
        """
        Load data from database table with simple, reliable approach.

        Parameters:
        - table_name: name of table to load
        - limit: maximum number of rows to load
        - offset: number of rows to skip (for chunked loading)

        Returns:
        - DataFrame with table data
        """
        try:
            query = f"SELECT * FROM {table_name}"

            # Add offset if specified
            if offset is not None:
                query += f" OFFSET {offset}"

            # Add limit if specified
            if limit:
                query += f" LIMIT {limit}"

            # Simple, direct loading with reasonable timeout
            with self.engine.connect() as conn:
                # Set reasonable timeout for large queries
                conn.execute(text("SET statement_timeout = '600s'"))  # 10 minutes

                # Load data directly
                df = pd.read_sql(query, conn)

                return df

        except Exception as e:
            error_msg = str(e).lower()

            # Simple error handling
            if '502' in error_msg or 'timeout' in error_msg:
                st.error("❌ **Connection timeout (502 error)**")
                st.markdown("""
                **This happens with very large datasets. Try these solutions:**

                🔧 **Immediate fixes:**
                1. **Wait 30 seconds** and try again
                2. **Close other browser tabs** to free memory
                3. **Use smaller dataset** - split your data

                📊 **For large datasets:**
                - Process data in smaller batches
                - Consider sampling your data first
                """)
            else:
                st.error(f"❌ **Database error:** {str(e)}")

            return pd.DataFrame()

    def load_table_data_simple(self, table_name: str, progress_callback=None) -> pd.DataFrame:
        """
        Load large datasets using simple chunked approach.

        Parameters:
        - table_name: name of table to load
        - progress_callback: optional callback function for progress updates

        Returns:
        - DataFrame with table data
        """
        try:
            # Get table info
            table_info = self.get_table_info(table_name)
            total_rows = table_info.get('row_count', 0)

            if total_rows == 0:
                return pd.DataFrame()

            # Use simple chunk size - 200K rows per chunk
            chunk_size = 200000

            if progress_callback:
                progress_callback(f"Loading {total_rows:,} rows in chunks of {chunk_size:,}...")

            # Load data in chunks
            chunks = []
            offset = 0
            chunk_num = 0

            while offset < total_rows:
                chunk_num += 1

                try:
                    chunk_df = self.load_table_data(table_name, limit=chunk_size, offset=offset)

                    if len(chunk_df) == 0:
                        break

                    chunks.append(chunk_df)
                    offset += chunk_size

                    # Update progress
                    progress_pct = min(100, (offset / total_rows) * 100)
                    if progress_callback:
                        progress_callback(f"Loaded chunk {chunk_num}: {len(chunk_df):,} rows ({progress_pct:.1f}%)")

                except Exception as e:
                    error_msg = str(e).lower()
                    if '502' in error_msg or 'timeout' in error_msg:
                        # For 502 errors, try with smaller chunk size
                        chunk_size = max(10000, chunk_size // 2)
                        if progress_callback:
                            progress_callback(f"Connection error, reducing chunk size to {chunk_size:,}")
                        continue
                    else:
                        raise e

            if not chunks:
                return pd.DataFrame()

            # Simple combination
            if progress_callback:
                progress_callback("Combining chunks...")

            final_df = pd.concat(chunks, ignore_index=True)

            if progress_callback:
                progress_callback(f"Successfully loaded {len(final_df):,} rows")

            return final_df

        except Exception as e:
            error_msg = str(e).lower()
            if '502' in error_msg or 'timeout' in error_msg:
                if progress_callback:
                    progress_callback("❌ Connection timeout - try reducing dataset size")
                st.error("❌ **502 Connection Error**")
                st.markdown("""
                **The dataset is too large for reliable loading:**

                🔧 **Try these solutions:**
                1. **Split your data** into smaller tables
                2. **Use data sampling** - load a subset first
                3. **Wait and try again** - server may be temporarily busy
                """)
            else:
                if progress_callback:
                    progress_callback(f"❌ Error: {str(e)}")
                st.error(f"❌ **Loading error:** {str(e)}")

            return pd.DataFrame()

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get information about a database table.

        Parameters:
        - table_name: name of table to analyze

        Returns:
        - Dictionary with table information
        """
        try:
            with self.engine.connect() as conn:
                # Get column information
                result = conn.execute(text(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = '{table_name}'
                    ORDER BY ordinal_position
                """))

                columns = []
                for row in result.fetchall():
                    columns.append({
                        'name': row[0],
                        'type': row[1],
                        'nullable': row[2] == 'YES'
                    })

                # Get row count
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                row_result = result.fetchone()
                row_count = row_result[0] if row_result is not None else 0

                return {
                    'table_name': table_name,
                    'columns': columns,
                    'row_count': row_count
                }

        except Exception as e:
            st.error(f"Error getting table info: {str(e)}")
            return {}

    def delete_table(self, table_name: str) -> bool:
        """
        Delete a table from the database.

        Parameters:
        - table_name: name of table to delete

        Returns:
        - Boolean indicating success
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
                conn.commit()

            st.success(f"Table '{table_name}' deleted successfully")
            return True

        except Exception as e:
            st.error(f"Error deleting table: {str(e)}")
            return False

    def _clean_table_name(self, name: str) -> str:
        """Clean table name for database compatibility."""
        # Replace spaces and special characters with underscores
        clean_name = "".join(c if c.isalnum() else "_" for c in name)
        # Remove consecutive underscores
        clean_name = "_".join(filter(None, clean_name.split("_")))
        # Ensure it starts with a letter
        if clean_name and clean_name[0].isdigit():
            clean_name = "table_" + clean_name
        # Limit length
        return clean_name[:50].lower()

    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for storage."""
        df_optimized = df.copy()

        for col in df_optimized.columns:
            if df_optimized[col].dtype == 'object':
                # Try to convert to numeric if possible
                try:
                    # Try to convert to numeric, avoiding deprecated warnings
                    converted = pd.to_numeric(df_optimized[col], errors='coerce')
                    # Keep original if too many values become null
                    if hasattr(converted, 'count') and hasattr(converted, '__len__'):
                        total_nulls = len(converted) - converted.count()
                        if total_nulls < len(df_optimized[col]) * 0.5:
                            df_optimized[col] = converted
                except Exception:
                    # Keep original column if conversion fails
                    pass

                # Convert to category if it has few unique values
                if df_optimized[col].dtype == 'object':
                    unique_ratio = df_optimized[col].nunique() / len(df_optimized)
                    if unique_ratio < 0.1:  # Less than 10% unique values
                        df_optimized[col] = df_optimized[col].astype('category')

        return df_optimized

    @safe_database_operation("CSV chunked import")
    def _import_csv_chunked(self, file, table_name: str) -> bool:
        """
        Import large CSV files using chunked processing to prevent memory overload.

        Parameters:
        - file: uploaded file object
        - table_name: name of table to create

        Returns:
        - Boolean indicating success
        """
        try:
            file.seek(0)

            # Determine optimal chunk size based on file size
            file_size_mb = file.size / (1024 * 1024)
            if file_size_mb > 1000:  # Very large files (>1GB)
                chunk_size = 10000
            elif file_size_mb > 500:  # Large files (>500MB)
                chunk_size = 20000
            elif file_size_mb > 100:  # Medium files (>100MB)
                chunk_size = 50000
            else:  # Smaller files
                chunk_size = 100000

            # Process CSV in chunks
            chunk_iter = pd.read_csv(file, chunksize=chunk_size)
            first_chunk = True
            total_rows = 0

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, chunk in enumerate(chunk_iter):
                # Skip adding metadata columns - data already has symbol/date structure
                # Just optimize data types
                chunk = self._optimize_dtypes(chunk)

                # Import chunk to database
                if_exists = 'replace' if first_chunk else 'append'
                chunk.to_sql(
                    table_name,
                    self.engine,
                    if_exists=if_exists,
                    index=False,
                    method='multi'
                )

                total_rows += len(chunk)
                first_chunk = False

                # Update progress
                if i % 10 == 0:  # Update every 10 chunks
                    estimated_total = (file_size_mb / 0.1) * chunk_size  # Rough estimate
                    progress = min(total_rows / estimated_total, 0.95)
                    progress_bar.progress(progress)
                    status_text.text(f"Processed {total_rows:,} rows...")

            # Complete progress
            progress_bar.progress(1.0)
            status_text.text(f"Successfully imported {total_rows:,} rows to table '{table_name}'")

            st.success(f"✅ CSV import completed: {total_rows:,} rows imported to '{table_name}'")
            return True

        except Exception as e:
            st.error(f"❌ Error during chunked CSV import: {str(e)}")
            return False