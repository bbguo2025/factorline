# Factor Analysis Platform

## Overview

This is a comprehensive quantitative factor analysis system built with Streamlit that enables users to upload financial data, engineer technical features, construct portfolios, and analyze performance. The system provides a complete workflow for quantitative investment research with a user-friendly web interface.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit multi-page application
- **Layout**: Wide layout with sidebar navigation
- **Styling**: Custom CSS for enhanced visual appeal
- **Pages**: Four main pages (Data Upload, Factor Builder, Portfolio Builder, Results Dashboard)
- **State Management**: Streamlit session state for data persistence across pages

### Backend Architecture
- **Core Modules**: Four pre-built Python modules handling specific functionality
- **Data Processing**: Pandas-based data manipulation with MultiIndex support
- **Mathematical Operations**: Custom operations library for time-series and cross-sectional analysis
- **Portfolio Construction**: Long-short equity portfolio backtesting engine
- **Performance Analysis**: Comprehensive portfolio analytics with visualization

### Data Architecture
- **Input Formats**: Supports both wide format (date + symbol columns) and long format (date, symbol, value rows)
- **Data Storage**: In-memory processing using Pandas DataFrames
- **Index Structure**: MultiIndex (date, symbol) for efficient time-series operations
- **Feature Storage**: Technical indicators stored as additional DataFrame columns

## Key Components

### 1. Mathematical Operations (`operations.py`)
- **Time-Series Operations**: Rolling statistics, z-scores, differences, delays
- **Cross-Sectional Operations**: Ranking, winsorization, market neutralization
- **Advanced Functions**: Regression analysis, group neutralization
- **Mathematical Functions**: Logarithms, power functions, clipping

### 2. Portfolio Construction (`portfolio_constructer.py`)
- **Simulation Settings**: Configurable parameters for backtesting
- **Portfolio Weighting**: Equal weight and signal-weighted approaches
- **Transaction Costs**: Realistic cost modeling with turnover calculations
- **Risk Management**: Position limits and universe constraints

### 3. Performance Analysis (`portfolio_analyzer.py`)
- **PortfolioAnalyzer**: Comprehensive performance metrics calculator
- **Risk Metrics**: Sharpe ratio, Sortino ratio, maximum drawdown
- **Return Metrics**: Annualized returns, volatility, cumulative performance
- **Visualization**: Interactive charts using Plotly

### 4. Database Management (`database_manager.py`)
- **DatabaseManager**: PostgreSQL database integration with memory-optimized chunked processing
- **File Analysis**: Large file structure analysis without loading entire datasets
- **Table Management**: Create, view, and manage database tables with 502 error prevention
- **Data Optimization**: Automatic data type optimization and memory management

### 5. Memory Optimization (`memory_optimizer.py`)
- **MemoryOptimizer**: Real-time memory monitoring and resource management
- **Chunked Processing**: Intelligent chunk sizing for large file operations
- **Error Prevention**: 502 error prevention with automatic retry logic
- **Resource Monitoring**: Live CPU and memory usage tracking

### 6. Web Interface (`pages/`)
- **Data Management**: Large file upload with chunked processing and database storage
- **Factor Analysis Notebook**: Advanced factor combination using pre-computed technical features
- **Portfolio Simulation**: One-click backtesting with comprehensive performance analysis
- **Results Visualization**: Interactive charts and risk metrics dashboard

## Data Flow

1. **Data Ingestion**: Users upload large CSV files with pre-computed technical features
2. **Memory-Optimized Storage**: Files processed in chunks and stored in PostgreSQL database
3. **Data Validation**: Format checking, structure validation, and memory safety checks
4. **Factor Combination**: Advanced factor construction using existing technical features
5. **Portfolio Construction**: Long-short portfolios created using custom factor signals
6. **Backtesting**: Historical performance simulation with transaction costs and risk metrics
7. **Analysis**: Comprehensive performance visualization and portfolio analytics

## External Dependencies

### Core Libraries
- **streamlit**: Web application framework
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **plotly**: Interactive visualization
- **scipy**: Scientific computing
- **statsmodels**: Statistical modeling

### Visualization
- **matplotlib**: Static plotting (fallback)
- **plotly.express**: Simplified interactive charts
- **plotly.graph_objects**: Advanced chart customization

### Data Processing
- **datetime**: Date and time handling
- **pathlib**: File system operations
- **io**: Input/output operations

## Deployment Strategy

### Development Environment
- **Platform**: Replit with Python runtime
- **Package Management**: pip with requirements.txt
- **File Structure**: Modular organization with separate pages and utilities

### Production Considerations
- **Scalability**: In-memory processing suitable for datasets up to several GB
- **Performance**: Caching implemented for computationally intensive operations
- **Error Handling**: Comprehensive validation and user feedback
- **User Experience**: Progressive workflow with state persistence

### Configuration Management
- **Default Settings**: Centralized configuration in `config.py`
- **User Preferences**: Stored in Streamlit session state
- **Template Factors**: Pre-built factor expressions for quick start

## Recent Changes
- July 08, 2025: **Major 502 Error Fix for Large Datasets**
  - Enhanced database loading with three-tier approach: Standard, Smart Load, and Stream Load
  - Implemented memory-efficient chunked combination to prevent crashes during data merging
  - Added streaming approach using temporary files for extremely large datasets with many columns
  - Fixed syntax errors and improved error handling with specific 502 error detection
  - Added comprehensive loading method guide with memory limits and best practices
  - System now supports datasets up to 50M rows × 500+ columns using streaming approach
- July 02, 2025: Created new PostgreSQL database with working connection
- July 02, 2025: Merged Data Upload and Database Manager into unified Data Management page
- July 02, 2025: Implemented comprehensive data management with tabs for upload and database operations
- July 02, 2025: Added table preview, loading, and deletion functionality in single interface
- July 02, 2025: Enhanced error handling and connection status indicators
- July 02, 2025: Streamlined navigation to 2-page structure: Data Management + Factor Analysis Notebook
- July 02, 2025: Professional UI with consistent styling and real-time database status monitoring
- July 02, 2025: Fixed messy data management page layout and removed duplicate information sections
- July 02, 2025: Improved CSV processing with better chunking, encoding handling, and memory management
- July 02, 2025: Increased file size limit to 2GB to support large datasets (1.2GB+ files)
- July 02, 2025: Enhanced table management interface with cleaner actions and better user experience
- July 02, 2025: Completely redesigned Factor Analysis Notebook with simplified workflow
- July 02, 2025: Implemented single code cell for feature engineering with dropdown simulation settings
- July 02, 2025: Added one-click run button that executes complete analysis pipeline
- July 02, 2025: Streamlined interface: Load data → Code features → Configure settings → Run analysis
- July 02, 2025: Updated Factor Analysis Notebook to exact user specification with three-part structure
- July 02, 2025: Implemented factor generation code cell, simulation settings dropdown, and run button
- July 02, 2025: Added simulation_settings() and simulation() functions matching user's exact format
- July 02, 2025: Default factor code uses momentum_subindustry example with ts_decay and group_neutralize

### Major Analysis Engine Update (July 02, 2025)
- **Replaced Core Analysis Modules**: Updated operations.py, portfolio_constructer.py, and portfolio_analyzer.py to match user's exact local analysis workflow
- **Removed Unnecessary Metadata**: Eliminated unique symbol column requirement - data already contains symbol, date, and technical features
- **Implemented Real Portfolio Analysis**: Full portfolio construction with equal-weight and signal-weighted strategies
- **Added Complete Performance Visualization**: Interactive charts showing cumulative returns, turnover, and risk metrics
- **Integrated Contributor Analysis**: Top 10 long/short leg contributor identification for strategy attribution
- **Fixed Data Format Issues**: Proper MultiIndex (date, symbol) handling throughout the analysis pipeline
- **Enhanced Execution Environment**: Factor code execution with full access to data columns and technical operations

### Latest Major Improvements (July 02, 2025)
- **Enhanced Error Handling**: Added comprehensive error handler with retry logic and user-friendly messages
- **Performance Monitoring**: Real-time CPU and memory tracking with system status indicators
- **Database Optimization**: Fixed deprecated pandas warnings and improved connection pooling
- **Memory Management**: Intelligent chunk sizing and memory optimization with proactive warnings
- **User Experience**: Enhanced UI with animations, better status indicators, and progress tracking
- **Resilience**: Automatic retry mechanisms and graceful error recovery for network issues
- **System Intelligence**: Processing time estimation and resource-aware operation suggestions
- **Optimized Chunking**: Increased chunk sizes (10K-100K rows) to reduce processing overhead while maintaining 502 error prevention

### 502 Error Prevention System (July 02, 2025)
- **Memory Optimizer Module**: New dedicated memory management system to prevent resource overload
- **Chunked Database Import**: Completely rewritten CSV import to process large files in memory-efficient chunks
- **Pre-Upload Safety Checks**: Real-time memory validation before file processing begins
- **Enhanced Connection Pooling**: Optimized database timeouts and connection limits for large operations
- **Automatic Retry Logic**: Exponential backoff retry system for temporary 502 connection failures
- **Resource Monitoring**: Live memory usage warnings and intelligent chunk size optimization
- **Graceful Degradation**: System automatically adjusts processing parameters based on available resources

### Streamlined Architecture Update (July 02, 2025)
- **Removed Technical Feature Generation**: Eliminated `technical_features.py` as data already includes computed features
- **Unified Configuration**: Consolidated dual config files into single enhanced configuration system
- **Focused Factor Combination**: Updated workflow to emphasize factor combination rather than feature engineering
- **Updated Documentation**: Comprehensive updates to homepage, navigation guides, and system descriptions
- **Template Library**: New factor combination templates using pre-computed technical features
- **Simplified Workflow**: Streamlined from "upload → engineer → backtest" to "upload → combine → backtest"

## Changelog  
- July 01, 2025: Initial setup
- July 02, 2025: Major bug fixes and feature enhancements

## User Preferences

Preferred communication style: Simple, everyday language.