"""
Main entry point for the Factor Analysis Platform
Platform Overview - Professional quantitative research platform
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import time
import psutil
import os
import sys

# Import design system components
from styles.design_system import DesignSystem
from performance_monitor import PerformanceMonitor

# Enhanced page configuration
st.set_page_config(
    page_title="Factor Analysis Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global design system
DesignSystem.inject_global_styles()

# Initialize session state with improvements
session_defaults = {
    'data_uploaded': False,
    'features_computed': False,
    'portfolio_built': False,
    'last_activity': time.time(),
    'performance_metrics': {}
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

# System status monitoring
def display_system_monitor():
    """Display enhanced system monitoring"""
    metrics = PerformanceMonitor.get_system_metrics()

    with st.sidebar:
        st.markdown("### 📊 System Monitor")

        # Memory usage with color coding
        mem_color = "🔴" if metrics['memory_percent'] > 80 else "🟡" if metrics['memory_percent'] > 60 else "🟢"
        st.markdown(f"{mem_color} **Memory:** {metrics['memory_percent']:.1f}%")
        st.progress(metrics['memory_percent'] / 100)

        # CPU usage
        cpu_color = "🔴" if metrics['cpu_percent'] > 80 else "🟡" if metrics['cpu_percent'] > 60 else "🟢"
        st.markdown(f"{cpu_color} **CPU:** {metrics['cpu_percent']:.1f}%")
        st.progress(metrics['cpu_percent'] / 100)

        # Available memory warning
        if metrics['available_memory_gb'] < 1:
            st.warning(f"⚠️ Low memory: {metrics['available_memory_gb']:.1f} GB")
        else:
            st.info(f"💾 Available: {metrics['available_memory_gb']:.1f} GB")

# Display system monitoring
display_system_monitor()

# Create professional page header using design system
DesignSystem.create_page_header(
    title="Quantitative Factor Analysis Platform",
    description="Professional systematic trading strategy development and analysis",
    icon="🏛️"
)

# Performance status indicator
metrics = PerformanceMonitor.get_system_metrics()
status_type = "success" if metrics['memory_percent'] < 70 else "warning" if metrics['memory_percent'] < 85 else "error"
status_msg = f"System Status: {'Optimal' if status_type == 'success' else 'Moderate' if status_type == 'warning' else 'High Usage'} | Memory: {metrics['memory_percent']:.1f}% | CPU: {metrics['cpu_percent']:.1f}%"

DesignSystem.create_status_indicator(status_type, status_msg)

# Platform overview section
st.markdown("## 🎯 Platform Overview")

st.write("""
This is a professional-grade quantitative research platform designed for systematic trading strategy development. 
The platform processes pre-computed technical features and focuses on advanced factor combination, portfolio construction, 
and performance analysis to support institutional-quality investment research.
""")

# Key capabilities section
st.markdown("### ✨ Key Capabilities")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **🗄️ Data Management**
    - Large file support (up to 2GB)
    - PostgreSQL database integration  
    - Memory-optimized chunked processing
    - Automated data validation and optimization

    **🧮 Factor Engineering**
    - Pre-computed technical features
    - Advanced factor combination techniques
    - Cross-sectional and time-series operations
    - Market neutralization and group ranking
    """)

with col2:
    st.markdown("""
    **📊 Portfolio Construction**
    - Long-short equity strategies
    - Equal weight and signal-weighted approaches
    - Transaction cost modeling
    - Risk management and position limits

    **📈 Performance Analysis**
    - Comprehensive backtesting engine
    - Risk metrics (Sharpe, Sortino, Max Drawdown)
    - Interactive visualization with Plotly
    - Attribution analysis and top contributors
    """)

# Workflow section
st.markdown("### 🚀 Workflow")

workflow_steps = [
    {
        "title": "Upload Data",
        "description": "Upload large CSV files with pre-computed technical features to PostgreSQL database with memory-optimized processing",
        "page": "Data Management"
    },
    {
        "title": "Engineer Factors", 
        "description": "Combine existing technical features using advanced mathematical operations and cross-sectional analysis",
        "page": "Factor Analysis Notebook"
    },
    {
        "title": "Analyze Performance",
        "description": "Run comprehensive backtests with transaction costs, risk metrics, and interactive visualizations",
        "page": "Factor Analysis Notebook"
    }
]

for i, step in enumerate(workflow_steps, 1):
    with st.container():
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #f8f9fa, #e9ecef);
            border: 2px solid #e1e5e9;
            border-radius: 15px;
            padding: 2rem;
            margin: 1.5rem 0;
            position: relative;
        ">
            <div style="
                position: absolute;
                top: -15px;
                left: 30px;
                background: linear-gradient(135deg, #1f77b4, #ff7f0e);
                color: white;
                width: 30px;
                height: 30px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: bold;
            ">{i}</div>
            <h4 style="color: #1f77b4; margin-top: 0;">{step['title']}</h4>
            <p style="margin-bottom: 0.5rem;">{step['description']}</p>
            <small style="color: #6c757d;">📄 Available in: {step['page']} page</small>
        </div>
        """, unsafe_allow_html=True)

# System metrics
st.markdown("### 📊 System Metrics")

# Create metrics using design system
DesignSystem.create_metric_grid({
    f"Memory Usage": f"{metrics['memory_percent']:.1f}%",
    f"CPU Usage": f"{metrics['cpu_percent']:.1f}%", 
    f"Available Memory": f"{metrics['available_memory_gb']:.1f} GB",
    f"Session Status": "Active"
})

# Navigation guide with enhanced design
st.markdown("### 🧭 Quick Start Guide")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border: 2px solid #1f77b4;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        position: relative;
        transition: all 0.3s ease;
    ">
        <div style="
            position: absolute;
            top: -15px;
            left: 30px;
            background: linear-gradient(135deg, #1f77b4, #ff7f0e);
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        ">1</div>
        <h4 style="color: #1f77b4; margin-top: 0;">📊 Data Management</h4>
        <p>Upload your CSV files with pre-computed technical features. The system supports files up to 2GB with automatic chunked processing.</p>
        <small style="color: #6c757d;">Required: CSV with date, symbol, and technical indicators</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        border: 2px solid #1f77b4;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        position: relative;
        transition: all 0.3s ease;
    ">
        <div style="
            position: absolute;
            top: -15px;
            left: 30px;
            background: linear-gradient(135deg, #1f77b4, #ff7f0e);
            color: white;
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
        ">2</div>
        <h4 style="color: #1f77b4; margin-top: 0;">📝 Factor Analysis</h4>
        <p>Create custom factors using advanced mathematical operations, configure simulation settings, and run comprehensive backtests.</p>
        <small style="color: #6c757d;">One-click analysis with professional risk metrics</small>
    </div>
    """, unsafe_allow_html=True)

st.info("💡 **Session Persistence**: Your uploaded data and computed factors remain available across all pages during your analysis session.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; padding: 2rem;">
    <strong>Factor Analysis Platform</strong> - Professional quantitative research environment<br>
    Built for systematic trading strategy development and institutional-quality analysis
</div>
""", unsafe_allow_html=True)