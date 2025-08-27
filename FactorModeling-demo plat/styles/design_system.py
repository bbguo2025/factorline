"""
Unified Design System for Factor Analysis Platform
Professional styling components and CSS utilities
"""

import streamlit as st

class DesignSystem:
    """Centralized design system with consistent styling across all pages"""

    # Color palette
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e', 
        'success': '#2ca02c',
        'warning': '#d62728',
        'info': '#17a2b8',
        'light': '#f8f9fa',
        'dark': '#343a40',
        'white': '#ffffff',
        'border': '#e1e5e9',
        'shadow_light': 'rgba(0,0,0,0.08)',
        'shadow_medium': 'rgba(0,0,0,0.12)',
        'shadow_primary': 'rgba(31, 119, 180, 0.3)',
    }

    # Typography
    FONTS = {
        'main': "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        'mono': "'JetBrains Mono', 'Monaco', 'Menlo', monospace",
        'display': "'Inter', system-ui, sans-serif"
    }

    @staticmethod
    def inject_global_styles():
        """Inject global CSS styles for consistent design"""
        st.markdown(f"""
        <style>
            /* Import modern fonts */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

            /* CSS Custom Properties */
            :root {{
                --color-primary: {DesignSystem.COLORS['primary']};
                --color-secondary: {DesignSystem.COLORS['secondary']};
                --color-success: {DesignSystem.COLORS['success']};
                --color-warning: {DesignSystem.COLORS['warning']};
                --color-info: {DesignSystem.COLORS['info']};
                --color-light: {DesignSystem.COLORS['light']};
                --color-dark: {DesignSystem.COLORS['dark']};
                --color-white: {DesignSystem.COLORS['white']};
                --color-border: {DesignSystem.COLORS['border']};
                --shadow-light: {DesignSystem.COLORS['shadow_light']};
                --shadow-medium: {DesignSystem.COLORS['shadow_medium']};
                --shadow-primary: {DesignSystem.COLORS['shadow_primary']};
                --font-main: {DesignSystem.FONTS['main']};
                --font-mono: {DesignSystem.FONTS['mono']};
                --font-display: {DesignSystem.FONTS['display']};
                --border-radius: 12px;
                --border-radius-sm: 8px;
                --spacing-xs: 0.5rem;
                --spacing-sm: 1rem;
                --spacing-md: 1.5rem;
                --spacing-lg: 2rem;
                --spacing-xl: 3rem;
            }}

            /* Global typography improvements */
            .stApp {{
                font-family: var(--font-main);
                background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                min-height: 100vh;
            }}

            /* Enhanced headers */
            h1, h2, h3, h4, h5, h6 {{
                font-family: var(--font-display);
                font-weight: 600;
                letter-spacing: -0.025em;
                line-height: 1.2;
            }}

            h1 {{ font-size: 2.5rem; }}
            h2 {{ font-size: 2rem; }}
            h3 {{ font-size: 1.5rem; }}

            /* Improved button styling */
            .stButton > button {{
                background: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
                color: white;
                border: none;
                border-radius: var(--border-radius-sm);
                padding: 0.75rem 1.5rem;
                font-weight: 500;
                font-family: var(--font-main);
                transition: all 0.3s ease;
                box-shadow: 0 4px 12px var(--shadow-light);
            }}

            .stButton > button:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 20px var(--shadow-medium);
            }}

            /* Enhanced selectbox styling */
            .stSelectbox > div > div {{
                border-radius: var(--border-radius-sm);
                border: 2px solid var(--color-border);
                transition: border-color 0.3s ease;
            }}

            .stSelectbox > div > div:focus-within {{
                border-color: var(--color-primary);
                box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.1);
            }}

            /* Code block improvements */
            .stCodeBlock {{
                border-radius: var(--border-radius-sm);
                border: 1px solid var(--color-border);
                box-shadow: 0 2px 8px var(--shadow-light);
            }}

            /* Text area improvements */
            .stTextArea > div > div > textarea {{
                font-family: var(--font-mono);
                border-radius: var(--border-radius-sm);
                border: 2px solid var(--color-border);
                transition: border-color 0.3s ease;
            }}

            .stTextArea > div > div > textarea:focus {{
                border-color: var(--color-primary);
                box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.1);
            }}

            /* Progress bar styling */
            .stProgress > div > div > div {{
                background: linear-gradient(90deg, var(--color-primary), var(--color-secondary));
                border-radius: 10px;
            }}

            /* Sidebar improvements */
            .css-1d391kg {{
                background: linear-gradient(180deg, var(--color-white) 0%, var(--color-light) 100%);
                border-right: 1px solid var(--color-border);
            }}

            /* Metric improvements */
            [data-testid="metric-container"] {{
                background: var(--color-white);
                border: 1px solid var(--color-border);
                border-radius: var(--border-radius-sm);
                padding: var(--spacing-sm);
                box-shadow: 0 2px 8px var(--shadow-light);
            }}

            /* Tab styling */
            .stTabs [data-baseweb="tab-list"] {{
                background: var(--color-white);
                border-radius: var(--border-radius-sm);
                padding: 0.25rem;
                border: 1px solid var(--color-border);
            }}

            .stTabs [data-baseweb="tab"] {{
                border-radius: var(--border-radius-sm);
                padding: 0.75rem 1.5rem;
                font-weight: 500;
            }}

            .stTabs [aria-selected="true"] {{
                background: var(--color-primary);
                color: white;
            }}

            /* Expander improvements */
            .streamlit-expander {{
                border: 1px solid var(--color-border);
                border-radius: var(--border-radius-sm);
                box-shadow: 0 2px 8px var(--shadow-light);
                margin: var(--spacing-sm) 0;
            }}

            /* Alert improvements */
            .stAlert {{
                border-radius: var(--border-radius-sm);
                border: none;
                box-shadow: 0 4px 12px var(--shadow-light);
            }}
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def create_page_header(title: str, description: str, icon: str = "📊"):
        """Create a consistent page header"""
        st.markdown(f"""
        <div class="page-header">
            <div class="header-content">
                <h1><span class="header-icon">{icon}</span> {title}</h1>
                <p class="header-description">{description}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Add header-specific CSS
        st.markdown("""
        <style>
            .page-header {
                background: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
                padding: var(--spacing-xl) var(--spacing-lg);
                border-radius: var(--border-radius);
                margin-bottom: var(--spacing-lg);
                color: white;
                text-align: center;
                box-shadow: 0 8px 32px var(--shadow-primary);
                position: relative;
                overflow: hidden;
            }

            .page-header::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 50%);
                pointer-events: none;
            }

            .header-content {
                position: relative;
                z-index: 1;
            }

            .page-header h1 {
                margin: 0;
                font-size: 3rem;
                font-weight: 700;
                text-shadow: 0 2px 4px rgba(0,0,0,0.3);
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 1rem;
            }

            .header-icon {
                font-size: 3rem;
                filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
            }

            .header-description {
                margin: 1rem 0 0 0;
                font-size: 1.25rem;
                opacity: 0.95;
                font-weight: 400;
                text-shadow: 0 1px 2px rgba(0,0,0,0.2);
            }
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def create_section_card(title: str, content_func, expanded: bool = True, icon: str = ""):
        """Create a consistent section card"""
        with st.expander(f"{icon} {title}" if icon else title, expanded=expanded):
            st.markdown('<div class="section-content">', unsafe_allow_html=True)
            content_func()
            st.markdown('</div>', unsafe_allow_html=True)

    @staticmethod
    def create_metric_grid(metrics: dict, columns: int = 4):
        """Create a responsive metric grid"""
        cols = st.columns(columns)
        for i, (label, value) in enumerate(metrics.items()):
            with cols[i % columns]:
                st.metric(label, value)

    @staticmethod
    def create_status_indicator(status: str, message: str):
        """Create a status indicator with consistent styling"""
        color_map = {
            'success': 'var(--color-success)',
            'warning': 'var(--color-warning)', 
            'error': 'var(--color-warning)',
            'info': 'var(--color-info)'
        }

        icon_map = {
            'success': '✅',
            'warning': '⚠️',
            'error': '❌',
            'info': 'ℹ️'
        }

        color = color_map.get(status, 'var(--color-info)')
        icon = icon_map.get(status, 'ℹ️')

        st.markdown(f"""
        <div class="status-indicator status-{status}">
            <span class="status-icon">{icon}</span>
            <span class="status-message">{message}</span>
        </div>
        """, unsafe_allow_html=True)

        # Add status indicator CSS
        st.markdown(f"""
        <style>
            .status-indicator {{
                display: inline-flex;
                align-items: center;
                gap: var(--spacing-xs);
                padding: var(--spacing-xs) var(--spacing-sm);
                border-radius: 25px;
                font-weight: 500;
                margin: var(--spacing-xs) 0;
                border: 2px solid {color};
                background: {color}15;
                color: {color};
            }}

            .status-icon {{
                font-size: 1.1em;
            }}

            .status-message {{
                font-size: 0.9rem;
            }}
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def create_upload_zone(label: str = "Upload your file"):
        """Create an enhanced upload zone"""
        st.markdown(f"""
        <div class="upload-zone">
            <div class="upload-icon">📁</div>
            <h3>{label}</h3>
            <p>Drag and drop your file here, or click to browse</p>
            <small>Supports CSV files up to 2GB</small>
        </div>
        """, unsafe_allow_html=True)

        # Add upload zone CSS
        st.markdown("""
        <style>
            .upload-zone {
                background: linear-gradient(135deg, var(--color-light), #dee2e6);
                padding: var(--spacing-xl) var(--spacing-lg);
                border-radius: var(--border-radius);
                border: 3px dashed var(--color-primary);
                margin: var(--spacing-lg) 0;
                text-align: center;
                transition: all 0.3s ease;
                cursor: pointer;
            }

            .upload-zone:hover {
                border-color: var(--color-secondary);
                background: linear-gradient(135deg, #dee2e6, #ced4da);
                transform: translateY(-2px);
                box-shadow: 0 8px 24px var(--shadow-medium);
            }

            .upload-icon {
                font-size: 4rem;
                margin-bottom: var(--spacing-sm);
                opacity: 0.8;
            }

            .upload-zone h3 {
                color: var(--color-primary);
                margin: var(--spacing-sm) 0;
                font-weight: 600;
            }

            .upload-zone p {
                color: var(--color-dark);
                margin: var(--spacing-xs) 0;
                opacity: 0.8;
            }

            .upload-zone small {
                color: var(--color-dark);
                opacity: 0.6;
            }
        </style>
        """, unsafe_allow_html=True)