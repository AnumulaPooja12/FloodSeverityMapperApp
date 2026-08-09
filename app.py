import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from data_processor import FloodDataProcessor
from map_generator import FloodMapGenerator
from alert_system import ThresholdBasedAlertSystem
from xgboost_model import FloodSeverityXGBoostModel

# Page configuration
st.set_page_config(
    page_title="India Flood Severity Monitoring",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #0f766e 0%, #14b8a6 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.5rem;
        font-weight: 600;
    }
    .main-header p {
        color: #ccfbf1;
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
    }
    .legend-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .severity-indicator {
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .alert-card {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .critical-alert {
        background: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = FloodDataProcessor()
if 'map_generator' not in st.session_state:
    st.session_state.map_generator = FloodMapGenerator()
if 'alert_system' not in st.session_state:
    st.session_state.alert_system = ThresholdBasedAlertSystem()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False
if 'auto_initialized' not in st.session_state:
    st.session_state.auto_initialized = False

# Auto-load data and make predictions on first run
if not st.session_state.auto_initialized:
    with st.spinner("Loading flood data..."):
        df = st.session_state.data_processor.load_sample_data()
        if df is not None and not df.empty:
            st.session_state.data_loaded = True
            
            # Auto-generate predictions
            model_info = st.session_state.data_processor.get_model_info()
            if model_info and model_info['is_trained']:
                predicted_df = st.session_state.data_processor.predict_severity(df)
                st.session_state.data_processor.sample_data = predicted_df
                st.session_state.predictions_made = True
            
            st.session_state.auto_initialized = True

# Header
st.markdown("""
    <div class="main-header">
        <h1>India Flood Severity Monitoring</h1>
        <p>Real-Time Interactive Map with Predictive Analytics & Alert System</p>
    </div>
""", unsafe_allow_html=True)

# Severity Legend
st.markdown("""
    <div class="legend-box" style="color: black;">
        <h3 style="margin-top:0;">📍 Severity Levels</h3>
        <p style="margin: 0.5rem 0;">
            <span class="severity-indicator" style="background-color: #10b981;"></span> <strong>Green:</strong> Safe / Low Risk
        </p>
        <p style="margin: 0.5rem 0;">
            <span class="severity-indicator" style="background-color: #f59e0b;"></span> <strong>Orange:</strong> Medium Risk - Monitor Closely
        </p>
        <p style="margin: 0.5rem 0;">
            <span class="severity-indicator" style="background-color: #eab308;"></span> <strong>Yellow:</strong> High Risk - Take Precautions
        </p>
        <p style="margin: 0.5rem 0;">
            <span class="severity-indicator" style="background-color: #ef4444;"></span> <strong>Red:</strong> Critical - Immediate Action Required
        </p>
    </div>
""", unsafe_allow_html=True)

## Sidebar for filters
with st.sidebar:
    st.markdown("## 🔍 Filters")
    st.markdown("---")
    
    if st.session_state.data_loaded:
        df = st.session_state.data_processor.sample_data

        # ----------------------------------------
        # EXTRACT RIVER NAME FROM LOCATION COLUMN
        # ----------------------------------------
        import re
        def extract_river_name(loc):
            match = re.search(r"\((.*?)\)", str(loc))
            return match.group(1) if match else None

        df["river_name"] = df["location"].apply(extract_river_name)

        # ----------------------------------------
        # FILTER ONLY TELANGANA RIVERS
        # ----------------------------------------
        telangana_rivers = [
            "Godavari", "Krishna", "Bhima", "Maner", "Musi",
            "Pranahita", "Penganga", "Kinnersani", "Munneru",
            "Manjira", "Peddavagu", "Kadam", "Dindi",
            "Badhrakali"
        ]

        df = df[df["river_name"].isin(telangana_rivers)]

        # ----------------------------------------
        # RIVER DROPDOWN
        # ----------------------------------------
        rivers = ['All Telangana Rivers'] + sorted(df["river_name"].dropna().unique().tolist())

        selected_river = st.selectbox(
            "Select River",
            rivers,
            index=0
        )

        # ----------------------------------------
        # STATE FILTER (OPTIONAL)
        # If you want to remove this also, tell me.
        # ----------------------------------------
        dataset_states = ["Madhya Pradesh", "Bihar", "Maharashtra", "Gujarat", "Uttar Pradesh", "Tamil Nadu", "Jharkhand", "Rajasthan", "Chhattisgarh", "Jammu and Kashmir", "Kerala", "Andhra Pradesh", "Assam", "West Bengal", "Karnataka", "Punjab", "Telangana", "Haryana", "Delhi", "Chandigarh"]

        states = ["All States"] + dataset_states
        selected_state = st.selectbox("Select State", states, index=0)



        # ----------------------------------------
        # SEVERITY FILTER
        # ----------------------------------------
        severity_levels = st.multiselect(
            "Show Severity Levels",
            options=['Low', 'Medium', 'High', 'Critical'],
            default=['Low', 'Medium', 'High', 'Critical']
        )

        # ----------------------------------------
        # DATE FILTER
        # ----------------------------------------
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            date_range = st.date_input(
                "Date Range",
                value=(df['date'].min(), df['date'].max())
            )

        st.markdown("---")
        st.markdown("""
            ### ℹ️ About
            This system monitors flood severity across **Telangana rivers** using:
            
            - Predictive modeling  
            - Automated alerts  
            - Geographic visualization
        """)
    else:
        st.info("Loading flood data...")



# Main content
if st.session_state.data_loaded:
    
    df = st.session_state.data_processor.sample_data
    
    # IMPORTANT — create filtered_df inside the block
    filtered_df = df.copy()

    # Telangana river filter
    if selected_river != "All Telangana Rivers":
        filtered_df = filtered_df[filtered_df["river_name"] == selected_river]

    # State filter
    if 'state' in df.columns and selected_state != 'All States':
        filtered_df = filtered_df[filtered_df['state'] == selected_state]

    # Severity filter
    severity_col = 'predicted_severity' if 'predicted_severity' in filtered_df.columns else 'severity_level'
    filtered_df = filtered_df[filtered_df[severity_col].isin(severity_levels)]

    # Date filter
    if 'date' in filtered_df.columns and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['date'] >= pd.Timestamp(start_date)) &
            (filtered_df['date'] <= pd.Timestamp(end_date))
        ]


    
    # Generate alerts
    alerts = st.session_state.alert_system.generate_alerts(filtered_df)
    alert_summary = st.session_state.alert_system.get_alert_summary(alerts)
    
    # Key metrics at the top
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📍 Total Locations", f"{len(filtered_df):,}")
    
    with col2:
        critical_count = len(filtered_df[filtered_df[severity_col] == 'Critical'])
        st.metric("🔴 Critical Alerts", f"{critical_count}", delta_color="inverse")
    
    with col3:
        high_count = len(filtered_df[filtered_df[severity_col] == 'High'])
        st.metric("🟡 High Risk Areas", f"{high_count}", delta_color="inverse")
    
    with col4:
        if 'population_affected' in filtered_df.columns:
            total_pop = filtered_df['population_affected'].sum()
            st.metric("👥 Population at Risk", f"{total_pop:,.0f}")
    
    st.markdown("---")
    
    # Main Interactive Map
    st.markdown("### 🗺️ Interactive India Flood Severity Map")
    st.info("👇 Click on any marker to view detailed flood information for that location")
    
    # Generate map with predictions
    use_predictions = st.session_state.predictions_made
    flood_map = st.session_state.map_generator.create_severity_map(
        filtered_df,
        use_predicted=use_predictions,
        show_alerts=True,
        alerts_data=alerts
    )
    
    if flood_map:
        st_folium(flood_map, width=None, height=700)
    
    st.markdown("---")
    
    # Quick Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Severity Distribution")
        severity_counts = filtered_df[severity_col].value_counts()
        fig = px.pie(
            values=severity_counts.values,
            names=severity_counts.index,
            color=severity_counts.index,
            color_discrete_map={
                'Low': '#10b981',
                'Medium': '#f59e0b',
                'High': '#eab308',
                'Critical': '#ef4444'
            },
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### ⚠️ Active Alerts")
        alert_data = pd.DataFrame({
            'Severity': ['Critical', 'High', 'Medium', 'Low'],
            'Alert Count': [
                alert_summary['critical_count'],
                alert_summary['high_count'],
                alert_summary['medium_count'],
                alert_summary['low_count']
            ]
        })
        fig = px.bar(
            alert_data,
            x='Severity',
            y='Alert Count',
            color='Severity',
            color_discrete_map={
                'Critical': '#ef4444',
                'High': '#eab308',
                'Medium': '#f59e0b',
                'Low': '#10b981'
            }
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Critical Alerts Section
    if alert_summary['critical_count'] > 0 or alert_summary['high_count'] > 0:
        st.markdown("---")
        st.markdown("### 🚨 Critical & High Priority Alerts")
        
        critical_alerts = st.session_state.alert_system.get_critical_alerts(alerts)
        
        for alert in critical_alerts[:5]:
            severity_emoji = '🔴' if alert['max_severity'] == 'critical' else '🟡'
            
            with st.expander(
                f"{severity_emoji} {alert['location']} - {alert['max_severity'].upper()} ({alert['alert_count']} threshold violations)",
                expanded=False
            ):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**State:** {alert.get('state', 'Unknown')}")
                    st.markdown(f"**Timestamp:** {alert['timestamp']}")
                
                with col2:
                    st.markdown(f"**Severity Level:** {alert['max_severity'].upper()}")
                    st.markdown(f"**Total Violations:** {alert['alert_count']}")
                
                st.markdown("#### Threshold Violations:")
                for alert_detail in alert['alerts']:
                    metric_name = alert_detail['metric'].replace('_', ' ').title()
                    st.markdown(f"""
                    - **{metric_name}**: {alert_detail['value']:.2f} (Safe Limit: {alert_detail['threshold']})
                      - *Recommendation: {alert_detail['recommendation']}*
                    """)
    
    # State-wise Summary
    if 'state' in filtered_df.columns:
        st.markdown("---")
        st.markdown("### 📍 State-wise Summary")
        state_summary = st.session_state.data_processor.get_state_summary(filtered_df)
        st.dataframe(state_summary, use_container_width=True, hide_index=True)
    
    # Environmental Metrics Visualization
    st.markdown("---")
    st.markdown("### 🌧️ Environmental Parameters Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'rainfall_mm' in filtered_df.columns and 'water_depth_m' in filtered_df.columns:
            st.markdown("#### Rainfall vs Water Depth")
            fig = px.scatter(
                filtered_df,
                x='rainfall_mm',
                y='water_depth_m',
                color=severity_col,
                size='population_affected' if 'population_affected' in filtered_df.columns else None,
                hover_data=['location'],
                color_discrete_map={
                    'Low': '#10b981',
                    'Medium': '#f59e0b',
                    'High': '#eab308',
                    'Critical': '#ef4444'
                }
            )
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'river_level_m' in filtered_df.columns and 'soil_moisture_percent' in filtered_df.columns:
            st.markdown("#### River Level vs Soil Moisture")
            fig = px.scatter(
                filtered_df,
                x='river_level_m',
                y='soil_moisture_percent',
                color=severity_col,
                hover_data=['location'],
                color_discrete_map={
                    'Low': '#10b981',
                    'Medium': '#f59e0b',
                    'High': '#eab308',
                    'Critical': '#ef4444'
                }
            )
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

else:
    # Loading screen
    st.markdown("""
    ## 👋 Welcome to India Flood Severity Monitoring System
    
    This interactive system provides:
    - **🗺️ Real-time flood severity mapping** across India
    - **🎯 Predictive analytics** for early warning
    - **⚠️ Automated alerts** when conditions exceed safe limits
    - **📊 Visual insights** with color-coded severity markers
    
    ### How it works:
    
    1. **Green markers** indicate safe zones with low flood risk
    2. **Orange markers** show medium risk areas requiring monitoring
    3. **Yellow markers** indicate high risk zones needing precautions
    4. **Red markers** represent critical areas requiring immediate action
    
    Click on any location marker to view detailed flood information including rainfall, water levels, soil moisture, and more.
    
    Loading data...
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #64748b; padding: 1rem;'>
        <p><strong>India Flood Severity Monitoring System</strong></p>
        <p>Powered by Predictive Analytics & Real-Time Alert System</p>
        <p style='font-size: 0.9rem;'>Advanced Disaster Management for Public Safety</p>
    </div>
""", unsafe_allow_html=True)
