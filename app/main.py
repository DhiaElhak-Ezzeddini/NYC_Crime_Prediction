import streamlit as st
import folium
from streamlit_folium import st_folium
from datetime import datetime
import service
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Proj, transform
import requests
import plotly.graph_objects as go
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Page configuration
st.set_page_config(
    page_title="NYC Crime Risk Analyzer 🚔",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF6B6B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    .hero-banner {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #F093FB 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    .hero-banner h2 {
        color: white;
        font-size: 2.5rem;
        margin: 0.5rem 0;
        font-weight: 900;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        letter-spacing: 1px;
    }
    .hero-banner p {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.1rem;
        margin: 1rem 0 0 0;
        line-height: 1.6;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    .hero-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Helper functions
def get_coordinates(destination):
    """Get coordinates from location name using Nominatim"""
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": destination,
        "format": "json",
        "limit": 1,
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except:
        pass
    return None

def lon_lat_to_utm(lon, lat):
    """Convert lon/lat to UTM coordinates"""
    utm_proj = Proj(init="epsg:2263")
    utm_x, utm_y = transform(Proj(init="epsg:4326"), utm_proj, lon, lat)
    return utm_x, utm_y

def get_precinct_and_borough(lat, lon):
    """Get precinct and borough from coordinates"""
    try:
        shapefile = os.path.join(SCRIPT_DIR, 'shapes', 'geo_export_84578745-538d-401a-9cb5-34022c705879.shp')
        borough_sh = os.path.join(SCRIPT_DIR, 'borough', 'nybb.shp')
        
        precinct_gdf = gpd.read_file(shapefile)
        borough_gdf = gpd.read_file(borough_sh)
        
        point = Point(lon, lat)
        point2 = Point(lon_lat_to_utm(lon, lat))
        
        precinct = None
        borough = None
        
        for _, row in precinct_gdf.iterrows():
            if row['geometry'].contains(point):
                precinct = row['precinct']
                break
        
        for _, row in borough_gdf.iterrows():
            if row['geometry'].contains(point2):
                borough = row['BoroName']
                break
        
        return precinct, borough
    except Exception as e:
        st.warning(f"Could not load shapefiles: {e}")
        return None, None

def generate_base_map(default_location=[40.704467, -73.892246], default_zoom_start=11):
    """Generate base Folium map for NYC"""
    base_map = folium.Map(
        location=default_location, 
        control_scale=True, 
        zoom_start=default_zoom_start,
        min_zoom=11, 
        max_zoom=15,
        tiles='OpenStreetMap'
    )
    return base_map

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/New_York_City_Skyline_Illustration.jpg/800px-New_York_City_Skyline_Illustration.jpg", 
             use_container_width=True)
    st.markdown("# 🗽 NYC Crime Risk Analyzer")
    st.markdown("""
        ### 📊 About This Tool
        
        This application uses **machine learning** to predict crime risk in New York City based on:
        
        - 🕒 **Time & Date** - When you plan to visit
        - 📍 **Location** - Where in NYC you'll be
        - 👤 **Demographics** - Your personal information
        - 🏢 **Place Type** - Park, housing, station, etc.
        
        ### 🎯 How to Use
        
        1. **Click on the map** to select a location
        2. **Fill in your details** in the form
        3. **Click Submit** to get predictions
        
        ### ⚠️ Disclaimer
        
        This tool provides risk estimates based on historical data. Always stay aware of your surroundings and follow local safety guidelines.
    """)
    
    st.markdown("---")
    st.markdown("**Model Info:** XGBoost Classifier")
    st.markdown("**Data:** NYPD Crime Data (2008-2024)")

# Main content
st.markdown('<h1 class="main-header">🗽 NYC Crime Risk Analyzer 🚔</h1>', unsafe_allow_html=True)

# Introduction
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
        <div class="hero-banner">
            <div class="hero-icon">🛡️</div>
            <h2>Stay Informed, Stay Safe</h2>
            <p>Empower yourself with data-driven insights about crime risks in NYC. Make informed decisions about where and when you travel using advanced AI models trained on millions of historical crime records.</p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Map section
st.subheader("📍 Step 1: Select Your Location")
st.info("👆 Click anywhere on the map to choose your destination in NYC")

# Initialize session state for selected location
if 'selected_location' not in st.session_state:
    st.session_state.selected_location = None

# Render map
base_map = generate_base_map()
base_map.add_child(folium.LatLngPopup())

map_data = st_folium(base_map, height=450, width=None, key="main_map")

# Handle map clicks
if map_data and map_data.get('last_clicked'):
    lat = map_data['last_clicked']['lat']
    lon = map_data['last_clicked']['lng']
    
    # Get precinct and borough
    precinct, borough = get_precinct_and_borough(lat, lon)
    
    if borough and precinct:
        st.session_state.selected_location = {
            'lat': lat,
            'lon': lon,
            'precinct': precinct,
            'borough': borough
        }
        
        # Display selected location info
        st.success(f"✅ Location Selected: **{borough}** (Precinct: {precinct})")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latitude", f"{lat:.6f}")
        with col2:
            st.metric("Longitude", f"{lon:.6f}")
    else:
        st.warning("⚠️ Please select a location within NYC boundaries")

st.markdown("---")

# User information form
if st.session_state.selected_location:
    st.subheader("📋 Step 2: Enter Your Information")
    
    with st.form(key='prediction_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            # Demographics
            st.markdown("**👤 Demographics**")
            gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
            
            race = st.selectbox(
                "Race", 
                ['WHITE', 'WHITE HISPANIC', 'BLACK', 'ASIAN / PACIFIC ISLANDER', 
                 'BLACK HISPANIC', 'AMERICAN INDIAN/ALASKAN NATIVE', 'OTHER']
            )
            
            age = st.slider("Age", 0, 120, 25)
        
        with col2:
            # Time and Place
            st.markdown("**🕒 Time & Place**")
            date = st.date_input("Date", datetime.now())
            time = st.time_input("Time", datetime.now().time())
            hour = time.hour
            
            place = st.selectbox(
                "Location Type",
                ["In park", "In public housing", "In station", "On street"]
            )
        
        submit_button = st.form_submit_button("🔍 Predict Crime Risk", use_container_width=True)
    
    # Make prediction
    if submit_button:
        with st.spinner("🔄 Analyzing crime risk patterns..."):
            try:
                loc = st.session_state.selected_location
                
                # Create feature DataFrame
                X = service.create_df(
                    date=date,
                    hour=hour,
                    latitude=loc['lat'],
                    longitude=loc['lon'],
                    place=place,
                    age=age,
                    race=race,
                    gender=gender,
                    precinct=loc['precinct'],
                    borough=loc['borough']
                )
                
                # Get prediction
                pred_category, crime_types = service.predict(X)
                
                # Get probabilities if available
                probabilities = service.get_prediction_probability(X)
                
                st.markdown("---")
                st.subheader("📊 Prediction Results")
                
                # Main prediction box
                st.markdown(f"""
                    <div class="prediction-box">
                        <h2 style="text-align: center; margin-bottom: 1rem;">⚠️ Predicted Crime Category</h2>
                        <h1 style="text-align: center; font-size: 3rem; margin: 0;">{pred_category}</h1>
                    </div>
                """, unsafe_allow_html=True)
                
                # Crime types
                st.markdown("### 🔍 Possible Specific Crimes")
                cols = st.columns(3)
                for idx, crime in enumerate(crime_types[:6]):
                    with cols[idx % 3]:
                        st.markdown(f"• {crime}")
                
                # Probability chart
                if probabilities:
                    st.markdown("### 📈 Risk Distribution by Category")
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(probabilities.values()),
                            y=list(probabilities.keys()),
                            orientation='h',
                            marker=dict(
                                color=list(probabilities.values()),
                                colorscale='Reds',
                                showscale=True
                            )
                        )
                    ])
                    
                    fig.update_layout(
                        title="Crime Category Probabilities",
                        xaxis_title="Probability",
                        yaxis_title="Crime Category",
                        height=400,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Safety tips
                st.markdown("### 🛡️ Safety Recommendations")
                st.info(f"""
                    **Based on the predicted risk of {pred_category} crimes:**
                    
                    - Stay in well-lit, populated areas
                    - Keep valuables out of sight
                    - Be aware of your surroundings
                    - Trust your instincts
                    - Have emergency contacts readily available
                """)
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.exception(e)
else:
    st.info("👆 Please select a location on the map first")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>Made with ❤️ using Streamlit | Data: NYPD Historical Crime Records</p>
        <p>⚠️ This tool is for informational purposes only. Always exercise caution and follow official safety guidelines.</p>
    </div>
""", unsafe_allow_html=True)