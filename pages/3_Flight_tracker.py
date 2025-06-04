
import requests
import json
import pandas as pd
import numpy as np
import geopandas as gpd
import contextily as ctx
import tzlocal
import pytz 
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
import warnings
warnings.filterwarnings('ignore')
from plotly.graph_objs import Marker
import plotly.express as px
import streamlit as st
from flight_tracker_data import flight_data

headers = {"Authorization": "Bearer "}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def query_flight_data(geo_df, question):
    table_data = {
        "icao24": geo_df["icao24"].astype(str).iloc[:100].tolist(),
        "callsign": geo_df["callsign"].astype(str).replace({np.nan: None, np.inf: '0'}).iloc[:100].tolist(),
        "origin_country": geo_df["origin_country"].astype(str).replace({np.nan: None, np.inf: '0'}).iloc[:100].tolist(),
        "time_position": geo_df["time_position"].astype(str).replace({np.nan: '0', np.inf: '0'}).iloc[:100].tolist(),
        "last_contact": geo_df["last_contact"].astype(str).replace({np.nan: '0', np.inf: '0'}).iloc[:100].tolist(),
        "longitude": geo_df["longitude"].astype(str).replace({np.nan: '0', np.inf: '0'}).iloc[:100].tolist(),
        "latitude": geo_df["latitude"].astype(str).replace({np.nan: '0', np.inf: '0'}).iloc[:100].tolist(),
        "baro_altitude": geo_df["baro_altitude"].astype(str).replace({np.nan: '0', np.inf: '0'}).iloc[:100].tolist(),
        "on_ground": geo_df["on_ground"].astype(str).iloc[:100].tolist(),  # Assuming on_ground is boolean or categorical
        "velocity": geo_df["velocity"].astype(str).replace({np.nan: '0', np.inf: '0'}).iloc[:100].tolist(),
        "true_track": geo_df["true_track"].astype(str).replace({np.nan: '0', np.inf: '0'}).iloc[:100].tolist(),
        "vertical_rate": geo_df["vertical_rate"].astype(str).replace({np.nan: '0', np.inf: '0'}).iloc[:100].tolist(),
        "sensors": geo_df["sensors"].astype(str).replace({np.nan: None, np.inf: '0'}).iloc[:100].tolist(), # Assuming sensors can be None
        "geo_altitude": geo_df["geo_altitude"].astype(str).replace({np.nan: '0', np.inf: '0'}).iloc[:100].tolist(),
        "squawk": geo_df["squawk"].astype(str).replace({np.nan: None, np.inf: '0'}).iloc[:100].tolist(), # Assuming squawk can be None
        "spi": geo_df["spi"].astype(str).iloc[:100].tolist(),  # Assuming spi is boolean or categorical
        "position_source": geo_df["position_source"].astype(str).iloc[:100].tolist(),  # Assuming position_source is categorical
        "time": geo_df["time"].astype(str).replace({np.nan: '0', np.inf: '0'}).iloc[:100].tolist(),
        "geometry": geo_df["geometry"].astype(str).replace({np.nan: None, np.inf: '0'}).iloc[:100].tolist() # Assuming geometry can be None
    }

    # Construct the payload
    payload = {
        "inputs": {
            "query": question,
            "table": table_data,
        }
    }

    # Get the model response
    response = query(payload)

    # Check if 'answer' is in response and return it as a sentence
    if 'answer' in response:
        answer = response['answer']
        return f"The answer to your question '{question}': :orange[{answer}]"
    else:
        return "The model could not find an answer to your question."

def flight_tracking(flight_view_level, country, local_time_zone, flight_info, airport, color):
    geolocator = Nominatim(user_agent="flight_tracker")
    loc = geolocator.geocode(country)
    loc_box = loc[1]
    extend_left =+12*flight_view_level
    extend_right =+10*flight_view_level
    extend_top =+10*flight_view_level
    extend_bottom =+ 18*flight_view_level
    lat_min, lat_max = (loc_box[0] - extend_left), loc_box[0]+extend_right
    lon_min, lon_max = (loc_box[1] - extend_bottom), loc_box[1]+extend_top
    
    tile_zoom = 8 # zoom of the map loaded by contextily
    figsize = (15, 15)
    columns = ["icao24","callsign","origin_country","time_position","last_contact","longitude","latitude",
            "baro_altitude","on_ground","velocity","true_track","vertical_rate","sensors","geo_altitude",
            "squawk","spi","position_source",]
    data_url = "https://raw.githubusercontent.com/ashok2216-A/ashok_airport-data/main/data/airports.dat"
    column_names = ["Airport ID", "Name", "City", "Country", "IATA/FAA", "ICAO", "Latitude", "Longitude",
                    "Altitude", "Timezone", "DST", "Tz database time zone", "Type", "Source"]
    airport_df = pd.read_csv(data_url, header=None, names=column_names)
    airport_locations = airport_df[["Name", "City", "Country", "IATA/FAA", "Latitude", "Longitude"]]
    airport_country_loc = airport_locations[airport_locations['Country'] == str(loc)]
    airport_country_loc = airport_country_loc[(airport_country_loc['Country'] == str(loc)) & (airport_country_loc['Latitude'] >= lat_min) &
                            (airport_country_loc['Latitude'] <= lat_max) & (airport_country_loc['Longitude'] >= lon_min) &
                            (airport_country_loc['Longitude'] <= lon_max)]
    def get_traffic_gdf():
        url_data = (
                f"https://@opensky-network.org/api/states/all?"
                f"lamin={str(lat_min)}"
                f"&lomin={str(lon_min)}"
                f"&lamax={str(lat_max)}"
                f"&lomax={str(lon_max)}")
        json_dict = requests.get(url_data).json()

        unix_timestamp = int(json_dict["time"])
        local_timezone = pytz.timezone(local_time_zone) # get pytz timezone
        local_time = datetime.fromtimestamp(unix_timestamp, local_timezone).strftime('%Y-%m-%d %H:%M:%S')
        time = []
        for i in range(len(json_dict['states'])):
            time.append(local_time)
        df_time = pd.DataFrame(time,columns=['time'])
        state_df = pd.DataFrame(json_dict["states"],columns=columns)
        state_df['time'] = df_time
        gdf = gpd.GeoDataFrame(
                state_df,
                geometry=gpd.points_from_xy(state_df.longitude, state_df.latitude),
                crs={"init": "epsg:4326"},  # WGS84
            )
        # banner_image = Image.open('banner.png')
        # st.image(banner_image, width=300)
        st.title("Live Flight Tracker")
        st.subheader('Flight Details', divider='rainbow')
        st.write('Location: {0}'.format(loc))
        st.write('Current Local Time: {0}-{1}:'.format(local_time, local_time_zone))
        st.write("Minimum_latitude is {0} and Maximum_latitude is {1}".format(lat_min, lat_max))
        st.write("Minimum_longitude is {0} and Maximum_longitude is {1}".format(lon_min, lon_max))
        st.write('Number of Visible Flights: {}'.format(len(json_dict['states'])))
        st.write('Plotting the flight: {}'.format(flight_info))
        st.subheader('Map Visualization', divider='rainbow')
        st.write('****Click ":orange[Update Map]" Button to Refresh the Map****')
        return gdf

    geo_df = get_traffic_gdf()
    if airport == 0:
        fig = px.scatter_mapbox(geo_df, lat="latitude", lon="longitude",color=flight_info,
                            color_continuous_scale=color, zoom=4,width=1200, height=600,opacity=1,
                            hover_name ='origin_country',hover_data=['callsign', 'baro_altitude',
        'on_ground', 'velocity', 'true_track', 'vertical_rate', 'geo_altitude'], template='plotly_dark')
    elif airport == 1:
        fig = px.scatter_mapbox(geo_df, lat="latitude", lon="longitude",color=flight_info,
                            color_continuous_scale=color, zoom=4,width=1200, height=600,opacity=1,
                            hover_name ='origin_country',hover_data=['callsign', 'baro_altitude',
        'on_ground', 'velocity', 'true_track', 'vertical_rate', 'geo_altitude'], template='plotly_dark')
        fig.add_trace(px.scatter_mapbox(airport_country_loc, lat="Latitude", lon="Longitude",
                                        hover_name ='Name', hover_data=["City", "Country", "IATA/FAA"]).data[0])
    else: None
    fig.update_layout(mapbox_style="carto-darkmatter")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    out = st.plotly_chart(fig, theme=None)
    return out

st.set_page_config(
    layout="wide"
)

# Configuration options (previously in sidebar) now in main component
st.subheader("Configure Map", divider='rainbow')
Refresh = st.button('Update Map', key=1)
on = st.toggle('View Airports')
if on:
    air_port = 1
    st.write(':rainbow[Nice Work Buddy!]')
    st.write('Now Airports are Visible')
else:
    air_port = 0
view = st.slider('Increase Flight Visibility', 1, 6, 2)
st.write("You Selected:", view)
cou = st.text_input('Type Country Name', 'north america')
st.write('The current Country name is', cou)
time = st.text_input('Type Time Zone Name (Ex: America/Toronto, Europe/Berlin)', 'Asia/Kolkata')
st.write('The current Time Zone is', time)
info = st.selectbox(
    'Select Flight Information',
    ('baro_altitude', 'on_ground', 'velocity', 'geo_altitude'))
st.write('Plotting the data of Flight:', info)
clr = st.radio('Pick A Color for Scatter Plot', ["rainbow", "ice", "hot"])
if clr == "rainbow":
    st.write('The current color is', "****:rainbow[Rainbow]****")
elif clr == 'ice':
    st.write('The current color is', "****:blue[Ice]****")
elif clr == 'hot':
    st.write('The current color is', "****:red[Hot]****")
else:
    None

# Call the flight tracking function
flight_tracking(flight_view_level=view, country=cou, flight_info=info,
                local_time_zone=time, airport=air_port, color=clr)
