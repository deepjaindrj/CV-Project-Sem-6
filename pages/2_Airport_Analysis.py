import streamlit as st

import insights_db
from datetime import timedelta, date, datetime
from plotly.tools import FigureFactory as FF
import plotly.express as px
import pandas as pd
import plotly.figure_factory as plotly_ff

st.set_page_config(layout="wide", initial_sidebar_state='expanded',
                   page_icon="images/Aircraft Intelligence Hub Badge.ico")


def display_statistics(airport, start_time, end_time):  # displays 3 statistics value for airport
    col1, col2, col3 = st.columns(3)

    # Total images over airport and added last 30 days
    total_images, added_this_month = insights_db.total_images_over_airport(airport)
    col1.metric("Total Images", value=f"{total_images}", delta=f"{added_this_month} 30 days")

    last_image_date, last_image_aircrafts = insights_db.last_image_over_airport(airport)
    formatted_last_date = last_image_date.strftime("%Y-%m-%d") if last_image_date else None
    col2.metric("Latest Image", value=f"{formatted_last_date}", delta=f"{last_image_aircrafts} Aircrafts", delta_color="off")

    start_date = datetime(start_time.year, start_time.month, start_time.day, 0, 0, 0)
    end_date = datetime(end_time.year, end_time.month, end_time.day, 0, 0, 0)

    avg_aircrafts_per_image, avg_images_per_month = insights_db.averages_over_airport(airport, end_date, start_date)
    col3.metric("Averages", f"{avg_aircrafts_per_image} Aircraft/Img", f"{avg_images_per_month} Images/Month")


def display_scatter(airport, start_date, end_date):  # displays scatter graph
    months_labels = []
    current_date = start_date
    while current_date <= end_date:
        month = current_date.strftime("%Y-%m")
        current_date += timedelta(days=32 - current_date.day)
        months_labels.append(month)

    histogram_data = insights_db.scatter_data(airport, start_date, end_date)

    if not histogram_data or not months_labels:
        st.warning("No data available for the selected period.")
        return

    try:
        # Check for empty sublists in histogram_data
        if any(not data for data in histogram_data):
            st.warning("Some data is missing. Unable to create distribution plot.")
            return

        # Filter out empty sublists before creating the plot
        non_empty_data = [data for data in histogram_data if data]
      
        # fig = FF.create_distplot(non_empty_data, months_labels, bin_size=5)
        # fig = plotly.figure_factory.create_distplot(non_empty_data, months_labels, bin_size=5)
        fig = plotly_ff.create_distplot(non_empty_data, months_labels, bin_size=5)
        fig.update_layout(title_text="Image Distribution", xaxis_title="Month", yaxis_title="Aircrafts Count",)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating distribution plot: {e}")


def display_tabs(cleaned_airport):  # displays all 3 tabs
    tab1, tab2, tab3 = st.tabs(["Weekly Graph", "Ranking", "Over Threshold"])
    with tab1:  # daily and hourly distribution graph tab
        x_label = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        y_label = ["07:00-08:00", "08:00-09:00", "09:00-10:00", "10:00-11:00", "11:00-12:00",
                   "12:00-13:00", "13:00-14:00", "14:00-15:00", "15:00-16:00", "16:00-17:00"]
        x, y = insights_db.get_daily_distribution(cleaned_airport)
        df = pd.DataFrame({'Day': [day for day in x_label for _ in range(len(y_label))],
                           'Time': y_label * len(x_label),
                           'Images': [val for day_values in y for val in day_values]})

        # Create the heatmap with a smoother color scale (Viridis)
        fig = px.imshow(df.pivot(index='Time', columns='Day', values='Images'),
                        labels={'x': 'Day of Week', 'y': 'Time'},
                        title='Image Count Heatmap',
                        color_continuous_scale='RdBu_r', text_auto=True)
        st.plotly_chart(fig)

    with tab2:  # ranking tab
        all_ranks_data, rank = insights_db.get_airport_rank(cleaned_airport)
        airports, images = zip(*all_ranks_data)
        fig = px.bar(x=airports, y=images, labels={'x': 'Airport', 'y': 'Images'})
        st.header(f"Rank of {cleaned_airport} Airport is: {rank}")
        st.plotly_chart(fig)

    with tab3:  # over threshold graph tab
        first_img, last_img = insights_db.get_first_last_img(cleaned_airport)
        threshold_count = insights_db.get_over_threshold(cleaned_airport)
        data = {
            'Timestamp': pd.date_range(first_img, last_img, freq='M'),
            'Threshold Count': threshold_count[:]
        }
        fig = px.scatter(data,
                         title="Over Threshold Distribution",
                         x='Timestamp',
                         y='Threshold Count',
                         size='Threshold Count',
                         color='Threshold Count',
                         hover_data=['Timestamp', 'Threshold Count'],
                         color_continuous_scale='Viridis')

        st.plotly_chart(fig)


def display_all(cleaned_airport, start_time, end_time):  # displays all statistics
    display_statistics(cleaned_airport, start_time, end_time)
    display_scatter(cleaned_airport, start_time, end_time)
    display_tabs(cleaned_airport)


def display_header():  # displays header information
    st.header('Airport Research Tools :airplane_departure: 🛬', divider='blue')


display_header()

airport_name = st.multiselect("Choose Airport for Analysis",
    placeholder="Airport Name",
    options=[
        "Amsterdam - Amsterdam Airport Schiphol (AMS)",
        "Atlanta - Hartsfield-Jackson International Airport (ATL)",
        "Bangkok - Suvarnabhumi Airport (BKK)",
        "Beijing - Capital International Airport (PEK)",
        "Chicago - O'Hare International Airport (ORD)",
        "Dallas/Fort Worth - Dallas/Fort Worth International Airport (DFW)",
        "Denver - Denver International Airport (DEN)",
        "Dubai - Dubai International Airport (DXB)",
        "Frankfurt - Frankfurt Airport (FRA)",
        "Guangzhou - Baiyun International Airport (CAN)",
        "Hong Kong - Hong Kong International Airport (HKG)",
        "Istanbul - Istanbul Airport (IST)",
        "London - Heathrow Airport (LHR)",
        "Los Angeles - Los Angeles International Airport (LAX)",
        "New York City - John F. Kennedy International Airport (JFK)",
        "Paris - Charles de Gaulle Airport (CDG)",
        "Seoul - Incheon International Airport (ICN)",
        "Shanghai - Pudong International Airport (PVG)",
        "Singapore - Changi Airport (SIN)",
        "Tokyo - Haneda Airport (HND)"
        ],
    max_selections=1,
    label_visibility="visible")

start_time = st.date_input("Start Time for Analysis", value=None)
end_time = st.date_input("End Time for Analysis", value=None)

cleaned_airport = str(
    airport_name[0][0:-1].replace(' (', '_')) if airport_name else "No Airport Specified"  # clean airport name


if start_time is not None and end_time is not None and airport_name is not None:
    display_all(cleaned_airport, start_time, end_time)


