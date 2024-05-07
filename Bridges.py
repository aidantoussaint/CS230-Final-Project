import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pydeck as pdk

default_year = 2000

# Read in data
def read_data():
    return pd.read_csv("Bridges.csv")

# Filter data by agency for the map
def filter_data(sel_agency, min_year):
    df = read_data()
    df = df[df['22 - Owner Agency'] == sel_agency]
    df = df[df['27 - Year Built'] > min_year]
    return df

# Function to render the Bar Chart
def generate_bar_chart(data, span_filter):
    if data.empty:
        st.write("No data available to display the bar chart.")
    else:
        # Filter the data based on the slider
        filtered_data = data[data.values >= span_filter]
        if filtered_data.empty:
            st.write("No designs with average spans above the selected threshold.")
        else:
            try:
                fig, ax = plt.subplots()
                ax.bar(filtered_data.index, filtered_data.values)
                ax.set_xlabel('Main Span Design')
                ax.set_ylabel('Average Number of Spans in Main Unit')
                ax.set_title('Average Number of Spans in Main Unit by Main Span Design')
                plt.xticks(rotation=45, ha='right')
                st.pyplot(fig)
            except Exception as e:
                st.write("Failed to render bar chart:", e)

# Function to render the map
def render_map(df):
    if df.empty:
        st.write("No data available to display the map.")
        return

    # Check and convert data types
    if not pd.api.types.is_float_dtype(df['16 - Latitude (decimal)']) or not pd.api.types.is_float_dtype(df['17 - Longitude (decimal)']):
        df['16 - Latitude (decimal)'] = pd.to_numeric(df['16 - Latitude (decimal)'], errors='coerce')
        df['17 - Longitude (decimal)'] = pd.to_numeric(df['17 - Longitude (decimal)'], errors='coerce')

    # Check for NaN values which could cause the dots not to appear
    if df['16 - Latitude (decimal)'].isnull().any() or df['17 - Longitude (decimal)'].isnull().any():
        st.write("Missing latitude or longitude data.")
        return
    # Prepare tooltips by formatting a'tooltip' column
    df['tooltip'] = df['8 - Structure Number'].apply(lambda x: f"Structure Number: {x}")

    # Rename columns for easier access
    df.rename(columns={'17 - Longitude (decimal)': 'longitude', '16 - Latitude (decimal)': 'latitude'}, inplace=True)

    # Define the view state of the map
    view_state = pdk.ViewState(
        latitude=df['latitude'].mean(),
        longitude=df['longitude'].mean(),
        zoom=7,
        pitch=0)

    # Define the layer with correct position and tooltip
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=df,
        get_position='[longitude, latitude]',
        get_color=[200, 30, 0, 160],
        get_radius=1000,  
        pickable=True,
        auto_highlight=True,
        tooltip={"text": "tooltip"}
    )

    # Create the deck.gl map
    st.pydeck_chart(pdk.Deck(
        initial_view_state=view_state,
        layers=[layer],
        tooltip={"text": "{tooltip}"}
    ))

def count_agencies(agencies, df):
    return [df[df["22 - Owner Agency"] == agency].shape[0] for agency in agencies]

def generate_pie_chart(counts, sel_agencies):
    fig, ax = plt.subplots()
    explodes = [0 for i in range(len(counts))]
    maximum_index = np.argmax(counts)
    explodes[maximum_index] = 0.1  # Slight explosion for the largest segment
    ax.pie(counts, labels=sel_agencies, explode=explodes, autopct="%.2f%%")
    ax.set_title("Percentage of Bridges by Agency")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    st.pyplot(fig)

def design_length(df):
    if '43B - Main Span Design' not in df.columns or '45 - Number of Spans in Main Unit' not in df.columns:
        st.write("Required columns for the bar chart are missing.")
        return pd.Series()
    return df.groupby('43B - Main Span Design')['45 - Number of Spans in Main Unit'].mean()

def app():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page:", ["Home", "Pie Chart", "Bar Chart", "Map"])
    
    data = read_data()

    if page == "Home":
        st.title("Welcome to the Bridge Data Visualization App")
        st.write("Please navigate using the sidebar to see different visualizations.")
        st.write("This program is a compilation of Bridges in Georgia including locations, designs, and owner agencies.")
        #Image Display
        st.image("gerogiabridge.jpg", caption="Anaklia-Ganmukhari Pedestrian Bridge")

    elif page == "Pie Chart":
        st.title("Pie Chart of Bridge Agencies")
        if data.empty:
            st.write("Loaded data is empty. Check the data source.")
        else:
            all_agencies = data['22 - Owner Agency'].unique().tolist()
            selected_agencies = st.multiselect("Select Agencies", all_agencies)
            if selected_agencies:
                pie_data = data[data['22 - Owner Agency'].isin(selected_agencies)]
                agency_counts = count_agencies(selected_agencies, pie_data)
                generate_pie_chart(agency_counts, selected_agencies)

    elif page == "Bar Chart":
        st.title("Bar Chart of Main Span Designs")
        if data.empty:
            st.write("No data available to render the bar chart.")
        else:
            design_data = design_length(data)
            if not design_data.empty:
                span_threshold = st.slider("Select minimum average number of spans:", int(design_data.min()), int(design_data.max()), int(design_data.mean()))
                generate_bar_chart(design_data, span_threshold)
            else:
                st.write("No data available to generate the bar chart.")

    elif page == "Map":
        st.title("Map of Bridges")
        if data.empty:
            st.write("Loaded data is empty. Check the data source.")
        else:
            all_agencies = data['22 - Owner Agency'].unique().tolist()
            selected_agency_map = st.selectbox("Select Agency for Map", all_agencies)
            filtered_data_map = filter_data(selected_agency_map, default_year)
            if filtered_data_map.empty:
                st.write("No data available for this agency with the specified criteria.")
            else:
                render_map(filtered_data_map)

if __name__ == "__main__":
    app()
