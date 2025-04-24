import streamlit as st
from streamlit_image_comparison import image_comparison


def display_header():
    st.set_page_config(page_icon="images/Aircraft Intelligence Hub Badge.ico")
    st.header('Aircraft Intelligence Hub', divider='blue')
    left_co, cent_co, right_co = st.columns(3, gap="small")
    with cent_co:
        st.image("images/Aircraft Intelligence Hub Badge Small.jpg")
    st.title("Aircraft and Airport Analysis Hub :airplane:")


def display_explanation():
    st.write("The Aircraft Intelligence Hub, powered by advanced algorithms, streamlines satellite image analysis for enhanced precision and rapid insights. Leveraging automation, it offers valuable intelligence on airport dynamics, aircraft flow, and economic factors, contributing to informed decision-making.")
    left_co, cent_co, right_co = st.columns(3, gap="small")
    with cent_co:
        st.write("**Explore examples down below**")

def display_example_tabs():  # displays example columns
    image_names = "examples/Interesting.jpg", "examples/Interesting Analyzed.png"
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["**Image Analysis**", "**Airport Image Distribution**",
                                            "**Airport Image Heatmap**",
                                            "**Airport Ranking**",
                                            "**Airport Over Threshold**"])
    with tab1:  # displays image analysis examples
        image_comparison(image_names[0], image_names[1])
    with tab2:
        st.image("examples/Image Distribution.png")
    with tab3:
        st.image("examples/Image Count Heatmap.png")
    with tab4:
        st.image("examples/Airport Ranking.png")
    with tab5:
        st.image("examples/Threshold Distribution.png")


display_header()
display_explanation()
display_example_tabs()
