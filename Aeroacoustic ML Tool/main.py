import audio_recorder
import audio_cropper
import audio_plotter
import nozzle_controller
import audio_feature_generator
import audio_feature_regressor
import streamlit as st

st.set_page_config(layout="wide")

PAGES = {
    "Recorder": audio_recorder,  # "key": value
    "Cropper": audio_cropper,
    "Feature Generator": audio_feature_generator,
    "Feature Regressor": audio_feature_regressor,
    "Plotter": audio_plotter,
    "Nozzle Controller": nozzle_controller,
}
st.sidebar.title("Navigation")

selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]

# with st.spinner(f'Loading {selection}'):
# page.main()
page.main()
