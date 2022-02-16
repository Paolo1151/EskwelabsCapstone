from .cosmo_backend import COSMO

import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

STEP = 1e-6
FORMAT = '%.6f'

def render():
    st.title("COS-MO Dashboard")
    
    # Create form for submission of OOI
    with st.form("ooiForm"):
        ooi_df = pd.DataFrame({}, index=[0])

        ooi_df['ObjectOfInterest'] = st.text_input("OOIId")
        ooi_df['OrbitalPeriod'] = st.number_input("Orbital Period (Days)", step=STEP, format=FORMAT)
        ooi_df['TransitDuration'] = st.number_input("Transit Duration (Hours)", step=STEP, format=FORMAT)
        ooi_df['TransitDepth'] = st.number_input("Transit Depth (ppm)", step=STEP, format=FORMAT)
        ooi_df['PlanetEarthRadius'] = st.number_input("Planet Radius (Earths)", step=STEP, format=FORMAT)
        ooi_df['PlanetEquilibriumTemperature'] = st.number_input("Planet Equilibrium Temperature (K)", step=STEP, format=FORMAT)
        ooi_df['StellarEffectiveTemperature'] = st.number_input("Stellar Effective Temperature (K)", step=STEP, format=FORMAT)
        ooi_df['StellarLogG'] = st.number_input("Stellar Surface Gravity in log base 10 (log_10(cm/s^2))", step=STEP, format=FORMAT)
        ooi_df['StellarSunRadius'] = st.number_input("Stellar Radius (Suns)", step=STEP, format=FORMAT)
        
        submitted = st.form_submit_button("Submit")

        

    if submitted:
        cosmo = COSMO(ooi_df)
        show_results(cosmo)


def show_results(cosmo):
    st.title('Results')

    # Show results table
    results = cosmo.get_results()
    st.table(results)

    disp_proba, planet_proba = cosmo.get_results_proba()

    # Plot Disposition Probability    
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=pd.melt(disp_proba)['variable'], y=pd.melt(disp_proba)['value'])
    )

    fig.update_layout(
        title=f'Disposition Probabilities',
        font = dict(
            family='sans serif',
            size=18
        )
    )
    
    st.plotly_chart(fig)

    if type(planet_proba) == pd.DataFrame:
        # Plot Type Probability
        fig = go.Figure()
        fig.add_trace(
            go.Bar(x=pd.melt(planet_proba)['variable'], y=pd.melt(planet_proba)['value'])
        )

        fig.update_layout(
            title=f'PlanetType Probabilities',
            font = dict(
                family='sans serif',
                size=18
            )
        )
        st.plotly_chart(fig)

    # Compare with training data
    raw_df = cosmo.get_raw()

    numericals = [col for col in raw_df.columns if raw_df[col].dtype!=object]

    train_df = COSMO.get_train()

    for num in numericals:
        temp = train_df[num]

        fig = go.Figure()

        fig.add_trace(
            go.Histogram(x=temp)
        )

        fig.update_layout(
            title=f'Corpus of Data for {num} vs Object of Interest',
            font = dict(
                family='sans serif',
                size=18
            )
        )

        fig.add_vline(x=raw_df[num].values[0], line_dash='dash', line_color='red', annotation_text=raw_df['ObjectOfInterest'].values[0])

        st.plotly_chart(fig)

    
