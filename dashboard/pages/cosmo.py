from .cosmo_backend import COSMO

import streamlit as st
import pandas as pd
import numpy as np




import plotly.graph_objects as go

STEP = 1e-6
FORMAT = '%.6f'

def render():
    st.title("COSMO Dashboard")

    print(st.session_state)    

    # Create form for submission of OOI
    with st.form("ooiForm"):
        ooi_df = pd.DataFrame({}, index=[0])

        if st.session_state['fill']:
            ooi_df['ObjectOfInterest'] = st.text_input("OOIId", 'TOI 1063.01')
            ooi_df['OrbitalPeriod'] = st.number_input("Orbital Period (Days)", step=STEP, format=FORMAT, value=10.0665634)
            ooi_df['TransitDuration'] = st.number_input("Transit Duration (Hours)", step=STEP, format=FORMAT, value=1.981)
            ooi_df['TransitDepth'] = st.number_input("Transit Depth (ppm)", step=STEP, format=FORMAT, value=640.00)
            ooi_df['PlanetEarthRadius'] = st.number_input("Planet Radius (Earths)", step=STEP, format=FORMAT, value=2.09966)
            ooi_df['PlanetEquilibriumTemperature'] = st.number_input("Planet Equilibrium Temperature (K)", step=STEP, format=FORMAT, value=615.00)
            ooi_df['StellarEffectiveTemperature'] = st.number_input("Stellar Effective Temperature (K)", step=STEP, format=FORMAT, value=5552.00)
            ooi_df['StellarLogG'] = st.number_input("Stellar Surface Gravity in log base 10 (log_10(cm/s^2))", step=STEP, format=FORMAT, value=4.61783)
            ooi_df['StellarSunRadius'] = st.number_input("Stellar Radius (Suns)", step=STEP, format=FORMAT, value=0.79)
        else:
            ooi_df['ObjectOfInterest'] = st.text_input("OOIId")
            ooi_df['OrbitalPeriod'] = st.number_input("Orbital Period (Days)", step=STEP, format=FORMAT)
            ooi_df['TransitDuration'] = st.number_input("Transit Duration (Hours)", step=STEP, format=FORMAT)
            ooi_df['TransitDepth'] = st.number_input("Transit Depth (ppm)", step=STEP, format=FORMAT)
            ooi_df['PlanetEarthRadius'] = st.number_input("Planet Radius (Earths)", step=STEP, format=FORMAT)
            ooi_df['PlanetEquilibriumTemperature'] = st.number_input("Planet Equilibrium Temperature (K)", step=STEP, format=FORMAT)
            ooi_df['StellarEffectiveTemperature'] = st.number_input("Stellar Effective Temperature (K)", step=STEP, format=FORMAT)
            ooi_df['StellarLogG'] = st.number_input("Stellar Surface Gravity in log base 10 (log_10(cm/s^2))", step=STEP, format=FORMAT)
            ooi_df['StellarSunRadius'] = st.number_input("Stellar Radius (Suns)", step=STEP, format=FORMAT)
        

        if st.form_submit_button("Submit"):
            st.session_state['submitted'] = True

        if st.form_submit_button("Reset"):
            st.session_state['submitted'] = False

        if st.form_submit_button("Fill with Sample"):
            st.session_state['fill'] = True

        if st.form_submit_button("Clear"):
            st.session_state['fill'] = False

    if st.session_state['submitted']:
        cosmo = COSMO(ooi_df)
        show_results(cosmo)


def show_results(cosmo):
    st.title('Results')

    # Show results table
    results = cosmo.get_results()
    st.table(results)

    disp_proba, planet_proba = cosmo.get_results_proba()

    if type(planet_proba) == pd.DataFrame:
        col1, col2 = st.columns([2, 2])

        with col1:
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
            
            st.plotly_chart(fig, use_container_width=True)

        with col2:
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
            st.plotly_chart(fig, use_container_width=True)
    else:
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
        
        st.plotly_chart(fig, use_container_width=True)

    # Compare with training data
    raw_df = cosmo.get_raw()
    numericals = [col for col in raw_df.columns if raw_df[col].dtype!=object]
    train_df = COSMO.get_train()

    comp_plot = st.selectbox("Column", numericals)
    temp = train_df[comp_plot]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=temp)
    )
    fig.update_layout(
        title=f'Corpus of Data for {comp_plot} vs Object of Interest',
        font = dict(
            family='sans serif',
            size=18
        )
    )
    fig.add_vline(x=raw_df[comp_plot].values[0], line_dash='dash', line_color='red', annotation_text=raw_df['ObjectOfInterest'].values[0])

    st.plotly_chart(fig, use_container_width = True)


    
