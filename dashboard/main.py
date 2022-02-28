import sys
import os
# sys.path.append(os.path.join(os.path.dirname(__file__), ''))

import streamlit as st

from pages import cosmo, credits, about, explore

import warnings

def main():
    st.set_page_config(
        page_title='COSMO Dashboard',
        layout='wide',
        initial_sidebar_state='collapsed'
    )
    
    warnings.filterwarnings('ignore')

    if 'submitted' not in st.session_state:
        st.session_state['submitted'] = False
    
    if 'fill' not in st.session_state:
        st.session_state['fill'] = False

    page_list = [
        'COSMO Dashboard',
        'Data Exploration',
        'About COSMO',
        'Credits'
    ]

    with st.sidebar:
        st.subheader("Pages")
        page = st.radio("Pages", page_list)

    if page == 'COSMO Dashboard':
        cosmo.render()
    elif page == 'Data Exploration':
        explore.render()
    elif page == 'About COSMO':
        about.render()
    elif page == 'Credits':
        credits.render()

if __name__ == '__main__':
    main()