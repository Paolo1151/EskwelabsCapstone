import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ''))

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

    cosmo.render()

if __name__ == '__main__':
    main()