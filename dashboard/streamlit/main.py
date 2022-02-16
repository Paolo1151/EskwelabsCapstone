import streamlit as st

from pages import cosmo, credits, about, explore

def main():
    page_list = [
        'COSMO Dashboard',
        'Data Exploration',
        'About COSMO',
        'Credits'
    ]

    page = st.sidebar.radio("Page", page_list)

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