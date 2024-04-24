import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Gen-AI Based AI News Bulletin and Voice Bot! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    ### This Demo provide us a detailed Financial News & Question/Answers for selected Tickers.

    **👈 Select a demo from the sidebar**
"""
)