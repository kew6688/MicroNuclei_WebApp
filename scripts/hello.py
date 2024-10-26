import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Welcome to Micronuclei Detection! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    This is an open-source app framework built specifically for
    micronuclei detection projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [project github](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    ### See more demos
    - [Process_workflow.ipynb](https://github.com/streamlit/demo-self-driving)
    - [mn_track.ipynb](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)