import streamlit as st
st.subtitle(":blue[Authors: Srijit Ghatak(G24AIT2079) and Mahantesh Hiremath(G24AIT2178)]  :speech_balloon:")

st.divider()






















st.markdown(
    '''
    <style>
    .streamlit-expanderHeader {
        background-color: blue;
        color: white; # Adjust this for expander header color
    }
    .streamlit-expanderContent {
        background-color: blue;
        color: white; # Expander content color
    }
    </style>
    ''',
    unsafe_allow_html=True
)

footer="""<style>

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: #2C1E5B;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤️ by <a style='display: inline; text-align: center;' href="https://iitj-ml-learnings.streamlit.app/" target="_blank">Srijit and Mahantesh</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)  