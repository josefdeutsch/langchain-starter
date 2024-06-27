import streamlit as st
import langchain_helper as lch
import textwrap

# Streamlit app
st.title("Patricia's YouTube Assistant")

with st.sidebar:
    with st.form(key='my_form'):
        youtube_url = st.text_area(
            label="What is the YouTube video URL?",
            max_chars=50,
            height=None
        )
        query = st.text_area(
            label="Ask me about the video?",
            max_chars=75,
            key="query",
            height=None
        )
        submit_button = st.form_submit_button(label='Submit')

#Use agenda example:
#https://destinyhosted.com/agenda_publish.cfm?id=45623&mt=ALL&vl=true&get_month=6&get_year=2024&dsp=ag&seq=1494
#https://www.youtube.com/watch?v=R2q08ZP8h74
if query and youtube_url and submit_button:
    db = lch.create_db_from_youtube_video_url(youtube_url)
    response, docs = lch.get_response_from_query(db, query)
    
    st.subheader("Answer:")
    formatted_response = textwrap.fill(response, width=85)
    st.markdown(f"<pre>{formatted_response}</pre>", unsafe_allow_html=True)

