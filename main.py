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
            max_chars=50,
            key="query",
            height=None
        )
        submit_button = st.form_submit_button(label='Submit')

if query and youtube_url and submit_button:
    db = lch.create_db_from_youtube_video_url(youtube_url)
    response, docs = lch.get_response_from_query(db, query)
    
    st.subheader("Answer:")
    formatted_response = textwrap.fill(response, width=85)
    st.markdown(f"<pre>{formatted_response}</pre>", unsafe_allow_html=True)
    
    st.subheader("Relevant Transcript Sections:")
    for doc in docs:
        st.markdown(f"<pre>{doc.page_content}</pre>", unsafe_allow_html=True)
