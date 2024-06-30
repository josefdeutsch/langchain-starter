import streamlit as st
import langchain_helper as lch

import streamlit as st
import langchain_helper as lch
import textwrap

# Streamlit app
st.title("Patricia's Pdf Assistant")

with st.sidebar:
    with st.form(key='my_form'):
        
        query = st.text_area(
            label="Ask me about the video?",
            max_chars=75,
            key="query",
            height=None
        )
        submit_button = st.form_submit_button(label='Submit')

if query and submit_button:
    pages = lch.load_pdf_from_disk("/Users/Joseph/Documents/langchain-starter/annualreport.pdf")

    db = lch.get_database(pages)

    #retriever = get_retriever_from_database(db) 

    llm = lch.get_language_model()

    vectorstore_info = lch.get_vector_store_info(db)

    #retiever = lch.get_retriever_from_database(db)

    vector_tool = lch.get_vector_store_tools(llm, vectorstore_info)

    agent_executor = lch.get_vector_store_agent(llm, vector_tool)

    response = agent_executor.run(query)
    
    st.subheader("Answer:")
    formatted_response = textwrap.fill(response, width=85)
    st.markdown(f"<pre>{formatted_response}</pre>", unsafe_allow_html=True)
    
    response = db.similarity_search_with_score(response)
    st.expander("expand").write(response)
    

