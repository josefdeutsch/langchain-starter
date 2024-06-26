import os
from langchain_community.document_loaders import YoutubeLoader
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

# Set the environment variable for the API key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

embeddings = OpenAIEmbeddings()

def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    
    #faiss-cpu 1.7.4
    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = OpenAI(model="gpt-3.5-turbo-instruct")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a highly knowledgeable assistant specialized in providing detailed and comprehensive answers 
        to questions based on YouTube video transcripts.

        Answer the following question as precisely and verbosely as possible, ensuring that your response is exhaustive 
        and utilizes all relevant information found in the provided transcript segments.

        Question: {question}

        Use the following video transcript to answer the question:
        {docs}

        Your answers should be long, detailed, and written in modern English. Make sure to include all pertinent details 
        and elaborate on any points that are directly related to the question. If the transcript provides examples, 
        data, or specific quotes, incorporate them into your response to enhance its comprehensiveness.
        """
    )
    chat_prompt = ChatPromptTemplate(
    messages=[
        SystemMessage(
            content="""
            You are a highly knowledgeable assistant specialized in providing detailed and comprehensive answers 
            to questions based on YouTube video transcripts. Answer questions as precisely and verbosely as possible,
            ensuring that your responses are exhaustive and utilize all relevant information found in the provided transcript segments.
            """
        ),
        HumanMessage(
            content="""
            Question: {question}

            Use the following video transcript to answer the question:
            {docs}

            Your answers should be long, detailed, and written in modern English. Make sure to include all pertinent details 
            and elaborate on any points that are directly related to the question. If the transcript provides examples, 
            data, or specific quotes, incorporate them into your response to enhance its comprehensiveness.
            """
        )
    ],
    input_variables=["question", "docs"]
    )
    parser = StrOutputParser()
    #chain = LLMChain(llm=llm, prompt=prompt)
    
    chain = prompt | llm | parser
    # Use a dictionary to pass the inputs to the invoke method
    response = chain.invoke({"question": query, "docs": docs_page_content})
    print(response)
    #response = response.replace("\n", "")
    return response, docs




