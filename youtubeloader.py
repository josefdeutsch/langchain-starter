import os
from langchain_community.document_loaders import YoutubeLoader
from langchain_openai import OpenAIEmbeddings

from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

# Set the environment variable for the API key
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv('LANGCHAIN_TRACING_V2')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

embeddings = OpenAIEmbeddings()

def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    #https://python.langchain.com/v0.2/docs/integrations/document_loaders/youtube_transcript/
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()

    #https://python.langchain.com/v0.2/docs/how_to/recursive_text_splitter/#splitting-text-from-languages-without-word-boundaries
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    
    #faiss-cpu 1.7.4
    #https://python.langchain.com/v0.2/docs/integrations/vectorstores/faiss/#as-a-retriever
    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k=4):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model="gpt-4")

    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )
    parser = StrOutputParser()
    #chain = LLMChain(llm=llm, prompt=prompt)
    
    chain = prompt | llm | parser
    # Use a dictionary to pass the inputs to the invoke method
    response = chain.invoke({"question": query, "docs": docs_page_content})
    #response = response.replace("\n", "")
    return response, docs

def main():
    url = 'https://www.youtube.com/watch?v=R2q08ZP8h74'
    query = 'is this a good youtubevideo?'
    db = create_db_from_youtube_video_url(url)
    response, docs = get_response_from_query(db, query)
    print(response)


if __name__ == "__main__":
    main()