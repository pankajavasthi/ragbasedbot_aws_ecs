#import Essential dependencies
import streamlit as sl
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import sentence_transformer, huggingface
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from details import api_key
from details import gpi_key

from gtts import gTTS 

import yfinance as yf
import openai
from IPython.display import Audio
from pathlib import Path
#from pages.test import get_news_headlines, get_headlines_for_tickers,send_data_and_get_response,create_personalized_cast


#function to load the vectordatabase
def load_knowledgeBase():
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        DB_FAISS_PATH = 'vectorstore/db_faiss'
        db = FAISS.load_local(DB_FAISS_PATH, embeddings,allow_dangerous_deserialization=True)
        return db
        
#function to load the OPENAI LLM
def load_llm():
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key= gpi_key)
        return llm

#creating prompt template using langchain
def load_prompt():
        prompt = """ You need to answer the question in the sentence as same as in the  pdf content. . 
        Given below is the context and question of the user.
        context = {context}
        question = {question}
        if the answer is not in the pdf answer "Kindly reframe the question or context is not present in the given content"
         """
        prompt = ChatPromptTemplate.from_template(prompt)
        return prompt


def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

def text_to_audio(text):
        myobj = gTTS(text=text, lang= 'en', slow=False)
        return myobj.save("welcome.mp3")




        
knowledgeBase=load_knowledgeBase()
llm=load_llm()
prompt=load_prompt()

sl.set_page_config(page_title="Gen-AI Voice Bot", page_icon="📈")

sl.markdown("# Gen-AI Voice Bot Demo")
sl.sidebar.header("Gen-AI Voice Bot Demo")
sl.write(
    """This demo illustrates a Gen-AI Voice Bot based interation using RAG Based LLMs ! """
)
        
sl.header("Welcome to RAG Based GEN-AI Voice Bot")
        
query=sl.text_input('Please ask you Query realted to selected Ticker')
        
        

                
#getting only the chunks that are similar to the query for llm to produce the output
similar_embeddings=knowledgeBase.similarity_search(query)
similar_embeddings=FAISS.from_documents(documents=similar_embeddings, embedding=OpenAIEmbeddings(api_key=api_key))
                
#creating the chain for integrating llm,prompt,stroutputparser
retriever = similar_embeddings.as_retriever()
rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
                )
                
response=rag_chain.invoke(query)
sl.write(response)
audio = text_to_audio(response)
sl.audio("welcome.mp3")