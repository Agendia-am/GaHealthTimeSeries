import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# 1. Load CSVs and concatenate
csv_files = [
    'dch_news_analysis_complete.csv',
    'dch_news_analysis_improved.csv',
    'dch_news_analysis.csv'
]
texts = []
for file in csv_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        for col in df.columns:
            texts.extend(df[col].astype(str).tolist())

# 2. Split text into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.create_documents(texts)

# 3. Create embeddings and store in Chroma
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings, persist_directory="chroma_db")

# 4. Set up retriever
retriever = vectorstore.as_retriever()

# 5. Set up LLM and QA chain
llm = OpenAI(temperature=0)
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def answer_query(query: str):
    return qa.run(query)

def main():
    st.title("RAG Q&A over CSV Data")
    user_query = st.text_input("Enter your question:")
    if user_query:
        with st.spinner("Generating answer..."):
            answer = answer_query(user_query)
        st.write("**Answer:**", answer)

if __name__ == "__main__":
    main()
