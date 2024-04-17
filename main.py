import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

def get_response(txt_file, openai_api_key, query_text):
    if txt_file is not None:
        documents = [txt_file.read().decode()]
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0)
    docs = text_splitter.create_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma.from_documents(
        docs, 
        embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=openai_api_key), 
        chain_type='stuff', 
        retriever=retriever)
   
    return qa.run(query_text)

st.title('Talk with your txt files')

txt_file = st.file_uploader('Upload a text file', type='txt')
query = st.text_input('Enter your question:', disabled=not txt_file)

result = []
with st.form('myform', clear_on_submit=True):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (txt_file and query))

    submitted = st.form_submit_button('Submit', disabled=not(txt_file and query))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = get_response(txt_file, openai_api_key, query)
            result.append(response)
            del openai_api_key
if len(result):
    st.info(response)