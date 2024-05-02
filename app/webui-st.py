import streamlit as st
from translator import setup_translator_chain
from yt_summarizer import summarize_video, check_link
from summarizer import load_document, setup_summarization_chain

def summarize(url):
    if check_link(url):
        result = summarize_video(url)
    else:
        docs = load_document(url)
        llm_chain = setup_summarization_chain()
        result = llm_chain.run(docs)
    
    return result

def translate(text):
    llm_chain = setup_translator_chain()
    result = llm_chain.run(text)
    return result

st.title('Cobanov Web and Video Summarizer')
st.write('Easily summarize any web page or YouTube video with a single click.')

url = st.text_input('URL:')

if st.button('Summarize'):
    if url:
        with st.spinner('Summarizing....'):
            st.session_state.summary = summarize(url)
            st.write(st.session_state.summary)
    else:
        st.error('Please enter a valid URL!')

if st.button('🇹🇷 Translate'):
    if 'summary' in st.session_state and st.session_state.summary:
        with st.spinner('Translating....'):
            translation = translate(st.session_state.summary)
            st.write(translation)
    else:
        st.error('Please generate a summary first!')

st.write("""
        ```
        Model: llama3-8b
        Author: Mert Cobanov
        Contact: mertcobanov@gmail.com
        Repo: github.com/mertcobanov/easy-web-summarizer
        ```""")