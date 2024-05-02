import streamlit as st
from yt_summarizer import summarize_video

st.title('Web and Video Summarizer - TEST')
st.write('this is a test text')

url = st.text_input('URL:')

if st.button('Summarize'):
    if url:
        with st.spinner('Summarizing....'):
            sum = summarize_video(url)
            st.write(sum)

