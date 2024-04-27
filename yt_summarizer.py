from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.chains.summarize import load_summarize_chain
import re

def check_link(link):
    yt_regex = r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[\w-]+"
    return re.match(yt_regex, link) is not None

def get_transcript(video_link):
    # Get video transcript
    if check_link(video_link):
        loader = YoutubeLoader.from_youtube_url(video_link, language=["en", "en-US"])
        transcript = loader.load()
        return transcript
    return "Invalid YouTube URL."

def split_chunks(transcript):
    # Split the transcript into chunks
    # Llama 3 model takes up to 8192 input tokens, so I set chunk size to 7500 for leaving some space to model.
    splitter = TokenTextSplitter(chunk_size = 7500, chunk_overlap = 100)
    chunks = splitter.split_documents(transcript)
    return chunks

def yt_summarization_chain():
    llm = ChatOllama(model="llama3")
    summarize_chain = load_summarize_chain(llm=llm, chain_type="refine", verbose=True)
    return summarize_chain

if __name__ == "__main__":
    videos = ["https://www.youtube.com/watch?v=bYjQ9fzinT8", "https://www.youtube.com/watch?v=QCg0axyXxs4"]

    transcript = get_transcript(videos[0])
    chunks = split_chunks(transcript)

    sum_chain = yt_summarization_chain()
    result = sum_chain.run(chunks)
    
    print(result)