import re

from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import TokenTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.prompts import PromptTemplate


def check_link(link):
    """The function `check_link` uses a regular expression to check if a given link is a valid YouTube video link."""
    yt_regex = r"^(?:https?:\/\/)?(?:www\.)?youtu(?:be\.com\/watch\?v=|\.be\/)([\w\-\_]+)(?:\?.*)?$"
    return re.match(yt_regex, link) is not None


def get_transcript(video_link):
    if not check_link(video_link):
        return "Invalid YouTube URL."
    try:
        loader = YoutubeLoader.from_youtube_url(video_link, language=["en", "en-US"])
        transcript = loader.load()
        return transcript
    except Exception as e:
        return f"Failed to retrieve transcript: {e}"


def split_chunks(transcript):

    splitter = TokenTextSplitter(
        chunk_size=7500, chunk_overlap=100
    )  # Llama 3 model takes up to 8192 input tokens, so I set chunk size to 7500 for leaving some space to model.
    chunks = splitter.split_documents(transcript)
    return chunks


def yt_summarization_chain():
    prompt_template = PromptTemplate(
        template="""As a professional summarizer specialized in video content, create a detailed and comprehensive summary of the YouTube video transcript provided. While crafting your summary, adhere to these guidelines:
            1. Capture the essence of the video, focusing on main ideas and key details. Ensure the summary is in-depth and insightful, reflecting any narrative or instructional elements present in the video.

            2. Exclude any redundant expressions and non-critical details to enhance the clarity and conciseness of the summary.

            3. Base the summary strictly on the transcript provided, avoiding assumptions or additions from external sources.

            4. Present the summary in a well-structured paragraph form, making it easy to read and understand.

            5. Conclude with "[End of Notes, Message #X]", where "X" is the sequence number of the summarizing request, to indicate the completion of the task.

        By adhering to this optimized prompt, you are expected to produce a clear, detailed, and audience-friendly summary that effectively conveys the core content and themes of the YouTube video.

        "{text}"

        DETAILED SUMMARY:""",
        input_variables=["text"],
    )
    llm = ChatOllama(model="llama3:instruct", base_url="http://127.0.0.1:11434")
    summarize_chain = load_summarize_chain(
        llm=llm, prompt=prompt_template, verbose=True
    )
    return summarize_chain


def summarize_video(video_link):
    transcript = get_transcript(video_link)
    chunks = split_chunks(transcript)

    sum_chain = yt_summarization_chain()
    result = sum_chain.run(chunks)

    return result
