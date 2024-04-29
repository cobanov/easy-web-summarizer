import gradio as gr

from summarizer import load_document, setup_summarization_chain
from yt_summarizer import summarize_video
from translator import setup_translator_chain

def summarize(url):
    docs = load_document(url)
    llm_chain = setup_summarization_chain()
    result = llm_chain.run(docs)

    return [result, gr.Button("🇹🇷 Translate ", visible=True)]

def translate(text):
    llm_chain = setup_translator_chain()
    result = llm_chain.run(text)
    return result

def update_ui(content_type):
    if content_type == "Web":
        # Set visibility for Web URL input to True and Video URL input to False
        url_visibility = ""  # Empty string signifies no change for a text box.
        video_url_visibility = ""  # Set video URL field to empty since it should be hidden.
        btn_text = "Generate Summary"  # Button text for generating summary from a web URL
    elif content_type == "Video":
        # Set visibility for Web URL input to False and Video URL input to True
        url_visibility = ""  # Set web URL field to empty since it should be hidden.
        video_url_visibility = ""  # Empty string signifies no change for a text box.
        btn_text = "Summarize Video"  # Button text for summarizing video content
    else:
        # Hide both inputs (unlikely to need this else, but just in case)
        url_visibility = ""
        video_url_visibility = ""
        btn_text = ""  # Clear the button text

    return url_visibility, video_url_visibility, btn_text

with gr.Blocks() as demo:
    gr.Markdown(
        """# Cobanov Web and Video Summarizer
    Easily summarize any web page or YouTube video with a single click."""
    )

    with gr.Row():
        with gr.Column():
            content_type = gr.Radio(choices=["Web", "Video"], label="Select Content Type", value="Web")
            
            url = gr.Text(label="URL", placeholder="Enter URL here")
            video_url = gr.Text(label="YouTube Video URL", placeholder="Enter YouTube video URL here", visible=False)

            btn_generate = gr.Button("Generate")

            summary = gr.Markdown(label="Summary")
            btn_translate = gr.Button(visible=False)
        
        content_type.change(update_ui, inputs=[content_type], outputs=[url, video_url, btn_generate])

    gr.Examples(
        [
            "https://cobanov.dev/haftalik-bulten/hafta-13",
            "https://bawolf.substack.com/p/embeddings-are-a-good-starting-point",
        ],
        inputs=[url],
    )
    gr.Markdown(
        """
        ```
        Model: llama3-8b
        Author: Mert Cobanov
        Contact: mertcobanov@gmail.com
        Repo: github.com/mertcobanov/easy-web-summarizer
        ```"""
    )
    btn_generate.click(lambda url, video_url: summarize(url) if url else summarize_video(video_url), 
                     inputs=[url, video_url], 
                     outputs=[summary, btn_translate])
    btn_translate.click(translate, inputs=[summary], outputs=[summary])

demo.launch()