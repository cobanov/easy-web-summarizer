import gradio as gr

from summarizer import load_document, setup_summarization_chain
from translator import setup_translator_chain
from yt_summarizer import check_link, summarize_video


def summarize(url):
    if check_link(url):
        result = summarize_video(url)
    else:
        docs = load_document(url)
        llm_chain = setup_summarization_chain()
        result = llm_chain.run(docs)

    return [result, gr.Button("🇹🇷 Translate ", visible=True)]


def translate(text):
    llm_chain = setup_translator_chain()
    result = llm_chain.run(text)
    return result


with gr.Blocks() as demo:
    gr.Markdown(
        """# Cobanov Web and Video Summarizer
    Easily summarize any web page or YouTube video with a single click."""
    )

    with gr.Row():
        with gr.Column():
            url = gr.Text(label="URL", placeholder="Enter URL here")

            btn_generate = gr.Button("Generate")

            summary = gr.Markdown(label="Summary")
            btn_translate = gr.Button(visible=False)

    gr.Examples(
        [
            "https://cobanov.dev/haftalik-bulten/hafta-13",
            "https://bawolf.substack.com/p/embeddings-are-a-good-starting-point",
            "https://www.youtube.com/watch?v=4pOpQwiUVXc",
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
    btn_generate.click(summarize, inputs=[url], outputs=[summary, btn_translate])
    btn_translate.click(translate, inputs=[summary], outputs=[summary])

demo.launch(server_name="0.0.0.0")
