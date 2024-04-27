from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

# Get video transcript
videos = ["https://www.youtube.com/watch?v=bYjQ9fzinT8", "https://www.youtube.com/watch?v=QCg0axyXxs4"]

loader = YoutubeLoader.from_youtube_url(videos[0], language=["en", "en-US"])
transcript = loader.load()
#print(transcript)

# Split the transcript into chunks
# Llama 3 model takes up to 8192 input tokens, so I set chunk size to 7500 for leaving some space to model.
splitter = TokenTextSplitter(chunk_size = 7500, chunk_overlap = 100)
chunks = splitter.split_documents(transcript)
#print(chunks)
#print("chunks: ", len(chunks))

llm = ChatOllama(model="llama3")
summarize_chain = load_summarize_chain(llm=llm, chain_type="refine", verbose=True)

summary = summarize_chain.run(chunks)
print(summary)