import argparse
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.chat_models import ChatOllama

def setup_argparse():
    """Setup argparse to parse command line arguments."""
    parser = argparse.ArgumentParser(description="Summarize a document from a given URL.")
    parser.add_argument("-u", "--url", required=True, help="URL of the document to summarize")
    return parser.parse_args()

def load_document(url):
    """Load document from the specified URL."""
    loader = WebBaseLoader(url)
    return loader.load()

def setup_summarization_chain():
    """Setup the summarization chain with a prompt template and ChatOllama."""
    prompt_template = PromptTemplate(template="""Write a detailed long summary of the following:
"{text}"
DETAILED SUMMARY:""", input_variables=["text"])

    llm = ChatOllama(model="llama2")
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    return llm_chain

def main():
    args = setup_argparse()
    docs = load_document(args.url)
    result = llm_chain.run(docs)
    
    print(result)

if __name__ == "__main__":
    main()
