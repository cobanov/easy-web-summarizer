from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama


def setup_translator_chain():
    """Setup the translation chain with a prompt template and ChatOllama."""
    prompt_template = PromptTemplate(
        template="""As a professional translator, provide a detailed and comprehensive translation of the provided text into turkish, ensuring that the translation is accurate, coherent, and faithful to the original text.

        "{text}"

        DETAILED TRANSLATION:""",
        input_variables=["text"],
    )

    llm = ChatOllama(model="llama3:instruct", base_url="http://127.0.0.1:11434")
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    return llm_chain
