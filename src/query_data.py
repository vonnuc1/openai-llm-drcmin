import os
import openai
from langchain_community.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores.chroma import Chroma

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

load_dotenv(override=True, verbose=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CHROMA_PATH = os.environ.get("CHROMA_PATH", "")
MODEL_NAME = "gpt-4-turbo"


def set_openai_credentials():
    """Set OpenAI API credentials."""
    openai.api_type = os.environ.get("OPENAI_API_TYPE", "azure")
    openai.api_base = os.environ.get("OPENAI_API_BASE", "")
    openai.api_version = os.environ.get("OPENAI_API_VERSION", "")
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")

set_openai_credentials()

def init_llm(model=MODEL_NAME, deployment_name=MODEL_NAME, temperature = 0, max_tokens=2000, max_retries=5):
    llm = AzureChatOpenAI(
        model=model,
        deployment_name=deployment_name,
        openai_api_key=openai.api_key,
        openai_api_base=openai.api_base,
        temperature = temperature,
        max_tokens=max_tokens,
        max_retries=max_retries
        )
    return llm

def get_answer(llm, context, question):
    """zero-shot"""
    prompt_template = PromptTemplate(
        input_variables = [context, question],
        template = """
        You will be provided some context that you will use to answer a question. 
        This is the context:

        {context}

        ---

        Answer the question based on the above context: {question}
        """
    )
    chain_classification = LLMChain(llm=llm, prompt=prompt_template, output_key='output')
    a = chain_classification({'context': context, 'question': question})
    return a


def run_query(query_text = "Name a trial AstraZeneca has done for Baxdrostat."):
    #text query as default 

    #prepare database
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    #search database
    results = db.similarity_search_with_relevance_scores(query_text, k=4)
    #threshold = 0.2
    # if we didn't find context or score of highest ranked result is too low
    if len(results) == 0: #or results[0][1] < threshold:
        print("Unable to find matching results.")
        return
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(context_text)

    model = init_llm()
    answer = get_answer(model, context_text, query_text)
    return answer['output'], answer['context']

def main():
    llm = init_llm()
    print(llm)
    answer = run_query()
    print(answer)

if __name__ == "__main__":
    main()
