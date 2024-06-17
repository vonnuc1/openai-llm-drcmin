import os
import openai
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import pandas as pd
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain_core.messages import HumanMessage
from pypdf import PdfReader

load_dotenv(override=True, verbose=True)

MODEL_NAME = "gpt-4-turbo"

def set_openai_credentials():
    """Set OpenAI API credentials."""
    openai.api_type = os.environ.get("OPENAI_API_TYPE", "azure")
    openai.azure_endpoint = os.environ.get("OPENAI_API_BASE", "")
    openai.api_version = os.environ.get("OPENAI_API_VERSION", "")
    openai.api_key = os.environ.get("OPENAI_API_KEY", "")

set_openai_credentials()

def init_llm(model=MODEL_NAME, deployment_name=MODEL_NAME, temperature = 0, max_tokens=2000, max_retries=5):
    llm = AzureChatOpenAI(
        model=model,
        deployment_name=deployment_name,
        openai_api_key=openai.api_key,
        openai_api_base=openai.azure_endpoint,
        temperature = temperature,
        max_tokens=max_tokens,
        max_retries=max_retries
        )
    return llm

llm = init_llm()