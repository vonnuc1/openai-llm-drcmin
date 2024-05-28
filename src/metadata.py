from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain.vectorstores.chroma import Chroma
import shutil
#from langchain_openai import AzureOpenAIEmbeddings

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

load_dotenv(override=True, verbose=True)

DATA_PATH = os.environ.get("MD_PATH", "")
CHROMA_PATH = os.environ.get("CHROMA_PATH", "")


"""embeddings = AzureOpenAIEmbeddings(
    deployment_name = "text-embedding-ada-002",
    api_version="2023-05-15",
)"""

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    docs = loader.load()
    return docs

def split_text(docs: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 500,
        length_function = len,
        add_start_index = True
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunks)} chunks.")

    return chunks


def get_labels():
    names = []
    dates = []
    documents = load_documents()
    chunks = split_text(documents)
    for i in range(len(chunks)):
        name = str(chunks[i].metadata).split(' ')[2]
        date = str(chunks[i].metadata).split(' ')[1].split('/')[-1]
        if name:
            names.append(name.lower())
        else:
            names.append('not specified')
        dates.append(date)
    return names, dates

    

if __name__ == "__main__":
    labels = get_labels()
    print(labels[1])