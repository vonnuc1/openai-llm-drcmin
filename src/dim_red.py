from sklearn.manifold import TSNE
#from sklearn.decomposition import PCA
from umap import UMAP
import numpy as np

import matplotlib.pyplot as plt
from langchain.vectorstores.chroma import Chroma
import os 
import streamlit as st
from metadata import get_labels

import plotly.express as px

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
st.set_page_config(layout="wide")
st.subheader('Visualizing Large Language Model-Generated DRC Minute Embeddings')
st.info("""The following visualizations are the 2D decompositions of Large Language Model (LLM) DRC minute embeddings.
        Informative embedding models tend to place semantically similar documents close together, which means we expect documents discussing
        the same project to form clusters. The color of each embedding point corresponds to the project within which the document was created.""")

CHROMA_PATH = os.environ.get("CHROMA_PATH", "")
DATA_PATH = os.environ.get("MD_PATH", "")

embs = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embs)

labels = get_labels()
print(labels[0], labels[1])

X = np.array(db._collection.get(include=['embeddings'])['embeddings'])

X_2D = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(X)

print(X_2D.shape)

fig1 = px.scatter(x=X_2D[:,0], y=X_2D[:,1], color=labels[0], hover_name=labels[1])

# Use UMAP for dimensionality reduction
umap = UMAP(n_components=2)
X_umap = umap.fit_transform(X)

fig2 = px.scatter(x=X_umap[:,0], y=X_umap[:,1], color=labels[0], hover_name=labels[1])

col1, col2 = st.columns(2)

with col1:
    st.subheader('TSNE Decomposition')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader('UMAP Decomposition')
    st.plotly_chart(fig2, use_container_width=True)