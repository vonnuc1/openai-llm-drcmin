from sklearn.manifold import TSNE
from umap import UMAP
import numpy as np
import pandas as pd
from langchain.vectorstores.chroma import Chroma
import os 
import streamlit as st
from plots import create_scatterplot
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

st.set_page_config(layout="wide")
st.subheader('Visualizing Large Language Model-Generated DRC Minute Embeddings')
st.info("""The following visualizations are the 2D decompositions of Large Language Model (LLM) DRC minute embeddings.
        Informative embedding models tend to place semantically similar documents close together, which means we expect documents discussing
        the same project to form clusters. The color of each embedding point corresponds to the project within which the document was created.""")

CHROMA_PATH = os.environ.get("CHROMA_PATH", "")
DATA_PATH = os.environ.get("MD_PATH", "")

embs = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embs)

X = np.array(db._collection.get(include=['embeddings'])['embeddings'])

X_2D = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(X)

print(X_2D.shape)

# get labels from metadata
chunks = db._collection.get(include=['metadatas'])['metadatas']

names = [item['name'] for item in chunks]
dates = [item['date'] for item in chunks]
titles = [item['source'].split(' ', 1)[-1].replace(' (final).md', '') for item in chunks]

# different names for the same product (?)
names_dict = {'balcidapa': ['balci-dapa', 'balcidapa'],
            'baxdrodapa': ['baxdapa', 'baxdro-dapa', 'baxdrodapa', 'baxtro-dapa'],
            'zibodapa': ['zibo', 'zibo-dapa', 'zibodapa']}

for i in range(len(names)):
    for key, value in names_dict.items():
        if names[i] in value:
            names[i] = key
            break

data = {
    'x': X_2D[:,0],
    'y': X_2D[:,1],
    'name': names,
    'date': dates,
    'title': titles
}
df = pd.DataFrame(data)

fig1 = create_scatterplot(df, 'x', 'y')

# Use UMAP for dimensionality reduction
umap = UMAP(n_components=2)
X_umap = umap.fit_transform(X)

df['x_umap'] = X_umap[:,0]
df['y_umap'] = X_umap[:,1]

fig2 = create_scatterplot(df, 'x_umap', 'y_umap')

col1, col2 = st.columns(2)

with col1:
    st.subheader('TSNE Decomposition')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader('UMAP Decomposition')
    st.plotly_chart(fig2, use_container_width=True)