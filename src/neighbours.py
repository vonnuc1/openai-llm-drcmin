import streamlit as st
from sklearn.manifold import TSNE
from umap import UMAP
import numpy as np
import pandas as pd
from langchain.vectorstores.chroma import Chroma
import os 
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import plots

st.set_page_config(layout="wide")
st.subheader('Visualizing Large Language Model-Generated DRC Minute Embeddings, Prompt Embedding and its Nearest Neighbors')

CHROMA_PATH = os.environ.get("CHROMA_PATH", "")
DATA_PATH = os.environ.get("MD_PATH", "")

embs = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embs)


query_text = st.text_input("What do you want to know?") # eg. "Name a trial AstraZeneca has done for Baxdrostat."
new_emb = np.array(embs.embed_documents([query_text])) # embedding of prompt

X = np.array(db._collection.get(include=['embeddings'])['embeddings'])
new_X = np.concatenate((X, new_emb), axis=0)
new_X_2D = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(new_X)

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
    'x': new_X_2D[:,0],
    'y': new_X_2D[:,1],
    'name': names + ['prompt'],
    'date': dates + [''],
    'title': titles + [query_text],
    'metadata': chunks + ['']
}
df = pd.DataFrame(data)

# get closest neighbors
results = db.similarity_search_with_relevance_scores(query_text, k=4)
metadatas = [item[0].metadata for item in results]

fig1 = plots.create_scatterplot_with_prompt_and_neighbours(df, 'x', 'y', metadatas)

# Use UMAP for dimensionality reduction
umap = UMAP(n_components=2)
X_umap = umap.fit_transform(new_X)

df['x_umap'] = X_umap[:,0]
df['y_umap'] = X_umap[:,1]

fig2 = plots.create_scatterplot_with_prompt_and_neighbours(df, 'x_umap', 'y_umap', metadatas)

col1, col2 = st.columns(2)

with col1:
    st.subheader('TSNE Decomposition')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader('UMAP Decomposition')
    st.plotly_chart(fig2, use_container_width=True)

context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
st.write("The following context was used to answer your question:\n\n" + context_text)