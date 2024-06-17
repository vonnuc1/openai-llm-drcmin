from langchain.vectorstores.chroma import Chroma
from sklearn.manifold import TSNE
import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from metadata import get_labels
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

CHROMA_PATH = os.environ.get("CHROMA_PATH", "")
DATA_PATH = os.environ.get("MD_PATH", "")

embs = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embs)

vec = db._collection.get(include=['embeddings'])
print(np.shape(vec['embeddings']))

labels = get_labels()
print(labels)

# Extracting all file names in the target directory
file_names = [f for f in os.listdir(DATA_PATH) if os.path.isfile(os.path.join(DATA_PATH, f))]

if '.DS_Store' in file_names:
    file_names.remove('.DS_Store')

#print(len(file_names))

project_names = [f.split(' ')[1].lower() for f in file_names]
project_names = ['not specified' if x == '' else x for x in project_names]

#name_counts = Counter(project_names)

vec_tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(vec)

print(vec_tsne.shape)

plt.scatter(vec_tsne[:,0], vec_tsne[:,1])
plt.show()