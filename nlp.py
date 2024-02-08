from transformers import Word2VecTokenizerFast
from transformers import Word2VecModel
import torch
from sklearn.cluster import KMeans

# Load the pre-trained model
nouns = ["environment", "decarbonisation", "climate", "pollution", "sustainability"]
tokenizer = Word2VecTokenizerFast.from_pretrained("sentence-transformers/fasttext-crawl-300d-2M-subword")
tokenized_nouns = tokenizer(nouns, padding=True, truncation=True, return_tensors="pt")

# Load the pre-trained model
model = Word2VecModel.from_pretrained("sentence-transformers/fasttext-crawl-300d-2M-subword")
# Generate embeddings
with torch.no_grad():
    embeddings = model(**tokenized_nouns)[0]


# Reshape the embeddings
reshaped_embeddings = embeddings.numpy().reshape(-1, 300)

# Fit the KMeans model
kmeans = KMeans(n_clusters=3, random_state=0).fit(reshaped_embeddings)

# Get the cluster labels
cluster_labels = kmeans.labels_

for label, word in zip(cluster_labels, nouns):
    print(f"{word}: {label}")