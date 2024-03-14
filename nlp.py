import streamlit as st
<<<<<<< Updated upstream

import spacy
from sklearn.cluster import AgglomerativeClustering
import numpy as np

# Load the English model for spaCy
nlp = spacy.load("en_core_web_sm")

# Define the list of words
test_words = ['good', 'bad', 'great', 'excellent', 'terrible', 'wonderful']
words = [
  "authority",
  "development",
  "optimisation",
  "simulation",
  "dynamic",
  "traffic",
  "policy",
  "network",
  "system",
  "emission",
  "pollution",
  "industry",
  "fleet",
  "physics",
  "environment",
  "method",
  "datum",
  "reach",
  "accessibility",
  "data",
  "visualisation",
  "capability",
  "microsimulation",
  "sensing",
  "telematic",
  "quality",
  "air",
  "health",
  "carbon",
  "mobility",
  "climate",
  "geography",
  "standardisation",
  "infrastructure",
  "location",
  "toolkit",
  "leed",
  "swansea",
  "relationship",
  "forest",
  "science",
  "biodiversity",
  "present",
  "design",
  "engineering",
  "architecture",
  "equity",
  "connectivity",
  "framework",
  "regeneration",
  "strategy",
  "reviewer",
  "institution",
  "ceng",
  "transportation",
  "engagement",
  "innovation",
  "city",
  "consumption",
  "learning",
  "student",
  "crowdshipping",
  "jun",
  "education",
  "consultancy",
  "organisation",
  "activity",
  "disruption",
  "assessment",
  "evaluation",
  "firm",
  "travel",
  "behaviour",
  "cycling",
  "perspective",
  "analysis",
  "population",
  "survey",
  "geoscience",
  "mitigation",
  "satnav",
  "decarbonisation",
  "car",
  "capacity",
  "medium",
  "technology",
  "litigation",
  "rail",
  "responsibility",
  "risk",
  "resiliance",
  "emergency",
  "coordination",
  "planner",
  "congestion",
  "pricing",
  "modelling",
  "resilience",
  "sustainability",
  "monitoring",
  "language",
  "event",
  "workshop",
  "participation",
  "motorway",
  "community",
  "sheffield",
  "scholarship",
  "surface",
  "specialization",
  "flow",
  "control",
  "safety",
  "optimization",
  "program",
  "paradigm",
  "reliability",
  "equilibrium"]

# Tokenize the words using spaCy and get their vectors
word_vectors = [nlp(word).vector for word in words]

# Perform hierarchical clustering
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5).fit(word_vectors)
labels = clustering.labels_

# Analyze the clusters
clusters = {}
for i, word in enumerate(words):
    label = labels[i]
    if label not in clusters:
        clusters[label] = []
    clusters[label].append(word)

# Determine "good" and "bad" words
good_words = set()
bad_words = set()

for cluster in clusters.values():
    # Calculate the average vector for each cluster
    cluster_vectors = [nlp(word).vector for word in cluster]
    centroid = np.mean(cluster_vectors, axis=0)
    
    # Calculate similarity with predefined "good" and "bad" words
    similarity_to_good = np.mean([nlp("good").similarity(nlp(word)) for word in cluster])
    similarity_to_bad = np.mean([nlp("bad").similarity(nlp(word)) for word in cluster])

    # Classify the cluster
    if similarity_to_good > similarity_to_bad:
        good_words.update(cluster)
    else:
        bad_words.update(cluster)

print("Good words:", good_words)
print("Bad words:", bad_words)
=======
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def make_synonyms(nouns):
    tokens = word_tokenize(' '.join(nouns))
    tokens = [word.lower() for word in tokens]
>>>>>>> Stashed changes

    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

<<<<<<< Updated upstream

=======
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in tokens]

    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

    st.write(nouns)
    st.write(stemmed_words)
    st.write(lemmatized_words)
>>>>>>> Stashed changes
