import streamlit as st

st.title('Package Tester')
import spacy

import sys
st.write(sys.executable)

import torch
from bs4 import BeautifulSoup
from collections import Counter
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity