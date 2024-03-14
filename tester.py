import streamlit as st
from transformers import AutoModel, AutoTokenizer
st.title('Package Tester')
import spacy

import sys
st.write(sys.executable)

import torch
from bs4 import BeautifulSoup
from collections import Counter

from sklearn.metrics.pairwise import cosine_similarity