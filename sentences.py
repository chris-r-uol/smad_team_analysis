import streamlit as st
from transformers import AutoModel, AutoTokenizer
import torch
from torch.nn import functional as F

@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    return {'tokenizer':tokenizer,
            'model':model}

#Mean Pooling - Take attention mask into account for correct averaging
@st.cache_resource
def mean_pooling(_model_output, _attention_mask):
    token_embeddings = _model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = _attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


st.title('Sentence Analysis')

model = load_model('sentence-transformers/all-MiniLM-L6-v2')
sentences = ['This is an example sentence', 'Each sentence is converted']

# Two lists of sentences
sentences1 = [
    "The cat sits outside",
    "A man is playing guitar",
    "The new movie is awesome",
]

sentences2 = [
    "The dog plays in the garden",
    "A woman watches TV",
    "The new movie is so great",
]

tokenizer = model['tokenizer']
model = model['model']

st.write(sentences)

encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
st.write(sentence_embeddings)

st.header('Sentence Analysis II')
from sentence_transformers import SentenceTransformer, util
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# Compute embedding for both lists
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
embeddings2 = model.encode(sentences2, convert_to_tensor=True)
st.write(embeddings1)
cosine_scores = util.cos_sim(embeddings1, embeddings2)
for i in range(len(sentences1)):
    st.write("{} \t\t {} \t\t Score: {:.4f}".format(
        sentences1[i], sentences2[i], cosine_scores[i][i]
    ))