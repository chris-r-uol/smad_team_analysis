import requests
import streamlit as st
import re
import spacy

import torch
from bs4 import BeautifulSoup
from collections import Counter
from transformers import AutoModel, AutoTokenizer
#from sklearn.metrics.pairwise import cosine_similarity


token = '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'

@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    return {'tokenizer':tokenizer,
            'model':model}

def fetch_html(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        st.error(f'Error fetching HTML: {e}')
        return None
    
def scrape_profile_data(html):
    if html is None:
        return None
    
    soup = BeautifulSoup(html, 'html.parser')
    
    # Select the specific div with class="cms"
    cms_div = soup.find('div', class_='cms')
    
    if cms_div:
        # Extract text content from the selected div
        data_text = cms_div.get_text(strip=True)
        
        # Remove HTML tags
        data_text_no_html = BeautifulSoup(data_text, 'html.parser').get_text(strip=True, separator='\n')

        # Define a list of headings to split on
        headings = ['Profile', 'Responsibilities', 'Research interests', 'Research projects', 'Publications', 'Qualifications', 'Professional memberships', 'Student education', 'Research groups and institutes', 'Current posgraduate researchers', 'Postgraduate research ']

        # Create a regular expression pattern that matches any of the headings
        pattern = '|'.join(re.escape(heading) for heading in headings)

        # Split the text based on the headings using the regular expression pattern
        sections = re.split(pattern, data_text_no_html)

        # Extract data under each heading
        profile_text = sections[1].strip() if len(sections) > 1 else None
        responsibilities_text = sections[2].strip() if len(sections) > 2 else None
        research_interests_text = sections[3].strip() if len(sections) > 3 else None
        research_projects_texts = sections[4].strip() if len(sections) > 4 else None
        qualifications_text = sections[5].strip() if len(sections) > 5 else None


        profile_data = {
            'profile': profile_text,
            'responsibilities': responsibilities_text,
            'research_interests': research_interests_text,
            #'research_projects': research_projects_texts,
            #'qualifications': qualifications_text
        }
        return profile_data
    else:
        st.warning('Warning: No Class CMS DIV found on the page')
        return None

def analyse_text(text):
    
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text.lower())
    noun_phrases = set(chunk.text.strip().lower for chunk in doc.noun_chunks)
    nouns = set()
    for token in doc:
        if token.pos_ == 'NOUN':
            nouns.add(token.text)
    
    all_nouns = nouns.union(noun_phrases)
    st.subheader('spaCy')
    st.write(all_nouns)
    candidates = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    st.write(candidates)
    
    
    st.subheader('Keywords')
    input_model = load_model('distilroberta-base')
    
    tokenizer = input_model['tokenizer']
    model = input_model['model']

    candidate_tokens = tokenizer(candidates, padding=True, return_tensors = "pt")
    candidate_embeddings = model(**candidate_tokens)["pooler_output"]

    text_tokens = tokenizer([text], return_tensors = 'pt')
    text_embedding = model(**text_tokens)['pooler_output']
    
    st.subheader('Distance')
    candidate_embeddings = candidate_embeddings.detach().numpy()
    text_embeddings = text_embeddings.detach().numpy()

    top_k = 5
    distances = cosine_similarity(text_embedding, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_k:]]

    st.write(keywords)

    return
    '''
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text.lower())
    key_themes = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    theme_counts = Counter(key_themes)
    min_frequency = 2
    filtered_themes = {theme: count for theme, count in theme_counts.items() if count >= min_frequency}
    excluded_keywords = ['research', 'uk', 'vehicle', 'european', 'use', 'commons', 'impact', 'transport', 'road', 'phd', 'work', 'topic', 'complete',
                         'institute', 'study', 'fellow', 'university', 'senior', 'director', 'lead', 'significant', 'include', 'leadership']
    filtered_themes = {key: value for key, value in filtered_themes.items() if key not in excluded_keywords}
    
    return filtered_themes'''