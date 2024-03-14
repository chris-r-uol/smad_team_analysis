import requests
import streamlit as st
import re
import spacy
import words
import torch
from bs4 import BeautifulSoup
from collections import Counter
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances_argmin_min


EXCLUDED_KEYWORDS = ['research', 'uk', 'vehicle', 'european', 'use', 'commons', 'impact', 'transport', 'road', 'phd', 'work', 'topic', 'complete',
                         'institute', 'study', 'fellow', 'university', 'senior', 'director', 'lead', 'significant', 'include', 'leadership', 'leader',
                         'planning', 'programme', 'professor', 'solution', 'knowledge', 'smad', 'project', 'problem', 'example', 'co', 'group', 'drive',
                         'head', 'school', 'web', 'search', 'faculty', 'currently', 'undergraduate', 'lecturer', 'postgraduate', 'interest',
                         'list', 'department', 'course', 'degree', 'application', 'technique', 'allresearch', 'partner', 'manager',
                         'management', 'paper', 'teaching', 'scheme', 'employment', 'history', 'experience', 'journal', 'theme', 'sharing',
                         'engineer', 'solicitor', 'expertise', 'diploma', 'member', 'workpackage', 'multitute', 'msc', 'position', 'committee', 'hon',
                         'tutor', 'master', 'september', 'scholarship', 'hongkongbank', 'leed']

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
        data_text = cms_div.get_text().join(cms_div.strings)

        # Add a space after closing </ul> tag and before <p> tag
        data_text = re.sub(r'</ul>\s*<p>', '</ul> <p>', data_text)
        
        # Remove HTML tags
        data_text_no_html = BeautifulSoup(data_text, 'html.parser').get_text(strip=True)
        
        # Replace bullet points with a space
        data_text_no_html = re.sub(r'\s*•\s*', ' ', data_text_no_html)

        # Replace newline characters with a space
        data_text_no_html = data_text_no_html.replace('\n', ' ')

        # Add a space after closing </ul> tag and before <p> tag
        data_text_no_html = re.sub(r'</ul>\s*<p>', '</ul> <p>', data_text_no_html)

        # Replace multiple consecutive spaces with a single space
        data_text_no_html = re.sub(r'\s+', ' ', data_text_no_html)

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

nlp = spacy.load('en_core_web_sm')

#@st.cache_resource()
def analyse_text(text):
    doc = nlp(text.lower())
    top_k = 5

    candidates = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha and token.pos_ == 'NOUN']
    candidates = [key for key in candidates if key not in EXCLUDED_KEYWORDS]
    #st.write(candidates)
    tokenizer = st.session_state['input_model']['tokenizer']
    model = st.session_state['input_model']['model']
    
    #tokenizer = st.session_state['second_model']['tokenizer']
    #model = st.session_state['second_model']['model']

    if len(candidates) == 0:
        return {}
    
    candidate_tokens = tokenizer(candidates, padding=True, return_tensors = "pt", truncation = True)
    candidate_embeddings = model(**candidate_tokens)["pooler_output"]
    
    #st.write(len(candidate_embeddings))
    if len(candidate_embeddings) <= top_k:
        top_n = len(candidate_embeddings)
    else:
        top_n = top_k
    kmeans = KMeans(n_clusters=top_n, random_state=0).fit(candidate_embeddings.detach().numpy())
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    closest, _ = pairwise_distances_argmin_min(cluster_centers, candidate_embeddings.detach().numpy())
    cluster_representatives = [candidates[idx] for idx in closest]
    
    keyword_count = Counter(cluster_representatives)

    return keyword_count

def make_profile(name, profile_data):
    kw_all = Counter()
    for section, text in profile_data.items():
        kw = analyse_text(text)
        #st.write(kw)
        kw_all.update(kw)
    profile = {'id': name, 'interests': dict(kw_all)}

    return profile

def profile_from_link(url):
    name = url.split('/')[-1]

    html_content = fetch_html(url)
    profile_data = scrape_profile_data(html_content)
    profile = make_profile(name, profile_data)

    return profile
