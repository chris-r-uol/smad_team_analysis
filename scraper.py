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

EXCLUDED_KEYWORDS = ['research', 'uk', 'vehicle', 'european', 'use', 'commons', 'impact', 'road', 'phd', 'work', 'topic', 'complete',
                         'institute', 'study', 'fellow', 'university', 'senior', 'director', 'lead', 'significant', 'include', 'leadership', 'leader',
                         'planning', 'programme', 'professor', 'solution', 'knowledge', 'smad', 'project', 'problem', 'example', 'co', 'group', 'drive',
                         'head', 'school', 'web', 'search', 'faculty', 'currently', 'undergraduate', 'lecturer', 'postgraduate', 'interest',
                         'list', 'department', 'course', 'degree', 'application', 'technique', 'allresearch', 'partner', 'manager',
                         'management', 'paper', 'teaching', 'scheme', 'employment', 'history', 'experience', 'journal', 'theme', 'sharing',
                         'engineer', 'solicitor', 'expertise', 'diploma', 'member', 'workpackage', 'multitute', 'msc', 'position', 'committee',
                         'expert', 'editorial', 'hongkongbank', 'master', 'admission', 'tutor', 'transport', 'editor', 'hon', 'sector', 'instr',
                         'september', 'jun']

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

@st.cache_resource()
def analyse_text(text):

    
    doc = nlp(text.lower())
    
    candidates = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha and token.pos_ == 'NOUN']
    #candidates = [c for c in all_candidates if c not in EXCLUDED_KEYWORDS]
    
    tokenizer = st.session_state['input_model']['tokenizer']
    model = st.session_state['input_model']['model']

    candidate_tokens = tokenizer(candidates, padding=True, return_tensors = "pt", truncation = True)
    candidate_embeddings = model(**candidate_tokens)["pooler_output"]

    text_tokens = tokenizer([text], padding = True, return_tensors = 'pt', truncation = True)
    text_embeddings = model(**text_tokens)['pooler_output']
    
    #st.subheader('Distance')
    candidate_embeddings = candidate_embeddings.detach().numpy()
    text_embeddings = text_embeddings.detach().numpy()

    top_k = 15
    distances = cosine_similarity(text_embeddings, candidate_embeddings)
    keywords = [candidates[index] for index in distances.argsort()[0][-top_k:]]
    #keywords = [words.smad_words[kw] for kw in keywords]
    keyword_count = Counter(keywords)
    
    filtered_themes = {key: value for key, value in keyword_count.items() if key not in EXCLUDED_KEYWORDS}
    #st.write(keyword_count)
    #st.write(filtered_themes)

    #st.write(keywords)
    #return keyword_count
    return filtered_themes

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
