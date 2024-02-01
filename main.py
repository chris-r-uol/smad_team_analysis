import streamlit as st
import scraper
st.set_page_config(layout='wide')
st.title('Web Scraper and CRM Analysis')

st.header('Content')
url = st.text_input('Input URL here:')

html_content = scraper.fetch_html(url)

st.subheader('Profile Data')
profile_data = scraper.scrape_profile_data(html_content)
st.write(profile_data)





for section, text in profile_data.items():
    st.write(text)
    scraper.analyse_text(text)
    #scraper.analyse_text(text)
    #key_themes = scraper.analyse_text(text)


