import streamlit as st
from webScraping import scrape_website, extract_body_content, clean_body_content, split_dom_content
from parse import parse_with_ollama

st.title("AI Web Scraper")

url = st.text_input("Enter Website URL")

if st.button("Scrape Website") and url:
    st.info("Scraping website...")
    html = scrape_website(url)
    body = extract_body_content(html)
    cleaned = clean_body_content(body)
    
    st.session_state.dom_content = cleaned

    with st.expander("View DOM Content"):
        st.text_area("DOM Content", cleaned, height=300)

if "dom_content" in st.session_state:
    parse_description = st.text_area("Describe what you want to parse")

    if st.button("Parse Content") and parse_description:
        st.info("Parsing content...")
        dom_chunks = split_dom_content(st.session_state.dom_content)
        parsed = parse_with_ollama(dom_chunks, parse_description)
        st.text_area("Parsed Result", parsed, height=300)
