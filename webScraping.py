from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

def scrape_website(url):
    """
    Scrape any website URL using headless Chrome.
    Returns the page HTML as a string.
    """
    options = Options()
    options.add_argument("--headless=new")   # Headless browser
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        driver.get(url)
        html = driver.page_source
        return html
    finally:
        driver.quit()


def extract_body_content(html):
    soup = BeautifulSoup(html, "html.parser")
    body = soup.body
    return str(body) if body else ""


def clean_body_content(body_content):
    soup = BeautifulSoup(body_content, "html.parser")
    for tag in soup(["script", "style"]):
        tag.extract()
    cleaned = "\n".join(line.strip() for line in soup.get_text().splitlines() if line.strip())
    return cleaned


def split_dom_content(dom_content, max_length=6000):
    return [dom_content[i:i+max_length] for i in range(0, len(dom_content), max_length)]
