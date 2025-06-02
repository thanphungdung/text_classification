import pandas as pd
import re
#from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):
    # loader = PyPDFLoader(file_path)
    # documents = loader.load()
    # return documents
    return [{"page_content": "PDF reading temporarily disabled"}]


def load_csv_safely(file_path):
    encodings = ['utf-8', 'latin1', 'windows-1252']
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            return df
        except Exception:
            continue
    raise ValueError("Unsupported file encoding or corrupt CSV.")
    

def preprocess_text(text: str) -> str:
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # 4. Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # 5. Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text.split()