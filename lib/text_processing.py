import warnings
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup

def text_simplifier():
    nltk.download("stopwords", quiet=True)
    nltk.download('wordnet', quiet=True)
    lemmatizer = WordNetLemmatizer()

    def text_to_words(text):
        if text is not None:
            warnings.filterwarnings("ignore", category=UserWarning, module='bs4')
            soup = BeautifulSoup(text, 'html5lib')
            text = soup.get_text(" ", strip=True)
            text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
            words = text.split()
            words = [w for w in words if w not in stopwords.words("english")]
            words = [lemmatizer.lemmatize(w) for w in words]
        else:
            words = []
        return words

    def f(text):
        return " ".join(text_to_words(text))

    return f