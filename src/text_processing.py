import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Starting TextProcessor initialization...")

def download_nltk_data():
    print("Downloading NLTK data...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    print("NLTK data downloaded successfully!")

try:
    nlp = spacy.load('en_core_web_sm')
    print("spaCy model loaded successfully!")
except Exception as e:
    print(f"Error loading spaCy model: {e}")
    raise

class TextProcessor:
    def __init__(self):
        download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000)
        print("TextProcessor initialized successfully!")
        
    def preprocess_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        # Simple tokenization using split()
        tokens = text.lower().split()
        # Remove stopwords and non-alphabetic tokens
        tokens = [token for token in tokens if token.isalpha() and token not in self.stop_words]
        return ' '.join(tokens)
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using TextBlob"""
        # Handle NaN or non-string values
        if not isinstance(text, str):
            return {
                'mood': 'neutral',
                'sentiment_score': 0.0,
                'subjectivity': 0.0
            }
        
        try:
            analysis = TextBlob(text)
            
            # Get polarity (-1 to 1) and subjectivity (0 to 1)
            sentiment_score = analysis.sentiment.polarity
            subjectivity = analysis.sentiment.subjectivity
            
            # Classify the mood
            if sentiment_score > 0.3:
                mood = 'positive'
            elif sentiment_score < -0.3:
                mood = 'negative'
            else:
                mood = 'neutral'
                
            return {
                'mood': mood,
                'sentiment_score': sentiment_score,
                'subjectivity': subjectivity
            }
        except:
            # Return neutral values if analysis fails
            return {
                'mood': 'neutral',
                'sentiment_score': 0.0,
                'subjectivity': 0.0
            }
    
    def extract_semantic_features(self, text):
        """Extract semantic features using spaCy"""
        doc = nlp(text)
        
        # Extract entities, key phrases, and themes
        entities = [ent.text for ent in doc.ents]
        noun_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        return {
            'entities': entities,
            'noun_phrases': noun_phrases,
            'main_themes': self._extract_main_themes(doc)
        }
    
    def _extract_main_themes(self, doc):
        """Extract main themes based on noun and verb frequencies"""
        themes = []
        for token in doc:
            if token.pos_ in ['NOUN', 'VERB'] and token.text not in self.stop_words:
                themes.append(token.text)
        return list(set(themes))[:5]  # Return top 5 themes 