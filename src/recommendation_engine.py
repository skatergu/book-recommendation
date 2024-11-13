import pandas as pd
import numpy as np
from text_processing import TextProcessor
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationEngine:
    def __init__(self, books_df):
        self.books_df = books_df
        self.text_processor = TextProcessor()
        self.content_vectors = None
        self.initialize_vectors()
        
    def initialize_vectors(self):
        """Initialize TF-IDF vectors for content-based filtering"""
        # Combine relevant text fields
        combined_text = self.books_df.apply(
            lambda x: f"{x['title']} {x['author']} {x['description']} {x['genre']}", 
            axis=1
        )
        
        # Preprocess all texts
        processed_texts = combined_text.apply(self.text_processor.preprocess_text)
        
        # Create TF-IDF vectors
        self.content_vectors = self.text_processor.vectorizer.fit_transform(processed_texts)
        
        # Process sentiments for all books
        self.books_df['sentiment_analysis'] = self.books_df['description'].apply(
            self.text_processor.analyze_sentiment
        )
        
        # Extract semantic features
        self.books_df['semantic_features'] = self.books_df['description'].apply(
            self.text_processor.extract_semantic_features
        )
    
    def get_recommendations(self, filters, user_mood=None):
        """Get recommendations based on filters and user mood"""
        # Initial filtering based on basic criteria
        filtered_books = self._apply_basic_filters(filters)
        
        if filtered_books.empty:
            return []
        
        # Get content-based similarities
        similarities = self._get_content_similarities(filters)
        
        # Apply mood-based filtering if specified
        if user_mood:
            filtered_books = self._apply_mood_filtering(filtered_books, user_mood)
        
        # Combine all scores
        final_scores = self._combine_scores(filtered_books, similarities)
        
        # Get top recommendations
        recommendations = self._get_top_recommendations(filtered_books, final_scores)
        
        return recommendations.to_dict('records')
    
    def _apply_basic_filters(self, filters):
        """Apply basic filters (price, rating, etc.)"""
        mask = pd.Series(True, index=self.books_df.index)
        
        if 'price_range' in filters:
            min_price, max_price = filters['price_range']
            mask &= (self.books_df['price'] >= min_price) & (self.books_df['price'] <= max_price)
            
        if 'rating' in filters and filters['rating']:
            mask &= self.books_df['rating'] >= filters['rating']
            
        return self.books_df[mask]
    
    def _get_content_similarities(self, filters):
        """Calculate content-based similarities"""
        if 'title' in filters and filters['title']:
            query_text = self.text_processor.preprocess_text(filters['title'])
            query_vector = self.text_processor.vectorizer.transform([query_text])
            return cosine_similarity(self.content_vectors, query_vector).flatten()
        return np.ones(len(self.books_df))
    
    def _apply_mood_filtering(self, books, user_mood):
        """Filter books based on mood matching"""
        mood_mask = books['sentiment_analysis'].apply(
            lambda x: x['mood'] == user_mood
        )
        return books[mood_mask]
    
    def _combine_scores(self, filtered_books, similarities):
        """Combine different scoring factors"""
        # Normalize ratings
        rating_scores = filtered_books['rating'] / 5.0
        
        # Combine with content similarities
        final_scores = (0.6 * similarities[filtered_books.index] + 
                       0.4 * rating_scores)
        
        return final_scores
    
    def _get_top_recommendations(self, books, scores, n=10):
        """Get top N recommendations"""
        books = books.copy()
        books['score'] = scores
        return books.nlargest(n, 'score') 