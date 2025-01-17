import pandas as pd
import numpy as np
from text_processing import TextProcessor
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationEngine:
    def __init__(self, books_df, limit_size=True):
        print("Starting RecommendationEngine initialization")
        if limit_size:
            self.books_df = books_df.head(100).copy()  # Make a copy of the DataFrame
        else:
            self.books_df = books_df.copy()  # Ensure we are working with a copy
        print("Books DataFrame shape:", self.books_df.shape)
        print(self.books_df.head(10))  # Print the first 10 rows of the DataFrame
        
        self.text_processor = TextProcessor()
        self.content_vectors = None
        self.initialize_vectors()
        print("RecommendationEngine initialization complete")
        
    def initialize_vectors(self):
        """Initialize TF-IDF vectors for content-based filtering"""
        print("    - Converting column names to lowercase and stripping spaces")
        self.books_df.columns = self.books_df.columns.str.lower().str.strip()
        
        print("    - Filling NaN values")
        text_columns = ['title', 'author', 'description', 'main genre']
        self.books_df.loc[:, text_columns] = self.books_df[text_columns].fillna('')
        
        print("    - Combining text")
        combined_text = self.books_df.apply(
            lambda x: f"{x['title']} {x['author']} {x['description']} {x['main genre']}", 
            axis=1
        )
        
        print("    - Processing texts")
        processed_texts = combined_text.apply(self.text_processor.preprocess_text)
        
        print("    - Creating TF-IDF vectors")
        self.content_vectors = self.text_processor.vectorizer.fit_transform(processed_texts)
        
        print("    - Processing sentiments")
        self.books_df.loc[:, 'sentiment_analysis'] = self.books_df['description'].apply(
            self.text_processor.analyze_sentiment
        )
        
        print("    - Extracting semantic features")
        self.books_df.loc[:, 'semantic_features'] = self.books_df['description'].apply(
            self.text_processor.extract_semantic_features
        )
        
        # Convert price and rating columns to numeric
        self.books_df['price'] = pd.to_numeric(self.books_df['price'], errors='coerce')
        self.books_df['rating'] = pd.to_numeric(self.books_df['rating'], errors='coerce')
        
        print("    - Vector initialization complete")
    
    def get_recommendations(self, filters):
        # Extract filters
        title = filters.get('title', '').strip()
        genre = filters.get('genre', '').strip()
        author = filters.get('author', '').strip()
        price_range = filters.get('price_range', [0, 800])  # Default max price to 800
        rating = filters.get('rating', 0)
        mood = filters.get('mood', '')

        # Validate price range
        min_price, max_price = price_range
        if min_price is None:
            min_price = 0
        if max_price is None:
            max_price = 800  # Set default max price to 800

        # Build the filtering mask
        mask = pd.Series(True, index=self.books_df.index)  # Start with all True

        if title:
            mask &= self.books_df['title'].str.contains(title, case=False, na=False)
        if genre:
            mask &= self.books_df['main genre'].str.contains(genre, case=False, na=False)
        if author:
            mask &= self.books_df['author'].str.contains(author, case=False, na=False)
        mask &= (self.books_df['price'] >= min_price) & (self.books_df['price'] <= max_price)
        mask &= (self.books_df['rating'] >= rating)

        recommendations = self.books_df[mask]

        # Debugging output
        print("Mask:", mask)
        print("Filtered recommendations:", recommendations)

        return recommendations.to_dict(orient='records')
    
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