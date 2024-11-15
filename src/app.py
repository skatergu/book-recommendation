from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
from recommendation_engine import RecommendationEngine

app = Flask(__name__, 
    template_folder='templates',
    static_folder='static'
)
CORS(app)

print("Loading data...")
try:
    books = pd.read_csv('data/books.csv')
    print(f"Data loaded successfully! Shape: {books.shape}")
    print("Columns in DataFrame:", books.columns.tolist())  # Print column names
    
    print("Creating recommendation engine...")
    recommendation_engine = RecommendationEngine(books, limit_size=True)
    print("Recommendation engine created successfully!")
except Exception as e:
    print("Error during initialization:")
    print(e)  # Print the error message
    raise

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        try:
            genres = books['Main Genre'].unique().tolist()  # Use 'Main Genre'
            authors = books['Author'].unique().tolist()  # Use 'Author'
            
            return render_template('index.html', genres=genres, authors=authors)
        except Exception as e:
            print(f"Error in home route: {e}")
            return str(e), 500
    else:  # Handle POST requests for recommendations
        try:
            data = request.json
            print("\nReceived request with data:", data)
            
            recommendations = recommendation_engine.get_recommendations(data)
            print("\nGenerated recommendations:", recommendations)
            
            return jsonify(recommendations)
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = 8000
    print(f"\nStarting server at http://localhost:{port}")
    print("Press CTRL+C to quit")
    app.run(debug=True, host='localhost', port=port)
