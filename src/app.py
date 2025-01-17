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
    books = pd.read_csv('../data/books.csv')
    print(f"Data loaded successfully! Shape: {books.shape}")
    
    print("Creating recommendation engine...")
    recommendation_engine = RecommendationEngine(books, limit_size=True)
    print("Recommendation engine created successfully!")
except Exception as e:
    print("Error during initialization:")
    print(e) 
    raise

@app.route('/initial-data', methods=['GET'])
def get_initial_data():
    try:
        genres = books['Main Genre'].dropna().unique().tolist()
        authors = books['Author'].dropna().unique().tolist()
        return jsonify({"genres": genres, "authors": authors})
    except Exception as e:
        print(f"Error in /initial-data endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET' or request.method == 'POST':
        try:
            genres = books['Main Genre'].unique().tolist() 
            authors = books['Author'].unique().tolist() 
            
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
        
@app.route('/genres', methods=['GET'])
def get_genres():
    try:
        genres = books['Main Genre'].unique().tolist()
        return jsonify({"genres": genres})
    except Exception as e:
        print(f"Error in /genres endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/authors', methods=['GET'])
def get_authors():
    try:
        authors = books['Author'].unique().tolist()
        return jsonify({"authors": authors})
    except Exception as e:
        print(f"Error in /authors endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    try:
        data = request.json
        print("\nReceived request with data:", data)

        recommendations = recommendation_engine.get_recommendations(data)
        print("\nGenerated recommendations:", recommendations)
        
        return jsonify(recommendations)
    except Exception as e:
        print(f"Error in /recommendations endpoint: {e}")
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    port = 8000
    print(f"\nStarting server at http://localhost:{port}")
    print("Press CTRL+C to quit")
    app.run(debug=True, host='localhost', port=port)
