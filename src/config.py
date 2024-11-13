import os

# Get the project root directory (this will point to the BOOK-RECS folder)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOKS_PATH = os.path.join(PROJECT_ROOT, 'data', 'books.csv') 