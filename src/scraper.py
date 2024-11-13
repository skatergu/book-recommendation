import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

class BookScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def get_book_description(self, title, author):
        try:
            # Create search query
            search_query = f"{title} {author} goodreads"
            search_url = f"https://www.google.com/search?q={search_query}"
            
            # Get Goodreads URL from Google search
            response = requests.get(search_url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find first Goodreads link
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if 'goodreads.com/book/show' in href:
                    goodreads_url = href.split('&')[0].replace('/url?q=', '')
                    break
            else:
                return "No description available"

            # Get Goodreads page
            time.sleep(2) 
            response = requests.get(goodreads_url, headers=self.headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find description
            description_div = soup.find('div', {'class': 'BookPageMetadataSection__description'})
            if description_div:
                description = description_div.get_text().strip()
                return description
            
            return "No description available"

        except Exception as e:
            print(f"Error fetching description for {title}: {str(e)}")
            return "No description available"

    def update_books_csv(self, csv_path):
        df = pd.read_csv(csv_path)
        
        print("Available columns in CSV:", df.columns.tolist())
        
        # Add description column if it doesn't exist
        if 'description' not in df.columns:
            df['description'] = ''

        # Update descriptions for books without them
        for index, row in df.iterrows():
            if pd.isna(df.at[index, 'description']) or df.at[index, 'description'] == '':
                print(f"Fetching description for: {row['Title']}")
                description = self.get_book_description(row['Title'], row['Author'])
                df.at[index, 'description'] = description
                # Add delay to avoid hitting rate limits
                time.sleep(2)

        df.to_csv(csv_path, index=False)
        return df