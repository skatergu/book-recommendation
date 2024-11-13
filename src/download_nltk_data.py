import nltk

# Download required NLTK data
resources = [
    'punkt',
    'stopwords',
    'wordnet',
    'averaged_perceptron_tagger',
    'punkt_tab'
]

for resource in resources:
    print(f"Downloading {resource}...")
    nltk.download(resource) 