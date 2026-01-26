import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
import os

# --- NLTK Configuration ---
# Setting up local directory for NLTK resources to ensure portability
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_path): 
    os.makedirs(nltk_data_path)
nltk.data.path.append(nltk_data_path)

print("Downloading required NLTK resources...")
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)

# Initialize Lemmatizer and Stopwords list
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans the input text by:
    1. Converting to lowercase
    2. Removing special characters/numbers
    3. Removing stopwords
    4. Lemmatization (reducing words to their base form)
    """
    # Step 1: Lowercase and Special Character Removal
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    
    # Step 2: Tokenization
    words = text.split()
    
    # Step 3: Lemmatization and Stopword Removal
    # This reduces words like 'crying' to 'cry' and removes common words like 'is', 'the'
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    return " ".join(cleaned_words)

# --- Data Loading and Execution ---
input_file = 'Cleaned_Mental_Health_Data.csv'

if os.path.exists(input_file):
    df = pd.read_csv(input_file)
    print(f"Dataset loaded. Processing {len(df)} rows...")
    
    # Apply preprocessing to the 'statement' column
    df['processed_statement'] = df['statement'].apply(preprocess_text)
    
    # Save the processed output for Model Training
    output_file = 'Processed_Data_Final.csv'
    df.to_csv(output_file, index=False)
    print(f"Success! '{output_file}' has been created.")
else:
    print(f"Error: '{input_file}' not found in the current directory.")