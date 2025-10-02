# Multilingual Social Media Sentiment Analysis

## Overview
This project performs sentiment analysis on social media text data using Natural Language Processing (NLP) techniques and VADER (Valence Aware Dictionary and sEntiment Reasoner) sentiment analysis tool. The project includes comprehensive text preprocessing, sentiment classification, and visualization through word clouds.

## Features
- **Text Preprocessing**: Comprehensive cleaning including lowercasing, punctuation removal, stopword removal, lemmatization, and emoji removal
- **Sentiment Analysis**: Uses VADER sentiment analyzer to classify text as Positive or Negative
- **Data Visualization**: Generates word clouds for overall text and positive sentiment text
- **Data Cleaning**: Handles duplicates, missing values, and irrelevant columns
- **Multilingual Support**: Designed to handle text data from various social media platforms

## Requirements
The project requires the following Python libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import warnings
import string
import re
import nltk
from wordcloud import WordCloud
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
```

## Installation
1. Clone the repository:
```bash
git clone <repository-url>
cd Multilingual_Social_Media_Sentiment_Analysis
```

2. Install required packages:
```bash
pip install numpy pandas matplotlib seaborn plotly nltk wordcloud
```

3. Download necessary NLTK data:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

## Usage
1. **Data Loading**: The notebook loads data from `sentiment_dataset.csv`
2. **Data Cleaning**: 
   - Removes unnecessary columns (`Unnamed: 0.1`, `Unnamed: 0`)
   - Handles duplicates and missing values
   - Selects relevant columns (`Text`, `Sentiment`)
3. **Text Preprocessing**:
   - Converts text to lowercase
   - Removes punctuation, digits, and emojis
   - Eliminates stopwords
   - Applies lemmatization
   - Cleans URLs and extra whitespaces
4. **Sentiment Analysis**:
   - Uses VADER to calculate sentiment scores
   - Classifies sentiments as Positive or Negative
   - Maps sentiments to numerical values (-1 for Negative, 1 for Positive)
5. **Visualization**:
   - Generates word clouds for overall text and positive sentiments

## Data Format
The dataset should contain at least two columns:
- `Text`: Social media text content
- `Sentiment`: Original sentiment labels (used for comparison)

## Key Functions
- `clean_text()`: Handles basic text cleaning and normalization
- `remove_punctuation()`: Removes punctuation characters
- `remove_stopwords()`: Eliminates common stopwords
- `lemmatizer()`: Reduces words to their base forms
- `remove_emojis()`: Removes emoji characters from text
- `remove_urls()`: Cleans URLs from text

## Results
The analysis provides:
- Cleaned and preprocessed text data
- Sentiment classification (Positive/Negative)
- Visual word cloud representations
- Processed dataset ready for further analysis or modeling

## Notes
- The project is optimized for English text but can be adapted for other languages
- VADER is particularly effective for social media text sentiment analysis
- Additional preprocessing steps can be added based on specific dataset requirements

## License
This project is open source and available under the [MIT License](LICENSE).
