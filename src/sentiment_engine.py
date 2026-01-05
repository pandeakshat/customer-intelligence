import pandas as pd
import numpy as np
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

class SentimentAnalyzer:
    """
    NLP Engine for Customer Reviews.
    """
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.lda_model = None
        self.vectorizer = None
        
    def analyze_sentiment(self, df: pd.DataFrame, text_col: str):
        """
        Adds 'Sentiment_Score' and 'Sentiment_Label' to the dataframe.
        """
        if text_col not in df.columns:
            return df
            
        # Clean text helper
        def clean_text(text):
            if pd.isna(text): return ""
            text = str(text).lower()
            text = re.sub(r'[^a-z\s]', '', text) # Remove punctuation
            return text

        # Apply VADER
        scores = []
        labels = []
        clean_texts = []
        
        for text in df[text_col]:
            # Handle NaN/Empty
            if pd.isna(text) or str(text).strip() == "":
                scores.append(0)
                labels.append('Neutral')
                clean_texts.append("")
                continue

            # Clean for topic modeling later
            clean_texts.append(clean_text(text))
            
            # Score (VADER handles punctuation well, so we use raw text for scoring)
            score = self.analyzer.polarity_scores(str(text))['compound']
            scores.append(score)
            
            if score >= 0.05:
                labels.append('Positive')
            elif score <= -0.05:
                labels.append('Negative')
            else:
                labels.append('Neutral')
                
        result_df = df.copy()
        result_df['Sentiment_Score'] = scores
        result_df['Sentiment_Label'] = labels
        result_df['Clean_Text'] = clean_texts
        
        return result_df

    def extract_topics(self, df: pd.DataFrame, text_col='Clean_Text', n_topics=5):
        """
        Runs LDA to find top themes in the text.
        """
        if text_col not in df.columns:
            return {}

        # Vectorize (Turn text into numbers)
        # Drop empty strings to avoid errors
        valid_text = df[text_col].dropna()
        valid_text = valid_text[valid_text.str.len() > 2] # Must be > 2 chars
        
        if valid_text.empty:
            return {}

        self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        dtm = self.vectorizer.fit_transform(valid_text)
        
        # Train LDA
        self.lda_model = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        self.lda_model.fit(dtm)
        
        # Extract Keywords for each topic
        topics = {}
        feature_names = self.vectorizer.get_feature_names_out()
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            # Get top 10 words for this topic
            top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
            topics[f"Topic {topic_idx+1}"] = top_words
            
        return topics

    def get_topic_distribution(self, df: pd.DataFrame, text_col='Clean_Text'):
        """
        Assigns the dominant topic to each review.
        """
        if not self.lda_model or text_col not in df.columns:
            return df
            
        valid_mask = (df[text_col].notna()) & (df[text_col].str.len() > 2)
        valid_text = df.loc[valid_mask, text_col]
        
        if valid_text.empty:
            return df

        dtm = self.vectorizer.transform(valid_text)
        topic_results = self.lda_model.transform(dtm)
        
        # Get index of max probability
        dominant_topics = topic_results.argmax(axis=1)
        
        # Align with original index
        df_out = df.copy()
        
        # Initialize columns
        df_out['Topic_ID'] = -1
        df_out['Topic_Label'] = "Unknown"
        
        # Assign only to valid rows
        df_out.loc[valid_mask, 'Topic_ID'] = dominant_topics
        df_out.loc[valid_mask, 'Topic_Label'] = [f"Topic {t+1}" for t in dominant_topics]
        
        return df_out