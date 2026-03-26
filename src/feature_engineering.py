import pandas as pd
def drop_unnecessary_columns(df):
    df = df.copy()
    df = df.drop(columns=['id', 'title'], errors='ignore')
    return df


GENRE_MAP = {
    'Action-Adventure':             'Action & Adventure',
    'Comedy':                        'Comedies',
    'Stand-Up Comedy':               'Comedies',
    'Stand-Up Comedy & Talk Shows':  'Comedies',
    'Docuseries':                    'Documentaries',
    'Romantic Movies':               'Dramas',
    'Coming of Age':                 'Dramas',
    'Independent Movies':            'Dramas',
    'LGBTQ Movies':                  'Dramas',
    'Latin American Movies':         'International Movies',
    'Middle Eastern Movies':         'International Movies',
    'Anime Series':                  'Animation',
    'Anime Features':                'Animation',
    "Kids' TV":                      'Children & Family Movies',
    'Family':                        'Children & Family Movies',
    'Korean TV Shows':               'International TV Shows',
    'British TV Shows':              'International TV Shows',
    'Spanish-Language TV Shows':     'International TV Shows',
    'Horror Movies':                 'Thrillers',
    'Crime TV Shows':                'Crime & Thriller TV',
    'TV Horror':                     'Crime & Thriller TV',
    'Romantic TV Shows':             'TV Dramas',
    'Sports Movies':                 'Action & Adventure',
    'Sci-Fi & Fantasy':              'Action & Adventure',
    'Fantasy':                       'Action & Adventure',
    'Music & Musicals':              'Documentaries',
    'Reality TV':                    'TV Comedies',
}

def consolidate_genres(df, min_samples=100):
    df = df.copy()
    df['listed_in'] = df['listed_in'].apply(
        lambda x: GENRE_MAP.get(
            str(x).split(',')[0].strip(),
            str(x).split(',')[0].strip()
        )
    )
    counts = df['listed_in'].value_counts()
    valid  = counts[counts >= min_samples].index
    df = df[df['listed_in'].isin(valid)].reset_index(drop=True)
    return df

import re
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def combine_text_features(df):
    df = df.copy()
    df['combined_text'] = (
        df['type'].str.lower().fillna('')    + ' ' +
        df['rating'].str.lower().fillna('')  + ' ' +
        df['country'].str.lower().fillna('') + ' ' +
        df['clean_desc'] + ' ' +
        df['clean_desc']  
    )
    return df

def split_features_target(df):
    df = df.copy()
    X = df.drop('listed_in', axis=1)
    y = df['listed_in']
    return X, y

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
def apply_tfidf(df_in):
    tfidf = TfidfVectorizer(
        max_features=40000,
        ngram_range=(1, 3),      
        stop_words='english',
        min_df=2,
        max_df=0.90,
        sublinear_tf=True        
    )
    X_text = tfidf.fit_transform(df_in['combined_text'])
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_cat = ohe.fit_transform(df_in[['type', 'rating']].fillna('unknown'))
    X_combined = hstack([X_text, X_cat])
    return X_combined, tfidf, ohe

import nltk
from nltk.corpus import stopwords














stop_words = set(stopwords.words('english'))




