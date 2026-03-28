import pandas as pd
def drop_unnecessary_columns(df):
    df = df.copy()
    df = df.drop(columns=['id','date_added'], errors='ignore')
    return df


GENRE_MAP = {
    'International Movies':     'International Movies',
    'International TV Shows':   'International TV Shows',
    'Dramas':                   'Dramas',
    'Comedies':                 'Comedies',
    'Action & Adventure':       'Action & Adventure',
    'Documentaries':            'Documentaries',
    'Children & Family Movies': 'Children & Family Movies',
    'Animation':                'Animation',
    'Thrillers':                'Thrillers',
    'Crime & Thriller TV':      'Crime & Thriller TV',
    

    'Action-Adventure':             'Action & Adventure',
    'TV Action & Adventure':        'Action & Adventure',
    'Sports Movies':                'Action & Adventure',
    'Sci-Fi & Fantasy':             'Action & Adventure',
    'Fantasy':                      'Action & Adventure',

    'Comedy':                        'Comedies',
    'Stand-Up Comedy':               'Comedies',
    'Stand-Up Comedy & Talk Shows':  'Comedies',
    'Reality TV':                    'Comedies',
    'Musical':                       'Comedies',

    'Docuseries':                    'Documentaries',
    'Science & Nature TV':           'Documentaries',
    'Animals & Nature':              'Documentaries',
    'Music & Musicals':              'Documentaries',

    'Romantic Movies':               'Dramas',
    'Coming of Age':                 'Dramas',
    'Independent Movies':            'Dramas',
    'LGBTQ Movies':                  'Dramas',
    'Drama':                         'Dramas',
    'TV Dramas':                     'Dramas',
    'Romantic TV Shows':             'Dramas',

    'Latin American Movies':         'International Movies',
    'Middle Eastern Movies':         'International Movies',

    'Anime Series':                  'Animation',
    'Anime Features':                'Animation',

    "Kids' TV":                      'Children & Family Movies',
    'Family':                        'Children & Family Movies',
    'Kids':                          'Children & Family Movies',

    'Korean TV Shows':               'International TV Shows',
    'British TV Shows':              'International TV Shows',
    'Spanish-Language TV Shows':     'International TV Shows',

    'Horror Movies':                 'Thrillers',
    'Crime TV Shows':                'Crime & Thriller TV',
    'TV Horror':                     'Crime & Thriller TV',
    'TV Mysteries':                  'Crime & Thriller TV',
    'TV Thrillers':                  'Crime & Thriller TV',   
}

def consolidate_genres(df, min_samples=150):
    df = df.copy()
    df['listed_in'] = df['listed_in'].apply(
        lambda x: next(
        (GENRE_MAP[g.strip()] 
        for g in str(x).split(',') 
        if g.strip() in GENRE_MAP),
        str(x).split(',')[0].strip()
        )
    )
    counts = df['listed_in'].value_counts()
    valid  = counts[counts >= min_samples].index
    df = df[df['listed_in'].isin(valid)].reset_index(drop=True)
    return df

import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [word for word in text.split() if word not in ENGLISH_STOP_WORDS ]
    return " ".join(words)

def combine_text_features(df):
    df = df.copy()
    df['combined_text'] = (
        df['type'].apply(clean_text).fillna('')    + ' ' +
        df['rating'].apply(clean_text).fillna('')  + ' ' +
        df['country'].apply(clean_text).fillna('') + ' ' +
        df['title'].apply(clean_text).fillna('') + ' ' +
        df['cast'].apply(clean_text).fillna('') + ' ' +
        df['director'].apply(clean_text).fillna('') + ' ' +
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
        max_features=60000,
        ngram_range=(1, 2),      
        stop_words='english',
        min_df=1,
        max_df=0.95,
        sublinear_tf=True        
    )
    X_text = tfidf.fit_transform(df_in['combined_text'])
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    X_cat = ohe.fit_transform(df_in[['type', 'rating','platform']].fillna('unknown'))
    X_combined = hstack([X_text, X_cat])
    return X_combined, tfidf, ohe






