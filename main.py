#%%
import numpy as np 
import pandas as pd


#%%
#movie data
data = pd.read_csv("./movies_metadata.csv")
data = data[['id', 'original_title', 'original_language']]
data = data.rename(columns={'id':'movieId'})
data = data[data['original_language']=='en']
data.head()

#%%
#ratings data
ratings = pd.read_csv("./ratings.csv")
ratings = ratings[['userId', 'movieId', 'rating']]
ratings = ratings.head(10000000000)

#%%
data.movieId = pd.to_numeric(data.movieId, errors='coerce')
ratings.movieId = pd.to_numeric(ratings.movieId, errors='coerce')


#%%
#single dataset by merging
singledata = pd.merge(ratings, data, on='movieId', how='inner')
singledata.head()

#%%
#movie matrix
matrix = singledata.pivot_table(index='userId', columns='original_title', values='rating')
matrix.head()

#%%
#pearson correlation - linear correlation
def pearsonR(s1, s2):
    s1_c = s1-s1.mean()
    s2_c = s2-s2.mean()
    return np.sum(s1_c*s2_c) / np.sqrt(np.sum(s1_c**2)*np.sum(s2_c**2))
#pxy = cov(X,Y)/varxvary
#N rec based on pearson
def recommend(movie, M, n):
    reviews=[]
    for title in M.columns:
        if title == movie:
            continue
        cor = pearsonR(M[movie], M[title])
        if np.isnan(cor):
            continue
        else:
            reviews.append((title, cor))
    
    reviews.sort(key=lambda tup: tup[1], reverse=True)
    return reviews[:n]
#%%
recs = recommend('Sleepless in Seattle', matrix, 100000000)
recs
#%%
