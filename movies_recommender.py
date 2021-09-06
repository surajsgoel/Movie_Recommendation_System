import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


##Step 1: Read CSV File

df = pd.read_csv("movies_dataset.csv")
print (df.columns)

##Step 2: Select Features

features = ['keywords','cast', 'genres', 'director']

##Step 3: Create a column in DF which combines all selected features

for feature in features:
	df[feature] = df[feature].fillna('')

def create_feature_string(row):
		return row["keywords"] + " " + row["cast"] + " " + row["genres"] + " " + row["director"]
	
df["combined_features"] = df.apply(create_feature_string, axis=1)


##Step 4: Create count matrix from this new combined column

cv = CountVectorizer()
cv_fit = cv.fit_transform(df["combined_features"])

##Step 5: Compute the Cosine Similarity based on the count_matrix

cosine_sim = cosine_similarity(cv_fit)

movie_user_likes = "Avatar"

## Step 6: Get index of this movie from its title

index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosine_sim[index]))


## Step 7: Get a list of similar movies in descending order of similarity score

similar_movies_sorted = sorted(similar_movies, key= lambda x:x[1],reverse=True)


## Step 8: Print titles of first 50 movies

count = 0

for movie in similar_movies_sorted:
	print(get_title_from_index(movie[0]))
	if count == 50:
		break

	count = count + 1