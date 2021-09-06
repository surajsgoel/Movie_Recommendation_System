from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

text = ["London Paris London", "Paris Paris London"]
cv = CountVectorizer()
cv_fit = cv.fit_transform(text)

similarity_scores = cosine_similarity(cv_fit) 

print (similarity_scores)

