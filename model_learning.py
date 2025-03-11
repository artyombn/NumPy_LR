import pandas as pd
import numpy as np

from dmia.classifiers import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

train_df = pd.read_csv('./data/train.csv')

review_summaries = list(train_df['Reviews_Summary'].values)
review_summaries = [l.lower() for l in review_summaries]

vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(review_summaries)
y = train_df.Prediction.values
model = LogisticRegression()
model.train(X, y, verbose=True, learning_rate=1.0, num_iters=1000, batch_size=256, reg=1e-3)

pos_features = np.argsort(model.w)[-5:]
neg_features = np.argsort(model.w)[:5]

fnames = vectorizer.get_feature_names_out()
print([fnames[p] for p in pos_features])
print([fnames[n] for n in neg_features])



