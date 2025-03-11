import pandas as pd
import tqdm
import logging

import matplotlib.pyplot as plt

from dmia.classifiers import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

logging.basicConfig(format="---------------------\n%(message)s",
                    level=logging.INFO
                    )
logger = logging.getLogger(__name__)


train_df = pd.read_csv('./data/train.csv')
# logger.info(train_df.head())

review_summaries = list(train_df['Reviews_Summary'].values)
review_summaries = [l.lower() for l in review_summaries]
# logger.info(review_summaries[:5])

vectorizer = TfidfVectorizer()
tfidfed = vectorizer.fit_transform(review_summaries)

X = tfidfed
y = train_df.Prediction.values
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

model = LogisticRegression()
model.train(X_train, y_train, learning_rate=0.01, num_iters=1000)

prediction = model.predict(X_train)
logger.info(f"first 10: {prediction[:10]},\n"
            f"last 10: {prediction[10:]}")

train_scores = []
test_scores = []
num_iters = 1000

for i in tqdm.trange(num_iters):
    # Сделайте один шаг градиентного спуска с помощью num_iters=1
    model.train(X_train, y_train, learning_rate=1.0, num_iters=1, batch_size=256, reg=1e-3)
    train_scores.append(accuracy_score(y_train, model.predict(X_train)))
    test_scores.append(accuracy_score(y_test, model.predict(X_test)))

logger.info('First model:')
logger.info(f"Train accuracy = {train_scores[-1]:.3f}")
logger.info(f"Test accuracy = {test_scores[-1]:.3f}")

if __name__ == "__main__":
    plt.figure(figsize=(10, 8))
    plt.plot(train_scores, 'r', test_scores, 'b')
    plt.show()