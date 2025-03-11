from sklearn import linear_model
from sklearn.metrics import accuracy_score

from predict_example import X_train, y_train, X_test, y_test

model = linear_model.SGDClassifier(max_iter=1000, random_state=42, loss="log_loss", penalty="l2", alpha=1e-3, eta0=1.0, learning_rate="constant")
model.fit(X_train, y_train)

print('Second model:')
print("Train accuracy = %.3f" % accuracy_score(y_train, model.predict(X_train)))
print("Test accuracy = %.3f" % accuracy_score(y_test, model.predict(X_test)))