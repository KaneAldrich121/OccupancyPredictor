import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

df = pd.read_csv('data.csv')

# TRAINING OPTIONS
# train = pd.DataFrame({'occupancy': df["Occupancy"], 'co2': df['co2']})
# train = pd.DataFrame({'occupancy': df["Occupancy"], 'voc': df['VOC']})
# train = pd.DataFrame({'occupancy': df["Occupancy"], 'pm25': df['pm25']})
# train = pd.DataFrame({'occupancy': df["Occupancy"], 'humidity': df['humidity']})
# train = pd.DataFrame({'occupancy': df["Occupancy"], 'co2': df['co2'], 'voc': df['VOC']})
# train = pd.DataFrame({'occupancy': df["Occupancy"], 'co2': df['co2'], 'voc': df['VOC'], 'pm25': df['pm25']})
train = pd.DataFrame({'occupancy': df["Occupancy"], 'co2': df['co2'], 'pm25': df['pm25'], 'humidity': df['humidity'], 'voc': df['VOC']})
test = pd.DataFrame()
y = train['occupancy']

X = train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

estimatorRuns = 0
estimatorCount = 1100
accuracies = []

while estimatorRuns < 100:
    clf = RandomForestClassifier(n_estimators=estimatorCount, max_depth=6)
    clf.fit(X_train, y_train)

    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=6, max_features='auto', max_leaf_nodes=None,
                           min_samples_leaf=1,
                           min_samples_split=2, min_weight_fraction_leaf=0.0,
                           n_estimators=estimatorCount, n_jobs=1, oob_score=False, random_state=None,
                           verbose=0, warm_start=False)

    clf.predict(X_test)
    # Make prediction and check model's accuracy
    prediction = clf.predict(X_test)
    acc = accuracy_score(np.array(y_test), prediction)
    accuracies.append(acc)
    estimatorRuns += 1

totalAccuracy = 0
for accuracy in accuracies:
    totalAccuracy += accuracy
averageAccuracy = totalAccuracy / len(accuracies)
print('Average Accuracy: ', averageAccuracy)
print('This is how many estimators: ', estimatorCount)
