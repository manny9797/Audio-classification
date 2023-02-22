import librosa
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def normalize(datas):
    data_silenced, n = librosa.effects.trim(data, top_db=30)
    data_normalized = np.pad(data_silenced, (0, max_length - len(data_silenced)), 'constant', constant_values=0)
    return data_normalized


df_development = pd.read_csv("C:/Users/emanu/OneDrive/Desktop/dsl_data/development.csv")
df_eval = pd.read_csv("C:/Users/emanu/OneDrive/Desktop/dsl_data/evaluation.csv")

metafeat = []
trainmeta = []
train_set = []  # TRAIN-SET
test_set = []  # TEST-SET
train_label = []
target_dBFS = -20
pca = PCA(n_components=60)  # PCA model to transform train-set and test-set
count = 0
max_length = 279552  # Max length of audio files without silence, resulted by cycling on all of them

# 1) TRANSFORMATION OF CATEGORICAL DATA IN 'Development.csv' AND 'evaluation.csv'

df_development['gender_factor'] = pd.factorize(df_development['gender'])[0]
df_development['fluency_factor'] = pd.factorize(df_development['Self-reported fluency level '])[0]
df_development['speaker_factor'] = pd.factorize(df_development['speakerId'])[0]
df_development['firstlanguage_factor'] = pd.factorize(df_development['First Language spoken'])[0]
df_development['currentlanguage_factor'] = pd.factorize(df_development['Current language used for work/school'])[0]
df_development['age_factor'] = pd.factorize(df_development.ageRange)[0]
df_development['gender_factor'] = pd.factorize(df_development['gender'])[0]
df_eval['gender_factor'] = pd.factorize(df_eval['gender'])[0]
df_eval['fluency_factor'] = pd.factorize(df_eval['Self-reported fluency level '])[0]
df_eval['speaker_factor'] = pd.factorize(df_eval['speakerId'])[0]
df_eval['firstlanguage_factor'] = pd.factorize(df_eval['First Language spoken'])[0]
df_eval['currentlanguage_factor'] = pd.factorize(df_eval['Current language used for work/school'])[0]
df_eval['age_factor'] = pd.factorize(df_eval['ageRange'])[0]
df_eval['gender_factor'] = pd.factorize(df_eval['gender'])[0]

# 2) EXTRACTION OF FEATURES AND DATA BY Development.csv AND Evaluation.csv

for i, row in df_development.iterrows():
    path = row['path']
    data, sr = librosa.load(f"C:/Users/emanu/OneDrive/Desktop/{path}")
    audio = normalize(data)
    f = [row['fluency_factor'], row['speaker_factor'], row['firstlanguage_factor'], row['currentlanguage_factor'],
         row['gender_factor'], row['age_factor']]
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    MFCC = np.ravel(mfccs, order='F')
    train_label.append(row['action'] + row['object'])
    train_set.append(np.array(MFCC))
    trainmeta.append(np.array(f))
    count = count + 1
    print(count)

count = 0

for i, row in df_eval.iterrows():
    path = row['path']
    data, sr = librosa.load(f"C:/Users/emanu/OneDrive/Desktop/{path}")
    audio = normalize(data)
    f = [row['fluency_factor'], row['speaker_factor'], row['firstlanguage_factor'], row['currentlanguage_factor'],
         row['gender_factor'], row['age_factor']]
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    MFCC = np.ravel(mfccs, order='F')
    test_set.append(np.array(MFCC))
    metafeat.append(np.array(f))
    count = count + 1
    print(count)

train_set = np.array(train_set)
trainmeta = pd.DataFrame(trainmeta)
test_set = np.array(test_set)
metaf = pd.DataFrame(metafeat)
min_max_scaler = MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(train_set)
np_scaled_test = min_max_scaler.transform(test_set)

trainset = pd.DataFrame(np_scaled)
test_set = pd.DataFrame(np_scaled_test)
trainset = pd.concat([trainset, trainmeta], axis=1)
trainset.to_csv("train_set.csv")
test_set = pd.concat([test_set, metaf], axis=1)
test_set.to_csv("test_set.csv")

# 3) PCA TRASFORMATION FOR BOTH THE SETS

np_scaled = pca.fit_transform(np_scaled)
np_scaled_test = pca.transform(np_scaled_test)
trainset = pd.DataFrame(np_scaled)

clf = RandomForestClassifier(n_estimators=1500)
clf.fit(np_scaled, np.array(train_label))
ypred = clf.predict(np_scaled_test)

ypred.to_csv(f"C:/Users/emanu/OneDrive/Desktop/submission.csv")