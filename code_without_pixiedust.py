!pip install scikit-learn
!pip install brunel

!pip install ibm_watson_machine_learning

!pip install  matplotlib

import matplotlib.pyplot as plt

import sklearn

import pandas as pd

from sklearn.ensemble import RandomForestClassifier 

import numpy as np 

from sklearn.model_selection import train_test_split 

from scipy.io import arff 

import brunel

from ibm_watson_machine_learning import APIClient

url = 'https://raw.githubusercontent.com/HisashiOne/AnahuacDataScience/main/Denormalized%20claims%20data.csv'
raw_df = pd.read_csv(url)
display(raw_df)
fraud_counts = raw_df['FLAG_FOR_FRAUD_INV'].value_counts()
plt.figure(figsize=(8, 6))
fraud_counts.plot(kind='bar', color=['lightgreen', 'skyblue'])
plt.title('Conteo de Casos de Fraude (0 = No, 1 = Sí)', fontsize=14)
plt.xlabel('FLAG_FOR_FRAUD_INV', fontsize=12)
plt.ylabel('Número de Registros', fontsize=12)
for i, v in enumerate(fraud_counts):
    plt.text(i, v + 10, str(v), ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.show()

plt.tight_layout()
plt.show()
