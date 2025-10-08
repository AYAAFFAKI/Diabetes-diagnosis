# -----------------------------
# Importation des bibliothèques nécessaires
# -----------------------------
import pandas as pd                      # Manipulation des données
import numpy as np                       # Calculs numériques
import matplotlib.pyplot as plt          # Visualisation des données
import seaborn as sns                    # Visualisation avancée
from sklearn.model_selection import train_test_split  # Séparer le dataset en train/test
from sklearn.ensemble import RandomForestClassifier  # Algorithme Random Forest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler   # Pour équilibrer les classes
import joblib                             # Sauvegarde du modèle
from collections import Counter           # Compter les occurrences
import warnings
warnings.filterwarnings('ignore')        # Ignorer les warnings

plt.style.use('fivethirtyeight')         # Style de graphique

# -----------------------------
# Chargement du dataset
# -----------------------------
data = pd.read_csv("C:\\Users\\pro\\.cache\\kagglehub\\datasets\\mathchi\\diabetes-data-set\\versions\\1\\diabetes.csv")

# Afficher les informations générales sur le dataset
data.info()
data.describe()
print("Nombre de doublons :", data.duplicated().sum())

# -----------------------------
# Analyse exploratoire
# -----------------------------
# Matrice de corrélation avec heatmap
sns.heatmap(data.corr(), annot=True, fmt='0.2f', linewidths=.5)

# Visualisation de la distribution de la colonne 'Pregnancies'
plt.figure(figsize=(20,10))
plt.subplot(1,3,1)
plt.title('Count Plot - Pregnancies')
sns.countplot(x='Pregnancies', data=data)

plt.subplot(1,3,2)
plt.title('Distribution Plot - Pregnancies')
sns.displot(data['Pregnancies'])

plt.subplot(1,3,3)
plt.title('Box Plot - Pregnancies')
sns.boxplot(y=data['Pregnancies'])
plt.show()

# Visualisation de la distribution des classes de sortie (Outcome)
sns.countplot(x='Outcome', data=data, palette=['g','r'])

# -----------------------------
# Préparation des données pour l'entraînement
# -----------------------------
x = data.drop('Outcome', axis=1)  # Variables explicatives
y = data['Outcome']               # Variable cible

# Équilibrage des classes avec over-sampling
rm = RandomOverSampler(random_state=41)
x_res, y_res = rm.fit_resample(x, y)
print('Ancienne répartition des classes :', Counter(y))
print('Nouvelle répartition après oversampling :', Counter(y_res))

# Séparation en données d'entraînement et de test
x_train, x_test, y_train, y_test = train_test_split(x_res, y_res, test_size=0.2, random_state=42)

# -----------------------------
# Entraînement du modèle
# -----------------------------
model = RandomForestClassifier(n_estimators=500, class_weight='balanced')  # Random Forest
model.fit(x_train, y_train)  # Entraînement

# -----------------------------
# Évaluation du modèle sur le jeu de test
# -----------------------------
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Matrice de confusion
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='g')

# Affichage des métriques
print(model)
print('Accuracy :', accuracy)
print('Recall :', recall)
print('F1-score :', f1)

# -----------------------------
# Évaluation sur un jeu de données externe
# -----------------------------
evaluation_data = pd.read_csv("diabetes_test_samples.csv")
x_eval = evaluation_data.drop('Outcome', axis=1)
y_eval = evaluation_data['Outcome']
y_pred_eval = model.predict(x_eval)

acc = accuracy_score(y_eval, y_pred_eval)
f1_eval = f1_score(y_eval, y_pred_eval)

print(f"Accuracy sur les nouvelles données : {acc*100:.2f}%")
print(f"F1-score sur les nouvelles données : {f1_eval*100:.2f}%")

# -----------------------------
# Sauvegarde du modèle pour usage futur
# -----------------------------
joblib.dump(model, "random_forest_model.pkl")
