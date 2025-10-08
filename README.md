Diagnostic du Diabète (Diabetes Diagnosis)

Ce projet vise à développer un modèle pour diagnostiquer le diabète en utilisant l'algorithme Random Forest et l'analyse de données avec Python et Jupyter Notebook.

📂 Contenu du projet

Le projet contient les fichiers suivants :

app_diabetes.py : Application Python pour utiliser le modèle.

diabetes.ipynb : Notebook Jupyter pour l'analyse des données et l'entraînement du modèle.

train_random_forest.py : Script pour entraîner le modèle Random Forest.

random_forest_model.pkl : Modèle entraîné (pickle).

diabetes_test_samples.csv : Jeu de données pour tester le modèle.

download_dataset.py : Script pour télécharger le dataset.

diabetes.ui : Interface utilisateur graphique (GUI).

diabete.png : Image illustrative.

.gitignore : Fichier pour ignorer certains fichiers ou dossiers dans Git.

⚙️ Prérequis

Avant d'exécuter le projet, installez les bibliothèques suivantes :

pip install pandas numpy scikit-learn matplotlib PyQt5

🚀 Comment exécuter le projet

Entraîner le modèle :

python train_random_forest.py


Lancer l'application :

python app_diabetes.py

📊 Analyse des données

Vous pouvez explorer les données et entraîner le modèle via le notebook Jupyter :

jupyter notebook diabetes.ipynb

🧪 Tester le modèle

Pour tester le modèle sur de nouvelles données :

python app_diabetes.py --test diabetes_test_samples.csv

🤝 Contributions Feel free to fork the repository and submit pull requests for improvements or fixes.

📞 Contact Created by Aya Affaki GitHub: AYAAFFAKI

Thank you for checking out this project!
