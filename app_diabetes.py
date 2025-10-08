import joblib
import numpy as np  
# Importation des modules nécessaires de PyQt5 et du système
from PyQt5 import uic
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QIcon
import sys

# Activer le scaling pour les écrans haute résolution (High DPI)
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

# Création de l'instance de l'application PyQt
app = QApplication(sys.argv)

# Charger le fichier UI créé avec Qt Designer
window = uic.loadUi("E:\\Projet\\Diabetes\\diabetes.ui")

# Définir le titre de la fenêtre et l'icône de l'application
window.setWindowTitle("Prédiction du Diabète – Votre santé compte")
window.setWindowIcon(QIcon("E:\\Projet\\Diabetes\\diabete.png"))

# Charger le modèle pré-entraîné (Random Forest)
model = joblib.load("random_forest_model.pkl")

# Fonction de prédiction du diabète
def predction():
    """
    Récupère les valeurs saisies par l'utilisateur dans les QTextEdit,
    les convertit en float, les organise dans un tableau numpy,
    puis effectue la prédiction avec le modèle chargé.
    Affiche ensuite le résultat dans label_4 avec un style coloré.
    """
    # Récupération des entrées utilisateur
    pregnancies = float(window.inputPregnancies.toPlainText())
    glucose = float(window.inputGlucose.toPlainText())
    bp = float(window.inputBP.toPlainText())
    skin = float(window.inputSkin.toPlainText())
    insulin = float(window.inputInsulin.toPlainText())
    bmi = float(window.inputBMI.toPlainText())
    dpf = float(window.inputDPF.toPlainText())
    age = float(window.inputAge.toPlainText())

    # Création du tableau de données pour le modèle
    dataUser = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])

    # Prédiction avec le modèle
    pred = model.predict(dataUser)

    # Affichage du résultat avec style
    if pred == 1:
        window.label_4.setText("La personne est diabétique")
        window.label_4.setStyleSheet("color: orange; font-weight: bold;")  # style pour diabétique
    else: 
        window.label_4.setText("La personne n'est pas diabétique")
        window.label_4.setStyleSheet("color: green; font-weight: bold;")  # style pour non diabétique

# Fonction pour vider tous les champs et réinitialiser le label de résultat
def clear():
    """
    Efface toutes les entrées utilisateur et réinitialise le label de résultat
    avec un style par défaut.
    """
    window.inputPregnancies.clear()
    window.inputGlucose.clear()
    window.inputBP.clear()
    window.inputSkin.clear()
    window.inputInsulin.clear()
    window.inputBMI.clear()
    window.inputDPF.clear()
    window.inputAge.clear()

    window.label_4.setText("Le résultat est : ")
    window.label_4.setStyleSheet(
        "background-color: #50dfff;"
        "border-radius:3.5px; border: 2px solid #43bcd4;"
        "font: 12px Arial; height: 800px; width: 200px;"
    )

# Connexion des boutons aux fonctions correspondantes
window.pushButton.clicked.connect(predction)  # Bouton Prédire
window.pushButton_2.clicked.connect(clear)   # Bouton Réinitialiser

# Lancer l'application
if __name__ == '__main__':
    window.show()
    sys.exit(app.exec_())