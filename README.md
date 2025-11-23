# Reconnaissance et transcription vocale des chiffres et lettres

## Description

Ce projet permet de détecter du texte (lettres et chiffres) depuis une webcam en temps réel et de le lire à voix haute grâce à un système de transcription vocale.  
Il utilise :

- **OCR (Optical Character Recognition)** via **pytesseract** pour détecter le texte.  
- **TTS (Text-to-Speech)** via **pyttsx3** pour prononcer le texte détecté.  

Le programme peut détecter différents types de textes : blocs, lignes uniques ou texte dispersé, et adapte la lecture en conséquence.

Pour plus d'informations concernant la bibliothèque Tesseract : https://github.com/tesseract-ocr/tesseract?tab=readme-ov-file#installing-tesseract

---

## Installation

1. **Installer Python 3.x** si ce n’est pas déjà fait : [Python.org](https://www.python.org/downloads/)

2. **Installer Tesseract OCR (le moteur OCR)** :

- Télécharge la version utilisée dans ce projet : [Tesseract 5.5.0](https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe)  
- Pendant l’installation :  
  - Choisis **un nouveau dossier d’installation** (ex. `C:\Program Files\Tesseract-OCR`)  
  - **Ajoute la langue française** pour pouvoir détecter le texte en français  

> ⚠️ ATTENTION : Tesseract doit être installé soit dans le répertoire suggéré lors de l’installation, soit dans un nouveau répertoire. Le programme de désinstallation supprime tout le répertoire d’installation. Si vous installez Tesseract dans un répertoire existant, celui-ci sera supprimé avec tous ses sous-dossiers et fichiers.

3. **Installer les bibliothèques Python nécessaires** :

```python
pip install numpy opencv-python pyttsx3 pytesseract==5.5.0
```

4. **Configurer le chemin de Tesseract dans le code** :

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```
⚠️ Modifie ce chemin si tu as installé Tesseract ailleurs.

---

## Résultat

Voici un exemple de détection de texte avec notre programme OCR :

![Résultat](Resultat.png)

---

## Conseils pour améliorer la précision de l'OCR

- **Éclairage uniforme** : évite les zones d’ombre ou trop lumineuses sur le texte.  
- **Contraste élevé** : privilégie un texte sombre sur un fond clair pour faciliter la détection.  
- **Distance et taille du texte** : la caméra doit être suffisamment proche pour que les lettres et chiffres soient lisibles.  
- **Prétraitement d’image** : tu peux ajuster le filtrage et le seuillage adaptatif pour améliorer le contraste et réduire le bruit.  
- **Langue** : assure-toi d’avoir installé le pack de langue française dans Tesseract pour détecter correctement les accents et caractères spéciaux.  
- **Choix du mode PSM** : sélectionne le mode adapté à la disposition du texte :  
  - **PSM 6** : bloc de texte  
  - **PSM 7** : ligne unique  
  - **PSM 12** : texte dispersé (ex : panneaux, scènes, objets)
