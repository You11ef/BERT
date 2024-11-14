import pandas as pd
import random
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from transformers import CamembertTokenizerFast, CamembertForTokenClassification, Trainer, TrainingArguments
from datasets import Dataset

# Téléchargement des ressources NLTK nécessaires
nltk.download('punkt')

import torch

# Vérifiez si un GPU est disponible
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Training on GPU")
else:
    device = torch.device('cpu')
    print("Training on CPU")

# Définition des mots-clés et des templates pour les phrases
keywords = ["photographe", "videur", "chanteur", "plombier", "développeur", "cuisinier", "serveur", "DJ", "musicien", "développeur web", "plombier chauffagiste", "cuisinier gastronomique"]
cities = ["Paris", "Grenoble", "Lyon", "Marseille", "Toulouse"]
templates = [
    "recherche d'un {} pour 3 jours, 1000 euros à {}",
    "{} mariage demain à 100 euros",
    "j'ai besoin d'un {} à {}",
    "{} disponible à partir de demain à {}",
    "Je cherche un {} pour demain",
    "Nous avons besoin d'un {} pour la soirée",
    "Je voudrais un {} pour le week-end",
    "Cherche {} pour événement à {}",
    "Avez-vous un {} disponible pour demain à {}?",
    "Besoin urgent d'un {} pour ce soir à {}",
    "Nous organisons un événement et avons besoin d'un {} à {}",
    "Je souhaite pour 10 jours avoir un {}, 1000 euros à {}",
    "Urgent, besoin d'un {} à {}",
    "Offre d'emploi : {} à {}",
    "Recherche urgent {} à {} pour événement spécial",
    "Recherche {} pour un concert à {}"
]

non_keywords_templates = [
    "Je cherche un appartement à {}",
    "Urgent : besoin de réparation voiture à {}",
    "Où trouver des pizzas à {} ?",
    "J'ai besoin d'un serrurier à {}",
    "Comment se rendre à {} ?",
    "Où puis-je acheter des billets à {} ?",
    "Besoin d'une nounou à {}",
    "Où est le meilleur restaurant à {} ?",
    "Urgent : plomberie à {}",
    "Je veux réserver une table à {}",
    "Où se trouve la gare à {} ?"
]

# Génération de données pour CSV
data = []

# Générer des exemples avec et sans mots-clés
for _ in range(600):
    keyword = random.choice(keywords)
    template = random.choice(templates)
    city = random.choice(cities)
    sentence = template.format(keyword, city)
    data.append({'sentence': sentence, 'label': 1})

for _ in range(600):
    template = random.choice(non_keywords_templates)
    city = random.choice(cities)
    sentence = template.format(city)
    data.append({'sentence': sentence, 'label': 0})

df = pd.DataFrame(data)
df = df.sample(frac=1).reset_index(drop=True)  # Mélanger les données
df.to_csv('generated_data.csv', index=False)

# Annoter les phrases pour la reconnaissance d'entités
def annotate_sentence(sentence):
    tokens = word_tokenize(sentence)
    labels = ['O'] * len(tokens)
    for keyword in keywords:
        keyword_tokens = word_tokenize(keyword)
        for i in range(len(tokens) - len(keyword_tokens) + 1):
            if tokens[i:i + len(keyword_tokens)] == keyword_tokens:
                labels[i] = 'B-METIER'
                for j in range(i + 1, i + len(keyword_tokens)):
                    labels[j] = 'I-METIER'
    return {'tokens': tokens, 'labels': labels}

annotated_data = df['sentence'].apply(annotate_sentence)
annotated_df = pd.DataFrame(list(annotated_data))
annotated_df.to_csv('annotated_data.csv', index=False)

# Convertir les labels textuels en indices numériques
label_map = {'O': 0, 'B-METIER': 1, 'I-METIER': 2}

def tokenize_and_align_labels(tokens, text_labels):
    tokenized_inputs = tokenizer(tokens, is_split_into_words=True, truncation=True, padding="max_length", max_length=512)
    word_ids = tokenized_inputs.word_ids()
    label_ids = []
    previous_word_idx = None
    
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(label_map[text_labels[word_idx]])
        else:
            label_ids.append(label_map[text_labels[word_idx]] if word_idx < len(text_labels) else -100)
        previous_word_idx = word_idx
    
    tokenized_inputs['labels'] = label_ids
    return tokenized_inputs

tokenizer = CamembertTokenizerFast.from_pretrained('camembert-base')

# Appliquer la tokenisation et l'alignement des labels
tokenized_data = [tokenize_and_align_labels(row['tokens'], row['labels']) for index, row in annotated_df.iterrows()]
tokenized_df = pd.DataFrame(tokenized_data)
tokenized_df.to_csv('tokenized_data.csv', index=False)

