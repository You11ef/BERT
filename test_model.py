import torch
from transformers import CamembertTokenizerFast, CamembertForTokenClassification
import numpy as np

# Charger le modèle et le tokenizer entraînés
model = CamembertForTokenClassification.from_pretrained('./trained_model')
tokenizer = CamembertTokenizerFast.from_pretrained('./trained_model')

# Charger les données de test
test_sentences = [
    "je recherche pour 2 jours un entrepreneur ou bien un photographe"
]

# Fonction pour faire des prédictions
def predict(sentences):
    model.eval()
    tokenized_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt", is_split_into_words=False)
    with torch.no_grad():
        outputs = model(**tokenized_inputs)
    predictions = torch.argmax(outputs.logits, dim=-1).numpy()
    return predictions, tokenized_inputs

# Faire des prédictions
predictions, tokenized_inputs = predict(test_sentences)

# Convertir les prédictions en étiquettes
label_map = {0: 'O', 1: 'B-METIER', 2: 'I-METIER'}
predicted_labels = [[label_map[label] for label in sentence] for sentence in predictions]

print(predictions)

# Afficher les résultats
for sentence, tokens, labels in zip(test_sentences, tokenized_inputs['input_ids'], predicted_labels):
    tokenized_sentence = tokenizer.convert_ids_to_tokens(tokens)
    print(f"Sentence: {sentence}")
    print("Tokens and Labels:")
    for token, label in zip(tokenized_sentence, labels):
        if token != tokenizer.pad_token:
            print(f"{token}: {label}")


