from transformers import Trainer, TrainingArguments, CamembertForTokenClassification, CamembertTokenizerFast
import pandas as pd
from datasets import Dataset

# Charger le tokenizer et les données
tokenizer = CamembertTokenizerFast.from_pretrained('camembert-base')
tokenized_data = pd.read_csv('tokenized_data.csv')

import torch

# Vérifiez si un GPU est disponible
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Training on GPU")
else:
    device = torch.device('cpu')
    print("Training on CPU")



# Convertir en format Dataset
def preprocess_function(examples):
    return {
        'input_ids': eval(examples['input_ids']),
        'attention_mask': eval(examples['attention_mask']),
        'labels': eval(examples['labels'])
    }

dataset = Dataset.from_pandas(tokenized_data)
dataset = dataset.map(preprocess_function)

# Séparer en train/val/test
dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_val_dataset = dataset['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = train_val_dataset['train']
val_dataset = train_val_dataset['test']
test_dataset = dataset['test']

# Arguments de l'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=5e-5,  # Ajustez le taux d'apprentissage
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,  # Augmentez le nombre d'époques
    weight_decay=0.01,
)

# Charger le modèle
model = CamembertForTokenClassification.from_pretrained('camembert-base', num_labels=3)

# Déplacer le modèle vers le bon appareil
model.to(device)


# Initialiser le Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Entraîner le modèle
trainer.train()

# Sauvegarder le modèle entraîné
model.save_pretrained('./trained_model')
tokenizer.save_pretrained('./trained_model')
