from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import pandas as pd

# 1. Предобработка данных (пример загрузки корпоративного датасета)
# Предположим, у нас есть CSV с колонками 'text' и 'label' (0 - негатив, 1 - позитив)
df = pd.read_csv("corp_reviews.csv")
dataset = load_dataset("pandas", data_files={"train": "corp_reviews.csv"})

# 2. Загрузка токенизатора и модели
model_name = "cointegrated/rubert-tiny2" # Легкая модель для русского языка
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. Настройка обучения
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
)

# 4. Обучение и сохранение
trainer.train()
model.save_pretrained("./corp_ai_model")
tokenizer.save_pretrained("./corp_ai_model")