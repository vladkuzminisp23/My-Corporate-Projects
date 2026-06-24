from transformers import pipeline
from PIL import Image

# 1. Загрузка моделей через pipeline (это надежнее и проще)
# Используем стабильную открытую модель для русского языка
sentiment_analyzer = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")

# ИИ-ассистент (чат-бот)
chatbot = pipeline("text2text-generation", model="cointegrated/rut5-small")

# Распознавание фото
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

def analyze_sentiment(text: str) -> str:
    # pipeline сам обрабатывает токенизацию и возвращает готовый словарь
    result = sentiment_analyzer(text)[0]
    label = result['label'].upper()
    
    # Модель blanchefort выдает: POSITIVE, NEGATIVE, NEUTRAL
    if 'POSITIVE' in label:
        return "Позитив"
    elif 'NEGATIVE' in label:
        return "Негатив"
    else:
        return "Нейтрально"

def get_chatbot_response(user_text: str) -> str:
    # Промпт для ассистента организации
    prompt = f"Вопрос клиента: {user_text}. Сформулируй вежливый ответ от лица компании."
    response = chatbot(prompt, max_length=100)[0]['generated_text']
    return response

def recognize_image(image: Image.Image) -> list:
    results = image_classifier(image)
    return [{"label": r['label'], "score": r['score']} for r in results]