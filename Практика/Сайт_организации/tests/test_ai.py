import pytest
from app.ml_models import analyze_sentiment, get_chatbot_response
from app.utils import preprocess_text

def test_preprocessing():
    assert preprocess_text("Привет, Мир!!!") == "привет мир"

def test_sentiment_positive():
    result = analyze_sentiment("Отличный сервис, мне все понравилось!")
    assert result == "Позитив"

def test_sentiment_negative():
    result = analyze_sentiment("Ужасное обслуживание, долго ждал.")
    assert result == "Негатив"

def test_chatbot_response():
    response = get_chatbot_response("Как вернуть товар?")
    assert isinstance(response, str)
    assert len(response) > 0