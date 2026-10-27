"""Detección de emociones mediante el servicio de Skills Network."""

import requests


def emotion_detector(text_to_analyze):
    """Devuelve las puntuaciones y la emoción dominante de un texto."""
    emotion_names = ("anger", "disgust", "fear", "joy", "sadness")

    if not isinstance(text_to_analyze, str) or not text_to_analyze.strip():
        return dict.fromkeys((*emotion_names, "dominant_emotion"))

    url = (
        "https://sn-watson-emotion.labs.skills.network/"
        "v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    )

    headers = {
        "grpc-metadata-mm-model-id":
            "emotion_aggregated-workflow_lang_en_stock"
    }

    payload = {
        "raw_document": {
            "text": text_to_analyze.strip()
        }
    }

    response = requests.post(
        url,
        json=payload,
        headers=headers,
        timeout=20,
    )

    if response.status_code == 400:
        return dict.fromkeys((*emotion_names, "dominant_emotion"))

    response.raise_for_status()

    data = response.json()
    emotions = data["emotionPredictions"][0]["emotion"]

    result = {
        name: emotions[name]
        for name in emotion_names
    }

    result["dominant_emotion"] = max(result, key=result.get)

    return result
