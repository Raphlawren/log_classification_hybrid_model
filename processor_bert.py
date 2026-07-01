import os
import joblib
import sklearn
from sentence_transformers import SentenceTransformer
from torch import embedding

# _model = None
# _embedding_model = None


# def _load_models():
#     global _model, _embedding_model
#     if _model is None:
#         model_path = os.path.abspath(
#             os.path.join(os.path.dirname(__file__), '..', 'models', 'log_classifier.joblib')
#         )
#         _model = joblib.load(model_path)
#     if _embedding_model is None:
#         _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
#     return _model, _embedding_model

def _load_models():
    # Load the sentence transformer model to compute log_message embeddings
    transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
    # Load the saved classification model
    # classification_logs/models/log_classifier.joblib
    # classifier_model = joblib.load('models/log_classifier.joblib')
    # classifier_model = joblib.load(os.path.join(os.path.dirname(__file__), 'models', 'log_classifier.joblib'))
    classifier_model = joblib.load('models/log_classifier.joblib')

    return classifier_model, transformer_model


def classify_with_bert(log_message):
    if log_message is None:
        raise ValueError('log_message must be a non-empty string')
    if not isinstance(log_message, str):
        log_message = str(log_message)
    log_message = log_message.strip()
    if not log_message:
        raise ValueError('log_message must be a non-empty string')

    model, embedding_model = _load_models()
    message_embedding = embedding_model.encode([log_message])
    probabilities = model.predict_proba(message_embedding)[0]
    if max(probabilities) < 0.5:
        return "Unclassified"
    
    predicted_label = model.predict(message_embedding)[0]
    
    return predicted_label



if __name__ == "__main__":
    logs =[
        "Ei, Eu quero trabalho",
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Hey bro, chill ya!",
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer"
    ]
    for log in logs:
        label = classify_with_bert(log)
        print(f"Log: {log} -> Predicted Label: {label}")