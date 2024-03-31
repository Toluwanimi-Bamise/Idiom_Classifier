import re
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence, text


class LSTMTextClassifier(tf.keras.Model):
    def __init__(self, model_path, tokenizer):
        super(LSTMTextClassifier, self).__init__()
        self.model = tf.keras.models.load_model(model_path)
        self.tokenizer = tokenizer  # Use provided tokenizer

    def preprocess_input(self, sentence):
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-zA-Z0-9\s]', '', sentence, re.I|re.A)
        sentence = re.sub(' +', ' ', sentence)
        sentence = sentence.strip()
        return sentence

    def perform_lstm_inference(self, sentence):
        preprocessed_text = self.preprocess_input(sentence)
        sequences = self.tokenizer.texts_to_sequences([preprocessed_text])
        padded_sequences = sequence.pad_sequences(sequences, maxlen=90)
        predictions = self.model.predict(padded_sequences)
        decoded_predictions = ["figurative" if pred >= 0.5 else "literal" for pred in predictions]

        return decoded_predictions

    @staticmethod
    def load_model(model_path, tokenizer):
        return LSTMTextClassifier(model_path, tokenizer)  # Pass tokenizer to constructor

    @staticmethod
    def get_tokenizer():
        # Load tokenizer here (you need to adjust this based on how the tokenizer was saved during training)
        tokenizer = text.Tokenizer()
        return tokenizer