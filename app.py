import streamlit as st
import bert_module
import lstm_module
import pickle
from overview import add_data_and_model_sections

# Cache the resource loading functions
@st.cache_resource
def load_bert_model():
    bert_tokenizer, bert_model = bert_module.get_model()
    return bert_tokenizer, bert_model


@st.cache_resource
def load_lstm_model():
    lstm_model_path = 'Idiom_Classification_LSTM.h5'

    # Load the tokenizer
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)

    lstm_model = lstm_module.LSTMTextClassifier.load_model(lstm_model_path, tokenizer)
    return lstm_model, tokenizer

# Load models
bert_tokenizer, bert_model = load_bert_model()
lstm_model, lstm_tokenizer = load_lstm_model()

# Add title and introductory text
st.title("Idiom Recognition App")

user_input = st.text_area('**Enter Text to Analyze**')
button = st.button("Analyze")

if user_input and button:
    # Perform inference using BERT model
    bert_prediction = bert_model.predict_text(bert_tokenizer, user_input)

    # Perform inference using LSTM model
    lstm_prediction = lstm_model.perform_lstm_inference(user_input)

    # Print predictions
    st.write("BERT Prediction: ", "figurative" if bert_prediction else "literal")
    st.write("LSTM Prediction: ", lstm_prediction[0])

# Call the function to add data and model sections
add_data_and_model_sections()