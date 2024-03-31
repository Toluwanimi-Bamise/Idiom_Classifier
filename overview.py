import streamlit as st


def add_data_and_model_sections():
    st.divider()
    # Add sections for data and model information
    st.subheader("About the Data: magpie dataset")
    st.write(" ")
    st.image('photos//word_cloud.png', caption='Idioms cloud of the magpie dataset')
    # Add dataset summary using markdown
    st.markdown("""
    #### **Dataset Summary**
    
    The MAGPIE corpus (Haagsma et al. 2020) is a large sense-annotated corpus of potentially idiomatic expressions (PIEs), based on the British National Corpus (BNC). Potentially idiomatic expressions are like idiomatic expressions, but the term also covers literal uses of idiomatic expressions, such as 'I leave work at the end of the day.' for the idiom 'at the end of the day'. This version of the dataset reflects the filtered subset used by Dankers et al. (2022) in their investigation on how PIEs are represented by NMT models. Authors use 37k samples annotated as fully figurative or literal, for 1482 idioms that contain nouns, numerals or adjectives that are colors (which they refer to as keywords). Because idioms show syntactic and morphological variability, the focus is mostly put on nouns. PIEs and their context are separated using the original corpus’s word-level annotations.
    
    #### **Languages**
    
    The language data in MAGPIE is in English (BCP-47 en)
    
    #### **Data Instances**
    
    The magpie configuration contains sentences with annotations for the presence, usage an type of potentially idiomatic expressions. An example from the train split of the magpie config (default) is provided below.
    
    ```json
    {
        'sentence': 'There seems to be a dearth of good small tools across the board.',
        'annotation': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        'idiom': 'across the board',
        'usage': 'figurative',
        'variant': 'identical',
        'pos_tags': ['ADV', 'VERB', 'PART', 'VERB', 'DET', 'NOUN', 'ADP', 'ADJ', 'ADJ', 'NOUN', 'ADP', 'DET', 'NOUN']
    } """)
    st.divider()
    ##############################################################
    st.subheader("About the Models")
    st.write("""
    
    **Overview**
    
    - Model Architectures: BERT, ELMO, LSTM
    - Task: Idiom classification (Figurative vs. Literal)
    
    **Input Format**
    
    The models expects text input in the form of a string containing an idiomatic expression.
    
    **Output Format**
    
    The models provides a binary prediction indicating whether the input idiomatic expression is figurative or literal.
    
    **Example Applications**
    
    - Identifying figurative language in text for sentiment analysis
    - Improving natural language understanding in conversational agents
    - Enhancing machine translation systems by preserving the intended meaning of idiomatic expressions
    
    """)

    # BERT-Idiom-Classifier
    st.subheader("1.BERT-Idiom-Classifier")
    st.image('photos//bert_acc.png', caption='BERT-Idiom-Classifier Accuracy')
    st.write("Model Accuracy: 90%")
    bert_limitations = """
    **Limitations**
    
    - Context Sensitivity: The classification accuracy may vary depending on the context in which the idiomatic expression appears.
    """
    st.write(bert_limitations)

    # ELMO-Idiom-Classifier
    st.subheader("2.ELMO-Idiom-Classifier")
    st.image('photos//elmo_acc.png', caption='ELMO-Idiom-Classifier Accuracy')
    st.write("Model Accuracy: 80%")
    elmo_limitations = """
    **Limitations**
    
    - Computational Complexity: ELMO embeddings require significant computational resources, which may limit real-time applications on resource-constrained devices.
    - Interpretability: The deep contextualized embeddings produced by ELMO may lack interpretability compared to traditional feature-based models.
    """
    st.write(elmo_limitations)

    # LSTM-Idiom-Classifier
    st.subheader("3.LSTM-Idiom-Classifier")
    st.image('photos//lstm_acc.png', caption='LSTM-Idiom-Classifier Accuracy')
    st.write("Model Accuracy: 80%")
    lstm_limitations = """
    **Limitations**
    
    - Sequential Processing: LSTM models may struggle with capturing long-range dependencies in text, which can affect the accuracy of classification for complex idiomatic expressions.
    """
    st.write(lstm_limitations)
