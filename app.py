import numpy as np 
import pickle
import streamlit as st
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

#load the LSTM model
model = load_model('next_word_lstm.h5')


#Load the tokenizer
with open('tokenizer.pickle' , 'rb') as handle:
    tokenizer = pickle.load(handle)


#load the function

def predict_next_word_model(model,tokenizer, text , max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if(len(token_list))>max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]

    token_list = pad_sequences([token_list] , maxlen=max_sequence_len-1 , padding='pre')
    predicted = model.predict(token_list , verbose=0)
    predicted_word_index = np.argmax(predicted , axis=1)
    for word , index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None


#Streamlit app
st.title("Next Word Prediction with LSTM RNN")
input_text = st.text_input("Enter the sequence of word", "Make you a wholsome")
if st.button("Predict the next word"):
    max_sequence_len = model.input_shape[1]+1
    next_word = predict_next_word_model(model , tokenizer , input_text , max_sequence_len)
    st.write(f'Next word prediction: {next_word}')

