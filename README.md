# NextWordLSTM

## Overview

NextWordLSTM is an NLP-based next-word prediction application built using **TensorFlow** and **LSTM (Long Short-Term Memory)** networks. The model learns sequential language patterns from text data and predicts the most probable next word for a given input sequence. The application is deployed using **Streamlit** and is being extended to a full end-to-end ML pipeline.

## Features

* LSTM-based deep learning model for next-word prediction
* Text preprocessing with tokenization and sequence padding
* Trained on literary text to capture contextual dependencies
* Interactive web interface built with Streamlit
* Scalable design for end-to-end deployment

## Tech Stack

* **Programming Language:** Python
* **Deep Learning:** TensorFlow, Keras
* **NLP:** Tokenization, Sequence Modeling
* **Model:** LSTM (Recurrent Neural Network)
* **Web App:** Streamlit

## Project Workflow

1. Data collection and text preprocessing
2. Tokenization and sequence generation
3. LSTM model training and evaluation
4. Model serialization
5. Streamlit-based deployment
6. End-to-end pipeline integration (in progress)

## Deployment

The application is deployed using **Streamlit**, providing an interactive interface where users can input text and receive next-word predictions in real time.

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Future Enhancements

* End-to-end ML pipeline integration
* Model performance optimization
* Support for larger and diverse datasets
* Cloud deployment and CI/CD integration

## Author

Swapnaneel Chatterjee

---

This project demonstrates practical application of LSTM-based sequence modeling and NLP techniques for real-world text prediction systems.
