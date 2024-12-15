# Sarcasm Detection Using Deep Learning
## Overview
This project implements a sarcasm detection model using deep learning techniques. The model is trained on the Sarcasm Headlines Dataset to classify headlines as sarcastic or not. It achieves an accuracy of 86% on the test set.
The project uses GoogleNews-vectors-negative300.bin pre-trained Word2Vec embeddings for feature representation, a hybrid CNN-BiLSTM architecture, and grid search for hyperparameter tuning.
## Features
Utilizes Word2Vec embeddings for word representation.
Implements a hybrid CNN-BiLSTM model for text classification.
Includes Attention mechanism to improve context understanding.
Supports hyperparameter tuning to optimize model performance.
Achieves 86% accuracy on sarcasm detection.
Provides a user-friendly test script for real-time predictions.
## Dataset
https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection<br/>
The project uses the Sarcasm Headlines Dataset, which contains labeled headlines for sarcasm detection:<br/>
headline: The text of the headline.<br/>
is_sarcastic: Binary label indicating if the headline is sarcastic.<br/>
## Setup Instructions
Prerequisites<br/>
1. Python 3.7 or above
2. Libraries:<br/>
Keras<br/>
Pandas<br/>
Numpy<br/>
Scikit-learn<br/>
Gensim<br/>
## Installation
Clone the repository:<br/>
git clone https://github.com/bhawna1224/sarcasm-detection.git<br/>
cd sarcasm-detection<br/>
Install the required packages:<br/>
pip install -r requirements.txt<br/>
Download the Word2Vec embeddings:<br/>
https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?resourcekey=0-wjGZdNAUop6WykTtMip30g<br/>
Extract GoogleNews-vectors-negative300.bin.gz file in the project directory.<br/>
## Usage
1. Training the Model
Run the train.py script to train the sarcasm detection model:
python train.py
This will:
Load and preprocess the dataset.
Train the model with hyperparameter tuning.
Save the trained model as sarcasm_model_with_tuning.keras.
Save the tokenizer as tokenizer.pkl.
2. Testing the Model
Run the test.py script to test the sarcasm detection model with custom input:
python test.py
Enter a sentence when prompted, and the script will predict whether it is sarcastic or not. Example:
Enter a sentence to check if it's sarcastic: "Oh great, another Monday morning."
Sarcastic (Confidence: 0.85)
3. Hyperparameter Tuning
The train.py script includes a grid search function to optimize the model architecture:
Filters: [64, 128]
Filter Width: [3, 5]
Hidden Units: [64, 128]
Dropout Fraction: [0.3, 0.5]
Modify the tune_hyperparameters function for further experimentation.
## Key Components
1. Word Embeddings
Pre-trained GoogleNews-vectors-negative300.bin Word2Vec embeddings are used to initialize the embedding layer.
2. Model Architecture
The model combines:
CNN for feature extraction.
BiLSTM for sequential data processing.
Attention Mechanism for context emphasis.
Dense layers for classification.
3. Metrics
Loss Function: Binary Crossentropy
Optimizer: Adadelta
Metrics: Accuracy
## Future Improvements
Incorporate additional datasets for enhanced generalization.<br/>
Experiment with alternative pre-trained embeddings (e.g., GloVe, FastText).<br/>
Implement other advanced architectures like transformers or BERT.
