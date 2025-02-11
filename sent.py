# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:51:23 2025

@author: JYOTSNA
"""

import ipywidgets as widgets
from IPython.display import display, clear_output
from queue import Queue
from threading import Thread
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import speech_recognition as sr  # Speech recognition library
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from textblob import TextBlob  # For sentiment analysis
import nltk
nltk.download('punkt')  # Download necessary data for TextBlob

# Create Queues to hold messages
messages = Queue()

# Function to record audio
def record(duration=5, fs=44100):
    clear_output(wait=True)  # Clear any previous output when starting recording
    print("Recording...")  # Display 'Recording...' when the button is clicked
    try:
        myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()  # Wait until recording is finished

        # Normalize the recording to 16-bit PCM
        myrecording = (myrecording * 32767).astype(np.int16)

        # Save the recording to a .wav file
        write("output.wav", fs, myrecording)  # Save as 'output.wav'
        clear_output(wait=True)  # Clear the "Recording..." message
        print("Audio saved as output.wav.")  # Display after recording completes

    except Exception as e:
        clear_output(wait=True)
        print(f"Error during recording: {e}")

# Function to recognize speech
def recognize_speech():
    recognizer = sr.Recognizer()
    clear_output(wait=True)  # Clear the output before speech recognition
    try:
        with sr.AudioFile("output.wav") as source:
            # Adjust the recognizer for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)  # Optional: duration in seconds to adapt

            # Read the entire audio file
            audio = recognizer.record(source)

            # Recognize speech using Google Web Speech API
            text = recognizer.recognize_google(audio)
            clear_output(wait=True)
            print(f"Recognized text: {text}")

            # Now that we have recognized text, classify and analyze it
            classify_speech(text)
            polarity = analyze_sentiment(text)  # Store polarity for additional use
            print(f"Polarity: {polarity}")

    except sr.UnknownValueError:
        clear_output(wait=True)
        print("Could not understand audio.")
    except sr.RequestError as e:
        clear_output(wait=True)
        print(f"Could not request results from Google Speech Recognition service; {e}")
    except Exception as e:
        clear_output(wait=True)
        print(f"Error during recognition: {e}")

# Machine Learning: Classification of speech with polarity feature
def classify_speech(text):
    # Sample data for training with polarity-based classification
    texts = [
        "Turn on the lights", "What is the time?", "How are you?",
        "Tell me a joke", "Play some music", "Is it raining?", 
        "Increase the volume", "What's your name?", "Who is the president?",
        "Stop the car", "Do you like pizza?"
    ]
    
    # Polarity labels: Positive, Neutral, Negative
    labels = [
        "command-positive", "question-neutral", "question-positive",
        "command-positive", "command-positive", "question-neutral",
        "command-neutral", "question-neutral", "question-neutral",
        "command-neutral", "question-positive"
    ]

    # Train a Naive Bayes model for speech classification
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.25, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Transform input text and predict the class
    input_text_vector = vectorizer.transform([text])
    prediction = model.predict(input_text_vector)
    print(f"Classified as: {prediction[0]}")

# Sentiment analysis using TextBlob (returns polarity)
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity

    # Print sentiment classification
    if sentiment > 0:
        print("Sentiment: Positive")
        return "Positive"
    elif sentiment == 0:
        print("Sentiment: Neutral")
        return "Neutral"
    else:
        print("Sentiment: Negative")
        return "Negative"

# Button click handlers
def on_record_button_clicked(b):
    # Start recording in a separate thread
    record_thread = Thread(target=record)
    record_thread.start()

# Button to recognize speech
def on_recognize_button_clicked(b):
    recognize_thread = Thread(target=recognize_speech)
    recognize_thread.start()

# Create buttons
record_button = widgets.Button(
    description='Record',
    button_style='success',
    tooltip='Start Recording',
    icon='microphone'
)

recognize_button = widgets.Button(
    description='Recognize Speech',
    button_style='info',
    tooltip='Recognize Speech from Audio',
    icon='comment'
)

# Attach button click handlers
record_button.on_click(on_record_button_clicked)
recognize_button.on_click(on_recognize_button_clicked)

# Display the buttons
display(record_button)
display(recognize_button)

# Output widget to display messages
output_area = widgets.Output()
display(output_area)