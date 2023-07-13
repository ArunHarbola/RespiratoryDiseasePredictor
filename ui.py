import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy 
import librosa 
#load the trained model to classify sign
from keras.models import load_model
model = load_model('predictorModel.h5')
max_pad_len = 862 # to make the length of all MFCC equal

def extract_features(file_name):
    """
    This function takes in the path for an audio file as a string, loads it, and returns the MFCC
    of the audio"""
   
    try:
        audio, sample_rate = librosa.load(file_name, duration=20) 
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = numpy.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
    except Exception as e:
        print("Error encountered while parsing file: ", file_name)
        return None 
    return mfccs


#dictionary to label all traffic signs class.
lbls = ['Bronchiectasis', 'Bronchiolitis' ,'COPD', 'Healthy' ,'Pneumonia' ,'URTI'] 
# Function to handle the "Browse" button click event
def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])
    if file_path:
        # Extract features from selected audio file
        input_data = extract_features(file_path)
        input_data = input_data.reshape(1, 40, 862, 1)
        
        # Make prediction using the model
        predictions = model.predict(input_data)
        predictions = predictions.reshape(6)
        
        # Get the label with the highest prediction
        max_element = lbls[numpy.argmax(predictions)]
        
        # Update the label in the UI
        result_label.config(text="Prediction: " + max_element)

# Create the main window
window = tk.Tk()
window.title("Audio Classification")
window.geometry("400x200")

# Create the UI components
browse_button = tk.Button(window, text="Browse", command=browse_file)
browse_button.pack(pady=20)

result_label = tk.Label(window, text="Prediction: ")
result_label.pack()

# Run the Tkinter event loop
window.mainloop()