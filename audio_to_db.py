import sqlite3
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Connect to the database
conn = sqlite3.connect('data.db')
c = conn.cursor()

# Load the CNN model for image recognition
model = load_model('image_model.h5')

# Take the image and audio from the user
image = cv2.imread('user_image.jpg')
audio = 'user_audio.mp3'

# Preprocess the image for input to the CNN model
image = cv2.resize(image, (224, 224)) # Resize the image to (224, 224)
image = np.expand_dims(image, axis=0) # Add a batch dimension
image = image.astype('float32') / 255.0 # Normalize the image
image = preprocess_input(image) # Preprocess the image for input to the CNN model

# Use the CNN model to extract the pattern from the image
pattern = model.predict(image)[0]

# Convert the pattern to a string representation for storing in the database
pattern_str = ','.join(str(p) for p in pattern)

# Store the pattern and audio file in the database
c.execute('INSERT INTO mappings (audio_file, pattern) VALUES (?, ?)', (audio, pattern_str))
conn.commit()

# Close the database connection
conn.close()
