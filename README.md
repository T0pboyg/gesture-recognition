# Gesture Meme Recognition

A real-time image similarity application built with Python, OpenCV, MediaPipe and MobileNetV3.

## Features

- Real-time webcam capture
- Image embedding generation
- Cosine similarity comparison
- Best matching meme detection
- Adjustable similarity threshold
- Automatic display of matching images

## Technologies

- Python
- OpenCV
- MediaPipe Image Embedder
- MobileNetV3
- NumPy

## Installation

1. Clone the repository.
2. Create a Python virtual environment:

   py -m venv venv

3. Install dependencies:

   python -m pip install -r requirements.txt

4. Create the following directories:

   models/
   memes/

5. Download the MobileNetV3 Small Image Embedder model from:

   https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite

6. Save the model inside the models directory.

7. Add your own reference images to the memes directory.

## Usage

Run:

python main.py

Press Q to exit the application.

## How It Works

The application generates image embeddings from stored reference images and live webcam frames.

It calculates cosine similarity between embeddings and displays the closest matching image when the similarity exceeds the configured threshold.

The similarity threshold can be changed using MATCH_THRESHOLD in main.py.

## Notes

Reference images and model files are not included in the repository.

A webcam is required to run the application.
