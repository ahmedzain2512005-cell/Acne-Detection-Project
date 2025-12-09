# Acne-Detection-Project

## Overview

This project is a deep learning-based classification model trained to identify and categorize different types of skin conditions related to acne. The model uses YOLO (You Only Look Once) for object detection and classification, and it is deployed as a web application using Streamlit. It can classify images into 16 distinct classes:

- Acne
- Blackhead
- Conglobata
- Crystanlline
- Cystic
- Flat_wart
- Folliculitis
- Keloid
- Milium
- Papular
- Purulent
- Scars
- Sebo-crystan-conglo
- Syringoma
- Unlabeled
- Whitehead

The application allows users to upload skin images for analysis, providing fast, reliable, and easy-to-interpret results to assist dermatologists and individuals in making informed decisions about skin health.

**Disclaimer:** 
This tool is for informational purposes only and is not a substitute for professional medical advice.

## Features

- AI-powered dermatology assistant using advanced computer vision techniques.
- Supports classification of 16 types of acne and related skin conditions.
- User-friendly Streamlit interface for uploading and analyzing images.
- Displays prediction results with confidence scores and a progress bar.
- Includes a responsive UI with loading spinner and styled result cards.

## Requirements

To run this project, install the dependencies listed in `requirements.txt`. The project also requires the pre-trained model file `best.pt`, which should be placed in the project directory or specified with the full path in the code.

## Installation

1. Clone the repository:
   git clone https://github.com/your-username/acne-detection-model.git
   cd acne-detection-model
   
2. Install the required packages:
   pip install -r requirements.txt
   
3. Add the trained model file:
   Place best.pt inside the project folder (not included in ZIP due to size limit).

## Usage

To run the Streamlit interface:
   streamlit run gui.py
Then open the app in your browser (usually at http://localhost:8501). Upload an image, wait for analysis, and the model will display the predicted class and confidence.

## Project Structure

- `gui.py`: Main Streamlit application with enhanced UI (progress bar, spinner, styled cards).
- `predict.py`: Simpler version of the Streamlit application.
- `best.pt`: Pre-trained YOLO model (not included; download separately).
- `README.md`: Project Explanation (This file).
- `requirements.txt`: List of Python dependencies.
