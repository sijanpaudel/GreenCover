# GreenCover Detection App

This project is a web application that uses deep learning to detect green cover (forested and unforested areas) in satellite images. The backend is powered by a U-Net model trained with segmentation techniques, and the frontend is built using React.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Backend Setup](#backend-setup)
- [Frontend Setup](#frontend-setup)
- [Usage](#usage)
- [License](#license)

## Features

- Upload a satellite image to detect forested vs. unforested areas.
- U-Net deep learning model for segmentation based on RGB images.
- Frontend built using React, allowing users to upload images and receive a green cover mask.

## Technologies Used

- **Backend**: 
  - Flask
  - PyTorch
  - Segmentation Models PyTorch (smp)
  - Flask-CORS for handling CORS issues

- **Frontend**:
  - React.js
  - Fetch API for making POST requests to the backend

## Backend Setup

1. **Clone the repository**:

   ```bash
   git clone https://github.com/sijanpaudel/GreenCover.git
   cd green-cover-detection

2. **Install dependencies**:

    - Make sure you have Python 3.7+ installed.
    - Install the required Python packages using pip:

    ```bash
    pip install -r requirements.txt

3. **Set up the U-Net model**:

    - Download the pre-trained U-Net model (if you don't already have it) and save it to the models/ directory.
    - The model should be named greenCover_detection.pth.

4. **Run the Flask backend**:
    
    ```bash
    python app.py

    ```

    The backend will start running at http://127.0.0.1:5000.

5. Enable CORS:

    - The backend uses flask-cors to handle cross-origin requests from the React frontend. Ensure this is set up correctly to avoid CORS issues when making requests from React.


## Frontend Setup

1. **Install Node.js**:

   - Make sure you have Node.js installed.
2. **Navigate to the frontend directory**:
   ```bash
    cd frontend/green-cover-detection/

3. **Install dependencies**:
    ```bash
    npm install

4. **Run the React app**:
    ```bash
    npm start
   
   ```
   The frontend will be available at http://localhost:3000.

## Usage
1. **Frontend**:

      - Open the React application in your browser (http://localhost:3000).
      - Upload a satellite image (RGB format) using the upload form.
      - The backend will process the image and return the mask for the green cover areas.
      - The mask will be displayed on the frontend.


2. **Backend**:

     - The backend listens for POST requests on /predict and expects an image file in the request.
     - The model processes the image and returns a binary mask indicating forested and unforested areas.
