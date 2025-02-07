from flask import Flask, request, jsonify
import torch
from PIL import Image, UnidentifiedImageError
import io
import numpy as np
import segmentation_models_pytorch as smp
from torchvision import transforms
import imghdr
from flask_cors import CORS  # Import CORS


app = Flask(__name__)

CORS(app)

# Load the U-Net model
model = smp.Unet(
    encoder_name="resnet34",        # Choose encoder (e.g., resnet34, efficientnet-b0, etc.)
    encoder_weights="imagenet",     # Use pre-trained weights on ImageNet
    in_channels=3,                  # Number of input channels (e.g., 3 for RGB images)
    classes=1                       # Number of output classes
)

# Load the saved state dictionary
model.load_state_dict(torch.load("models/greenCover_detection.pth", map_location=torch.device("cpu")))
model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize(size=(1088, 1920)),  # Resize to the input size expected by the model
    transforms.ToTensor(),                # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

def preprocess_image(image_bytes):
    """Preprocess the image for the model."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")  # Ensure image is RGB
        image = transform(image).unsqueeze(0)  # Add batch dimension
        return image
    except UnidentifiedImageError:
        raise ValueError("Invalid image file. Unable to open.")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image upload, preprocessing, and prediction."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    # Ensure a valid file is selected
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Check if file is an actual image
    if imghdr.what(file) not in ["jpeg", "png", "jpg"]:
        return jsonify({"error": "Invalid image format. Only JPEG and PNG are supported."}), 400

    try:
        file.seek(0)  # Reset file pointer before reading
        image_bytes = file.read()

        # Ensure the file is not empty
        if len(image_bytes) == 0:
            return jsonify({"error": "Uploaded file is empty."}), 400

        # Preprocess and predict
        image = preprocess_image(image_bytes)

        with torch.no_grad():
            output = model(image)
            mask = (output.squeeze().numpy() > 0.5).astype(int)  # Threshold to create binary mask

            # Image.fromarray(mask).save("y.jpg")

        # Convert mask to a list for JSON serialization
        mask_list = mask.tolist()

        return jsonify({"mask": mask_list})

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
