import React, { useState } from "react";
import "./App.css";

function App() {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");
  const [originalImage, setOriginalImage] = useState("");
  const [maskImage, setMaskImage] = useState("");

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!file) {
      setMessage("Please select an image first.");
      return;
    }

    setMessage("Processing...");
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to fetch segmentation mask");
      }

      const data = await response.json();

      setMessage("Segmentation complete!");

      // Convert the original image to base64 for displaying it
      const reader = new FileReader();
      reader.onload = () => {
        setOriginalImage(reader.result);
      };
      reader.readAsDataURL(file);

      // Convert the mask array into an image
      const maskArray = data.mask;
      const width = maskArray[0].length;
      const height = maskArray.length;

      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      canvas.width = width;
      canvas.height = height;

      const imageData = ctx.createImageData(width, height);
      for (let i = 0; i < height; i++) {
        for (let j = 0; j < width; j++) {
          const index = (i * width + j) * 4;
          const value = maskArray[i][j] * 255; // Convert binary mask to grayscale
          imageData.data[index] = value;
          imageData.data[index + 1] = value;
          imageData.data[index + 2] = value;
          imageData.data[index + 3] = 255; // Alpha channel
        }
      }

      ctx.putImageData(imageData, 0, 0);
      setMaskImage(canvas.toDataURL());
    } catch (error) {
      setMessage(`Error: ${error.message}`);
    }
  };

  return (
    <div className="App">
      <h1>Green Cover Detection</h1>

      <input type="file" onChange={handleFileChange} accept="image/*" />
      <button onClick={handleSubmit}>Upload & Predict</button>

      <p>{message}</p>

      {originalImage && (
        <div>
          <h2>Original Image</h2>
          <img src={originalImage} alt="Original" style={{ width: "50%" }} />
        </div>
      )}

      {maskImage && (
        <div>
          <h2>Segmentation Mask</h2>
          <img src={maskImage} alt="Mask" style={{ width: "50%" }} />
        </div>
      )}
    </div>
  );
}

export default App;
