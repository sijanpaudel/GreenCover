import requests
import json
import matplotlib.pyplot as plt
import numpy as np

resp = requests.post("http://127.0.0.1:5000/predict", files={'file': open("saved.jpg", "rb")})

# Ensure request was successful
if resp.status_code == 200:
    response_json = resp.json()  # Parse JSON response
    if "mask" in response_json:
        mask = response_json["mask"]
        # print("Mask received successfully:", mask)  # Process the mask as needed
    else:
        print("Unexpected response:", response_json)
else:
    print("Error:", resp.status_code, resp.text)  # Print error details

mask = np.array(mask)

plt.imshow(mask*255, cmap="gray")
plt.show()