from fastapi import FastAPI, File, UploadFile, HTTPException
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import io
import pytesseract

app = FastAPI()

# Load pre-trained Faster R-CNN model (ResNet-50 Backbone)
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Set the model to evaluation mode

# Define image transformation for Faster R-CNN
transform = transforms.Compose([transforms.ToTensor()])

@app.post("/detect_objects/")
async def detect_objects(file: UploadFile = File(...)):
    # Check if the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    # Read the image file
    content = await file.read()
    image = Image.open(io.BytesIO(content))

    # Transform image for Faster R-CNN (convert to tensor)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference using the Faster R-CNN model
    with torch.no_grad():  # Disable gradient calculation during inference
        predictions = model(image_tensor)

    # Filter detected objects based on confidence score > 0.5
    detected_objects = []
    for element in range(len(predictions[0]['labels'])):
        if predictions[0]['scores'][element] > 0.5:
            label = predictions[0]['labels'][element].item()
            score = predictions[0]['scores'][element].item()
            detected_objects.append({"label": label, "confidence": score})

    return {"detected_objects": detected_objects}

@app.post("/detect_text/")
async def detect_text(file: UploadFile = File(...)):
    # Check if the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")

    # Read the image file
    content = await file.read()
    image = Image.open(io.BytesIO(content))
    gray_image = np.array(image.convert("L"))

    # Perform text recognition using Tesseract (can also use other OCR libraries)
    
    text = pytesseract.image_to_string(gray_image)
    #os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR"

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


    return {"detected_text": text}
