import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor

def generate_grid_points(image_path, grid_spacing):
    # Load the image
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    
    # Generate grid points
    points = []
    res = []
    for y in range(0, height, grid_spacing):
        for x in range(0, width, grid_spacing):
            points.append((x, y))
            res.append([x, y])
    
    return points, res

def plot_points(image_path, points):
    # Load the image for plotting
    image = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Plot points
    x_coords, y_coords = zip(*points)
    plt.scatter(x_coords, y_coords, c='red', s=10)
    
    plt.show()
    
    
image_path = '/home/ubuntu/dataset/data_pulte/pulte/floorplans/fc959a9e-b508-4d72-a8dd-b2978e4d7205_1_second_floor.png'
grid_spacing = 50  # Adjust the spacing as needed

# Generate and plot grid points
points, res = generate_grid_points(image_path, grid_spacing)
plot_points(image_path, points)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

raw_image = Image.open(image_path).convert("RGB")
input_points = [res]  # 2D location of a window in the image

inputs = processor(raw_image, input_points=input_points, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)
scores = outputs.iou_scores

# Convert mask to numpy array
mask = masks[0][0].detach().cpu().numpy()

# Plot the image and the mask
plt.figure(figsize=(10, 10))
plt.imshow(raw_image)
plt.imshow(mask, alpha=0.5, cmap='jet')
plt.axis('off')
plt.show()