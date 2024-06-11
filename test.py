import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
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

def plot_points(image_path, points, url):
    # Load the image for plotting
    image = cv2.imread(image_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Plot points
    x_coords, y_coords = zip(*points)
    plt.scatter(x_coords, y_coords, c='red', s=1)
    plt.savefig(url)
    # plt.show()
    
    
images_name = ["ff44bc73-b1a2-4353-8161-ca3a241036c1_0_first_floor.png",
               "c55ca3b3-c9d2-4b1f-abae-57dd821b4aab_0_first_floor.png",
               "e5bd35be-b4b4-4017-aa64-cbb0fb44781a_0_first_floor.png",
               "fa09776f-32b9-4bda-aebf-aca862e88e5e_1_second_floor.png",
               "b3dcd55e-27a1-46b3-b682-5c49a3ae09d2_0_first_floor.png",
               "f7783d18-5ac4-47f1-95ca-2a1b813bfa9f_1_second_floor.png"]  
  
for image_name in images_name:   
    output_url = './output/sam_fine_3/'
    base_url = '/Users/amin/Desktop/higharc/Datasets/unlabeled/data_pulte/pulte/floorplans/'
    grid_spacing = 20  # Adjust the spacing as needed
    image_path = os.path.join(base_url, image_name)
    # Generate and plot grid points
    points, res = generate_grid_points(os.path.join(base_url, image_name), grid_spacing)
    plot_points(os.path.join(base_url, image_name), points, os.path.join(output_url, "plain", image_name))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    model = SamModel.from_pretrained("./SAM-ZOO_3")
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
    mask = masks[0][0].detach().cpu().numpy().transpose(1, 2, 0) * 1.0
    img = np.array(raw_image)
    print(img.shape)

    # Plot the image and the mask
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.imshow(mask, alpha=0.5, cmap='jet')
    plt.axis('off')
    plt.savefig(os.path.join(output_url, image_name))
    # plt.show()