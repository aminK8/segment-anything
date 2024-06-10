import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image 
from patchify import patchify  #Only to handle large images
import random
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.patches import Rectangle


from transformers import SamProcessor
from transformers import SamModel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import AdamW

import monai

from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize


import cv2
import json


class SAMDataset(Dataset):
    """
    This class is used to create a dataset that serves input images and masks.
    It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
    """
    def __init__(self, images_base_url, masks_base_url, json_url, processor):
        self.images_base_url = images_base_url
        self.masks_base_url = masks_base_url
        self.processor = processor
        self.json_url = json_url
        with open(self.json_url) as file:
            data = json.load(file)
        self.categories = data['categories']
        self.images = data['images']
        self.annotations = data['annotations']

    def __len__(self):
        return len(self.annotations)
    
    @staticmethod
    def get_bounding_box(bbox):
        rand = np.random.randint(20, 120)
        res = [0 , 0 , 0 , 0]
        res[0] = int(max(0, bbox[0] - rand))
        res[1] = int(max(0, bbox[1] - rand))
        res[2] = int(bbox[2] + 2 * rand)
        res[3] = int(bbox[3] + 2 * rand)
        return res, rand
    
    staticmethod
    def get_largest_bbox(seg_list):
        all_points = []
        for seg in seg_list:
            segmentation_np = np.array(seg).reshape((-1, 2))
            all_points.append(segmentation_np)
        
        # Concatenate all points into a single array
        all_points = np.vstack(all_points)
        
        # Get the minimum and maximum x and y values
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        
        # Return the bounding box in the format [x_min, y_min, width, height]
        return [x_min, y_min, x_max - x_min, y_max - y_min]
    

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_file_name = ''
        for image_info in self.images:
            if image_info['id'] == annotation['image_id']:
                image_file_name = image_info['file_name']
                
        # print(annotation['id'])
        image = Image.open(os.path.join(self.images_base_url, image_file_name))
        mask = Image.open(os.path.join(self.masks_base_url, image_file_name))

        image = np.array(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        mask = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
        for seg in annotation["segmentation"]:
            segmentation_np = np.array(seg).reshape((-1, 2)).astype(np.int32)
            cv2.fillPoly(mask, [segmentation_np], 255)
        mask = np.array(mask)
        
        bbox = SAMDataset.get_largest_bbox(annotation["segmentation"])
        new_bbox, rand = SAMDataset.get_bounding_box(bbox)
        
        image = image[new_bbox[1]:new_bbox[1] + new_bbox[3], new_bbox[0]:new_bbox[0] + new_bbox[2]]
        mask = mask[new_bbox[1]:new_bbox[1] + new_bbox[3], new_bbox[0]:new_bbox[0] + new_bbox[2]]
        original_width = image.shape[0]
        original_height = image.shape[1]
        
        new_height = 256
        new_width = 256

        # Scaling factors
        scale_x = new_height / original_height
        scale_y = new_width / original_width
        
        bbox[0] = rand * scale_x  # Scale x-coordinates
        bbox[2] *= scale_x  # Scale x-coordinates
        bbox[1] = rand * scale_y  # Scale y-coordinates
        bbox[3] *= scale_y  # Scale y-coordinates
        # get bounding box prompt
        bbox = np.array(bbox)
        prompt = bbox

        
        # prepare image and prompt for the model
        image = cv2.resize(image, (new_width, new_height))
        mask = cv2.resize(mask, (new_width, new_height))
        mask = (mask > 127) * 1

        inputs = self.processor(image, input_boxes=[[[prompt]]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        # for k,v in inputs.items():
        #     print(k,v.shape)
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

        # add ground truth segmentation
        inputs["ground_truth_mask"] = mask

        return inputs, (image, mask, bbox)
    
    

processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
images_base_url = "/home/ubuntu/dataset/data_pulte/pulte/floorplans"
masks_base_url = "/home/ubuntu/dataset/data_pulte/pulte/panoptic_semseg_maskdino_augmented_floorplans"
json_url = "/home/ubuntu/dataset/data_pulte/pulte/floorplans/_annotation_pulte_maskdino_augmented_file.json"

train_dataset = SAMDataset(images_base_url=images_base_url, masks_base_url=masks_base_url, json_url=json_url, processor=processor)


train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True, drop_last=False)
batch, info = next(iter(train_dataloader))
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
image, mask, bbox = info
print(image.shape)
print(mask.shape)
print(bbox.shape)
image = image[0]
mask = mask[0]
bbox = bbox[0]
# Plot the first image
axes[0].imshow(image)
rect1 = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='r', facecolor='none')
axes[0].add_patch(rect1)
axes[0].set_title("Image 1 with Bounding Box")
axes[0].axis('off')  # Hide the axis

# Plot the second image
axes[1].imshow(mask)
rect2 = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='b', facecolor='none')
axes[1].add_patch(rect2)
axes[1].set_title("Image 2 with Bounding Box")
axes[1].axis('off')  # Hide the axis

plt.show()


for k,v in batch.items():
    print(k,v.shape)
    

model = SamModel.from_pretrained("facebook/sam-vit-base")
# model = SamModel.from_pretrained("./SAM-ZOO/sam_model_checkpoint.pth")

# make sure we only compute gradients for mask decoder
for name, param in model.named_parameters():
    print(name)
    if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        param.requires_grad_(False)


# Initialize the optimizer and the loss function
optimizer = AdamW(model.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
#Try DiceFocalLoss, FocalLoss, DiceCELoss
# seg_loss = monai.losses.DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
seg_loss = torch.nn.MSELoss()

#Training loop
num_epochs = 20

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


best_loss = 1000000
interation_count = 0
for epoch in range(num_epochs):
    epoch_losses = []
    for batch, _ in tqdm(train_dataloader):
        interation_count += 1
        # forward pass
        outputs = model(pixel_values=batch["pixel_values"].to(device),
                        input_boxes=batch["input_boxes"].to(device),
                        multimask_output=False)

        # compute loss
        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(device)
        loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        # backward pass (compute gradients of parameters w.r.t. loss)
        optimizer.zero_grad()
        loss.backward()

        # optimize
        optimizer.step()
        epoch_losses.append(loss.item())
        if interation_count % 500 == 499:
            print(f"{interation_count}: loss is {loss.item()}")
            lo = loss.item()


    print(f'EPOCH: {epoch}')
    print(f'Mean loss: {mean(epoch_losses)}')
    if lo < best_loss:
        print("saved model.")
        torch.save(model.state_dict(), "./SAM-ZOO/sam_model_checkpoint.pth")
        best_loss = lo
