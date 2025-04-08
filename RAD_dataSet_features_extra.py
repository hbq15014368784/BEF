import torch
import torchvision
from torchvision import transforms
from PIL import Image
import os
from torchvision.ops import roi_align
import torch.nn as nn

# 1. Load Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 2. Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match the input size expected by the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Function to extract features using Faster R-CNN
def extract_fasterrcnn_features(image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Extract features using Faster R-CNN
    with torch.no_grad():
        # Get the feature maps from the backbone
        features = model.backbone(image)
        # Get the full model output to get the RoIs (Region of Interests)
        output = model(image)

    # Extract RoIs (Region of Interests) from the output
    rois = output[0]['boxes']  # These are the RoI boxes in (x1, y1, x2, y2) format

    if rois.shape[0] == 0:
        print(f"No RoIs detected for {image_path}, skipping.")
        return None  # Skip this image

    # Normalize RoI coordinates to match the feature map size
    rois = rois / 224 * list(features.values())[0].shape[2]  # Adjust RoI to the feature map resolution

    # Add batch index to the RoIs (needed for roi_align)
    batch_indices = torch.zeros((rois.shape[0], 1))  # All RoIs belong to batch 0
    rois_with_batch = torch.cat([batch_indices, rois], dim=1)

    # Apply RoI Align to extract features for each RoI
    feature_map = list(features.values())[0]  # Select the first feature map from the backbone
    aligned_features = roi_align(feature_map, rois_with_batch, output_size=(7, 7))  # Pool to 7x7 size

    # Pass aligned features through the RoI box_head to get 1024-d features
    aligned_features = aligned_features.view(aligned_features.size(0), -1)  # Flatten the feature maps
    features = model.roi_heads.box_head(aligned_features)  # Pass through the box head

    # Ensure the result is shaped as [36, 1024]
    if features.shape[0] > 36:
        features = features[:36, :]  # Select only the first 36 regions if there are more
    elif features.shape[0] < 36:
        padding = torch.zeros((36 - features.shape[0], features.shape[1]))
        features = torch.cat([features, padding], dim=0)  # Pad with zeros if less than 36

    return features

# 4. Main function to process a folder of images and save features
def save_features_for_images(image_folder, output_folder):
    linear_layer = nn.Linear(1024, 2048)  # Define the linear layer to upsample features

    # Get all image files in the specified folder
    image_ids = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_id in image_ids:
        image_path = os.path.join(image_folder, image_id)
        features = extract_fasterrcnn_features(image_path)

        if features is None:  # Skip if no features were extracted
            print(f"Skipping {image_id} due to no detected features.")
            continue


        # Upsample features to 2048 dimensions
        features = linear_layer(features)

        # Save features to .pth file
        output_path = os.path.join(output_folder, f"{os.path.splitext(image_id)[0]}.pth")
        torch.save(features, output_path)
        print(f"Saved features for {image_id} to {output_path}")

# 5. Usage
if __name__ == "__main__":
    image_folder = 'data_RAD/images/'  # Folder containing the images
    output_folder = 'data_RAD/rcnn_features/'  # Folder to save features

    os.makedirs(output_folder, exist_ok=True)  # Create output directory if it doesn't exist
    save_features_for_images(image_folder, output_folder)
