import cv2
import torch
from torchvision.transforms import transforms
from siamese import BranchNetwork, SiameseNetwork

# Create an instance of the BranchNetwork class
branch = BranchNetwork()

# Load an image
img = cv2.imread("path/to/image.jpg")

# Resize and normalize the image
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((180, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])
img = transform(img)

# Add batch dimension to the image
img = img.unsqueeze(0)

# Pass the image through the BranchNetwork to get the feature vector
features = branch(img)

# Print the shape of the feature vector
print(features.shape)