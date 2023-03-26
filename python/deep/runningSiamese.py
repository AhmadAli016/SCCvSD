# import cv2
import torch
from torchvision import transforms
from PIL import Image
from siamese import BranchNetwork, SiameseNetwork

# Load image and convert to grayscale
image = Image.open('001_AB_real_D (batchSize-7  - test_batch_7 - 30 epochs).png').convert('L')
# image = cv2.imread("001_AB_real_D (batchSize-7  - test_batch_7 - 30 epochs).png")

# Resize and normalize image
transform = transforms.Compose([
    transforms.Resize((180, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
image = transform(image)

# Add batch dimension
image = image.unsqueeze(0)

# Create Siamese network
branch = BranchNetwork()
siamese_network = SiameseNetwork(branch)

# Compute feature vector
feature_vector = siamese_network.feature(image)

# Convert to numpy array
feature_vector = feature_vector.detach().numpy()

print(feature_vector)