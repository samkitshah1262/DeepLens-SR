import numpy as np 
import torch 
import matplotlib.pyplot as plt

set_images_no = []
for i in range(10):
    set_images_no.append(np.load(f"ml4sci/sr/dataset3b/LR/LR_{i+1}.npy"))
set_images_sphere = []
for i in range(10):
    set_images_sphere.append(np.load(f"ml4sci/sr/dataset3b/HR/HR_{i+1}.npy"))


img = np.load("ml4sci/sr/dataset3b/LR/LR_1.npy")
img = torch.tensor(img, dtype=torch.float32)
print(img.shape)
img = np.load("ml4sci/sr/dataset3b/HR/HR_1.npy")
img = torch.tensor(img, dtype=torch.float32)
print(img.shape)
fig, axs = plt.subplots(1, 10, figsize=(20, 5))
for i in range(10):
    axs[i].imshow(set_images_no[i][0], cmap='gray')
    axs[i].axis('off')
plt.show()

fig, axs = plt.subplots(1, 10, figsize=(20, 5))
for i in range(10):
    axs[i].imshow(set_images_sphere[i][0], cmap='gray')
    axs[i].axis('off')
plt.show()