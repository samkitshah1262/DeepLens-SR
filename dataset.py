import os
import numpy as np
import torch

class LensDataset(torch.utils.data.Dataset):
    """Strong lensing dataset handler for .npy files"""

    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_files = sorted(os.listdir(lr_dir))  # Ensure order
        self.hr_files = sorted(os.listdir(hr_dir))
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.transform = transform

        # Ensure pairs match
        assert len(self.lr_files) == len(self.hr_files), "Mismatch in LR and HR files!"

    def __len__(self):
        return len(self.lr_files)

    def __getitem__(self, idx):
        # Load numpy arrays
        lr = np.load(os.path.join(self.lr_dir, self.lr_files[idx]))
        hr = np.load(os.path.join(self.hr_dir, self.hr_files[idx]))

        # Convert to torch tensors
        lr = torch.tensor(lr, dtype=torch.float32)
        hr = torch.tensor(hr, dtype=torch.float32) 

        # Apply transforms if provided
        if self.transform:
            lr = self.transform(lr)
            hr = self.transform(hr)

        return lr, hr
    



# set_images_lr = []
# for i in range(10):
#     set_images_lr.append(np.load(f"dataset3b/LR/LR_{i+1}.npy"))
# set_images_hr = []
# for i in range(10):
#     set_images_hr.append(np.load(f"dataset3b/HR/HR_{i+1}.npy"))


# img = np.load("dataset3b/LR/LR_1.npy")
# img = torch.tensor(img, dtype=torch.float32) 
# print("LR shape: ",img.shape)
# img = np.load("dataset3b/HR/HR_1.npy")
# img = torch.tensor(img, dtype=torch.float32) 
# print("HR shape: ",img.shape)

# fig, axs = plt.subplots(1, 10, figsize=(20, 5))
# for i in range(10):
#     axs[i].imshow(set_images_lr[i][0], cmap='gray')
#     axs[i].axis('off')
# plt.show()

# fig, axs = plt.subplots(1, 10, figsize=(20, 5))
# for i in range(10):
#     axs[i].imshow(set_images_hr[i][0], cmap='gray')
#     axs[i].axis('off')
# plt.show()
    
# def load_data(data_dir):
#     """ Loads .npy files and assigns labels based on folder names. """
#     file_paths, labels = [], []
    
#     class_names = ["no", "sphere", "vort"]
#     label_map = {name: idx for idx, name in enumerate(class_names)}

#     for class_name in class_names:
#         class_dir = os.path.join(data_dir, class_name)
#         for file in os.listdir(class_dir):
#             if file.endswith('.npy'):
#                 file_paths.append(os.path.join(class_dir, file))
#                 labels.append(label_map[class_name])

#     return file_paths, labels, label_map
