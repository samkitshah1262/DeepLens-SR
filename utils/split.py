import os
import shutil
from sklearn.model_selection import train_test_split
from constants import DATA_A_PATH,LR,HR,SPLIT_DATA_PATH_A,DATA_B_PATH,SPLIT_DATA_PATH_B
# Define dataset paths
dataset_root = DATA_A_PATH # Change this to your dataset path
lr_dir = os.path.join(dataset_root, LR)
hr_dir = os.path.join(dataset_root, HR)

# Define output directories
output_root = SPLIT_DATA_PATH_A  # Change this to your desired output path
train_lr_dir = os.path.join(output_root, "train", LR)
train_hr_dir = os.path.join(output_root, "train", HR)
val_lr_dir = os.path.join(output_root, "val", LR)
val_hr_dir = os.path.join(output_root, "val", HR)

# Create train/val directories
for dir_path in [train_lr_dir, train_hr_dir, val_lr_dir, val_hr_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Get all sample file names
lr_files = sorted(os.listdir(lr_dir))  # Ensure matching order
hr_files = sorted(os.listdir(hr_dir))

# Ensure pairs match
assert len(lr_files) == len(hr_files), "Mismatch in LR and HR files!"

# Split dataset (90% train, 10% val)
train_lr, val_lr, train_hr, val_hr = train_test_split(
    lr_files, hr_files, test_size=0.1, random_state=42
)

# Function to copy files
def move_files(files, src_dir, dest_dir):
    for file in files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(dest_dir, file))

# Move files to respective directories
move_files(train_lr, lr_dir, train_lr_dir)
move_files(train_hr, hr_dir, train_hr_dir)
move_files(val_lr, lr_dir, val_lr_dir)
move_files(val_hr, hr_dir, val_hr_dir)

print("Dataset split completed successfully!")

dataset_root = DATA_B_PATH # Change this to your dataset path
lr_dir = os.path.join(dataset_root, LR)
hr_dir = os.path.join(dataset_root, HR)

# Define output directories
output_root = SPLIT_DATA_PATH_B  # Change this to your desired output path
train_lr_dir = os.path.join(output_root, "train", LR)
train_hr_dir = os.path.join(output_root, "train", HR)
val_lr_dir = os.path.join(output_root, "val", LR)
val_hr_dir = os.path.join(output_root, "val", HR)

# Create train/val directories
for dir_path in [train_lr_dir, train_hr_dir, val_lr_dir, val_hr_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Get all sample file names
lr_files = sorted(os.listdir(lr_dir))  # Ensure matching order
hr_files = sorted(os.listdir(hr_dir))

# Ensure pairs match
assert len(lr_files) == len(hr_files), "Mismatch in LR and HR files!"

# Split dataset (90% train, 10% val)
train_lr, val_lr, train_hr, val_hr = train_test_split(
    lr_files, hr_files, test_size=0.1, random_state=42
)

# Function to copy files
def move_files(files, src_dir, dest_dir):
    for file in files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(dest_dir, file))

# Move files to respective directories
move_files(train_lr, lr_dir, train_lr_dir)
move_files(train_hr, hr_dir, train_hr_dir)
move_files(val_lr, lr_dir, val_lr_dir)
move_files(val_hr, hr_dir, val_hr_dir)

print("Dataset split completed successfully!")