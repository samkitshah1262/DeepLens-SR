from torchvision import transforms
from PIL import Image
class LensDataPreprocessor:
    def __init__(self, crop_size=75, scale_factor=2):
        self.train_transform = transforms.Compose([
            transforms.RandomChoice([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip()
            ]),
            transforms.RandomRotation(15),
            transforms.RandomCrop(crop_size*scale_factor),
            transforms.Lambda(lambda x: self._degrade(x, scale_factor)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.CenterCrop(crop_size*scale_factor),
            transforms.Lambda(lambda x: self._degrade(x, scale_factor)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def _degrade(self, hr, scale_factor):
        lr_size = (hr.size[1]//scale_factor, hr.size[0]//scale_factor)
        return hr.resize(lr_size, Image.BICUBIC)

    def get_transforms(self):
        return {
            'train': PairedTransform(self.train_transform),
            'val': PairedTransform(self.val_transform)
        }

class PairedTransform:
    def __init__(self, transform):
        self.transform = transform
        
    def __call__(self, hr):
        lr = self.transform(hr)
        hr = self.transform(hr)
        return lr, hr