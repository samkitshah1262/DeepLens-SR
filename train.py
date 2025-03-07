# Example usage
import torch
import torch.nn as nn
from model import Equiformer
from dataset import LensDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from trainer import Trainer
from constants import PATH_TRAIN_HR,PATH_TRAIN_LR,PATH_VAL_HR,PATH_VAL_LR,WANDB_PROJECT,WANDB_USERNAME

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = Equiformer(
        dim=config['dim'],
        num_blocks=config['num_blocks'],
        num_heads=config['num_heads'],
        upscale=config['upscale']
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    criterion = nn.L1Loss()
    
    train_dataset = LensDataset(
        lr_dir=config['train_lr_dir'],
        hr_dir=config['train_hr_dir'],
        transform=config['transform']
    )
    
    val_dataset = LensDataset(
        lr_dir=config['val_lr_dir'],
        hr_dir=config['val_hr_dir'],
        transform=config['transform']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # config.update({
    #     'wandb_project': 'ml4sci-superres',
    #     'wandb_entity': 'your-username',
    #     'tags': ['equiformer', 'lensing', 'super-resolution'],
    #     'log_interval': 50,
    #     'sample_interval': 200
    # })

    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        use_amp=config['use_amp']
    )
    
    # Training loop
    try:
        for epoch in range(1, config['epochs'] + 1):
            train_loss = trainer.train_epoch(epoch)
            val_loss = trainer.validate(epoch)

            # Early stopping
            if epoch - trainer.best_epoch > config['patience']:
                print(f"Early stopping at epoch {epoch}")
                break


    finally:
        trainer.logger.finish()

# Example configuration
config = {
    'dim': 75,
    'num_blocks': 8,
    'num_heads': 4,
    'upscale': 4,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'batch_size': 16,
    'epochs': 100,
    'patience': 10,
    'use_amp': True,
    'train_lr_dir': PATH_TRAIN_LR,
    'train_hr_dir': PATH_TRAIN_HR,
    'val_lr_dir': PATH_VAL_LR,
    'val_hr_dir': PATH_VAL_HR,
    'transform': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]),
    'wandb_project': WANDB_PROJECT,
    'wandb_entity': WANDB_USERNAME,
    'tags': ['gsoc2025', 'diffilens'],
    'log_interval': 50,
    'sample_interval': 100,
    'architecture': 'Equiformer'
}

# Start training
train_model(config)