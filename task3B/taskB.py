from trainer import DyMoETrainer
import torch
from utils.dataset import LensDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from constants import PATH_TRAIN_HR_B,PATH_TRAIN_LR_B,PATH_VAL_HR_B,PATH_VAL_LR_B,WANDB_PROJECT,WANDB_USERNAME, SAVED_MODEL_PATH
from task3B.model import DyMoE_SR
from task3A.loss import HybridLoss


def load_pretrained(config, model):
    checkpoint = torch.load(config['pretrained_path'])
    pretrained_dict = {
        k.replace('blocks.', 'experts.0.'): v 
        for k, v in checkpoint.items()
        if 'upsampler' not in k
    }
    
    model.load_state_dict(pretrained_dict, strict=False)
    
    # Freeze shared components if specified
    if config['freeze_backbone']:
        for name, param in model.named_parameters():
            if 'experts' in name and '0' in name:
                param.requires_grad = False

def train_model(config):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(device)
        
    model = DyMoE_SR(
        dim=config['dim'],
        num_blocks=config['num_blocks'],
        num_heads=config['num_heads'],
        num_experts=config['num_experts']
    )

    load_pretrained(config, model)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=3,
        verbose=True
    )
    
    criterion = HybridLoss(device=device)
    
    # preprocessor = LensDataPreprocessor(crop_size=75)
    # transforms = preprocessor.get_transforms()

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
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    trainer = DyMoETrainer(
        model = model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        config=config,
        use_amp=config['use_amp'],
    )
    
    # Training loop
    try:
        for epoch in range(1, config['epochs'] + 1):
            train_loss = trainer.train_epoch(epoch)
            val_loss = trainer.validate(epoch)

            # Early stopping
            if (epoch - trainer.best_epoch) > config['patience']:
                print(f"Early stopping at epoch {epoch}")
                break

    finally:
        trainer.logger.finish()


config = {
    'dim': 64,
    'num_blocks': 8,
    'num_heads': 4,
    'upscale': 2,
    'lr': 1e-4,
    'weight_decay': 1e-4,
    'batch_size': 4,
    'epochs': 100,
    'patience': 10,
    'use_amp': True,
    'train_lr_dir': PATH_TRAIN_LR_B,
    'train_hr_dir': PATH_TRAIN_HR_B,
    'val_lr_dir': PATH_VAL_LR_B,
    'val_hr_dir': PATH_VAL_HR_B,
    'transform': transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5]) 
    ]),
    'wandb_project': WANDB_PROJECT,
    'wandb_entity': WANDB_USERNAME,
    'tags': ['gsoc2025'],
    'log_interval': 50,
    'sample_interval': 200,
    'architecture': 'DyMoE-SR',
    'pretrained_path': SAVED_MODEL_PATH,
    'num_experts': 8,
    'freeze_backbone': True,
    'sparsity_lambda': 0.01,
}


train_model(config)

