import time
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from metrics import calculate_psnr, calculate_ssim
from logger import WandBLogger
import wandb

class Trainer:
    def __init__(self, model, train_loader, val_loader, 
                 optimizer, criterion, device, config,
                 scheduler=None, use_amp=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.scaler = GradScaler('cuda', enabled=use_amp)
        self.best_val_loss = float('inf')
        self.logger = WandBLogger(config, model)
        self.log_interval = config.get('log_interval', 50)
        self.sample_interval = config.get('sample_interval', 200)
        
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        psnr_values = []
        ssim_values = []
        start_time = time.time()
        
        for batch_idx, (lr, hr) in enumerate(self.train_loader):
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda',enabled=self.scaler.is_enabled()):
                outputs = self.model(lr)
                loss = self.criterion(outputs, hr)
                
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            batch_psnr = calculate_psnr(outputs, hr)
            batch_ssim = calculate_ssim(outputs, hr)
            psnr_values.append(batch_psnr)
            ssim_values.append(batch_ssim)
            
            if batch_idx % self.log_interval == 0:
                self.logger.log_metrics({
                    "train/loss": loss.item(),
                    "train/batch_psnr": batch_psnr,
                    "train/batch_ssim": batch_ssim,
                    "lr": self.optimizer.param_groups[0]['lr']
                }, commit=False)
                
            if batch_idx % self.sample_interval == 0:
                with torch.no_grad():
                    self.logger.log_images(lr[:1], outputs[:1], hr[:1])
                
        avg_loss = total_loss / len(self.train_loader)
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        epoch_time = time.time() - start_time
        
        self.logger.log_metrics({
            "epoch": epoch,
            "train/avg_loss": avg_loss,
            "train/avg_psnr": avg_psnr,
            "train/avg_ssim": avg_ssim,
            "epoch_time": epoch_time
        })
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        psnr_values = []
        ssim_values = []
        start_time = time.time()
        
        for lr, hr in self.val_loader:
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            
            outputs = self.model(lr)
            loss = self.criterion(outputs, hr)
            
            total_loss += loss.item()
            psnr_values.append(calculate_psnr(outputs, hr))
            ssim_values.append(calculate_ssim(outputs, hr))
            
        avg_loss = total_loss / len(self.val_loader)
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        epoch_time = time.time() - start_time
        
        self.logger.log_metrics({
            "val/loss": avg_loss,
            "val/psnr": avg_psnr,
            "val/ssim": avg_ssim,
            "epoch_time": epoch_time
        })

        self.logger.log_metrics({
            "val_output_dist": wandb.Histogram(outputs.cpu().numpy())
        })
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss,
            }, 'equiformer_best.pth')
            print("Saved best model!")
            self.logger.log_model(f'equiformer_best.pth', {
                'epoch': epoch,
                'val_loss': avg_loss,
                'val_psnr': avg_psnr
            })
            
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step(avg_loss)
            
        return avg_loss