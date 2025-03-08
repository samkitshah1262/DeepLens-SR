import torch
import wandb
from torch.cuda.amp import GradScaler
import time
import numpy as np
from utils.logger import WandBLogger
from torch.nn import functional as F
from utils.metrics import calculate_ssim,calculate_psnr

class DyMoETrainer:
    def __init__(self, model, train_loader, val_loader, 
                 optimizer, criterion, device, config,
                 scheduler=None, use_amp=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.device = device
        self.scheduler = scheduler
        self.scaler = GradScaler(device, enabled=use_amp)
        self.best_val_loss = float('inf')
        self.logger = WandBLogger(config, model)
        self.log_interval = config.get('log_interval', 50)
        self.sample_interval = config.get('sample_interval', 200)
        self.best_val_loss = float('inf')
        self.best_epoch = 0 

    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        total_sparsity = 0
        psnr_values = []
        ssim_values = []
        mse_values = []
        sparsity_values = []
        start_time = time.time()

        for batch_idx, (lr, hr) in enumerate(self.train_loader):
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad)
                    wandb.log({f"grad_norm/{name}": grad_norm})
        
                    if torch.isnan(grad_norm).any():
                        print(f"NaN gradients in {name}")
                        param.grad.data = torch.zeros_like(param.grad.data)

            self.optimizer.zero_grad()
            pred = self.model(lr)
            loss = self.criterion(pred, hr)

            # Sparsity regularization
            gate_weights = self.model.gate[3](self.model.gate[2](self.model.gate[1](self.model.gate[0](self.model.embed(lr)))))
            batch_sparsity_loss = torch.mean(torch.sum(gate_weights**2, dim=1))
            total_loss += (loss + self.config['sparsity_lambda'] * batch_sparsity_loss).item()
            total_sparsity += batch_sparsity_loss.item()
            
            self.scaler.scale(loss + self.config['sparsity_lambda'] * batch_sparsity_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update weights
            self.scaler.step(self.optimizer)
            self.scaler.update()
            with torch.no_grad():
                for p in self.model.parameters():
                    p.data = torch.clamp(p.data, -1, 1)
            batch_mse = F.mse_loss(pred, hr).item()
            batch_psnr = calculate_psnr(pred, hr)
            batch_ssim = calculate_ssim(pred, hr)
            psnr_values.append(batch_psnr)
            ssim_values.append(batch_ssim)
            mse_values.append(batch_mse)
            sparsity_values.append(batch_sparsity_loss)

            if batch_idx % self.log_interval == 0:
                self.logger.log_metrics({
                    "train/loss": loss.item(),
                    "train/batch_mse": batch_mse,
                    "train/batch_psnr": batch_psnr,
                    "train/batch_ssim": batch_ssim,
                    "train/batch_sparsity": batch_sparsity_loss,
                    "lr": self.optimizer.param_groups[0]['lr']
                }, commit=False)
                
            if batch_idx % self.sample_interval == 0:
                with torch.no_grad():
                    self.logger.log_images(lr[:1], pred[:1], hr[:1])
                
        avg_loss = total_loss / len(self.train_loader)
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        avg_mse = np.mean(mse_values)
        avg_sparsity = total_sparsity / len(self.train_loader)
        epoch_time = time.time() - start_time
       
        self.logger.log_metrics({
            "epoch": epoch,
            "train/avg_loss": avg_loss,
            "train/epoch_mse": avg_mse,
            "train/avg_psnr": avg_psnr,
            "train/avg_ssim": avg_ssim,
            "train/avg_sparsity": avg_sparsity,
            "epoch_time": epoch_time
        })
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        mse_values = []
        psnr_values = []
        ssim_values = []
        start_time = time.time()
        for lr, hr in self.val_loader:
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            if torch.isnan(lr).any() or torch.isinf(lr).any() or torch.isnan(hr).any() or torch.isinf(hr).any():
                print("Invalid input detected in validation!")
                continue
            outputs = self.model(lr)
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                print("Model produced NaN/Inf in validation!")
                continue
            loss = self.criterion(outputs, hr)

            total_loss += loss.item()
            mse_values.append(F.mse_loss(outputs, hr).item())
            psnr_values.append(calculate_psnr(outputs, hr))
            ssim_values.append(calculate_ssim(outputs, hr))
            
        avg_loss = total_loss / len(self.val_loader)
        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)
        avg_mse = np.mean(mse_values)
        epoch_time = time.time() - start_time
        
        self.logger.log_metrics({
            "val/loss": avg_loss,
            "val/mse": avg_mse,
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
            self.best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': avg_loss,
            }, 'equiformer_best.pth')
            print("Saved best model!")
            self.logger.log_model('equiformer_best.pth', {
                'epoch': epoch,
                'val_loss': avg_loss,
                'val_psnr': avg_psnr
            })
            
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step(avg_loss)
            
        return avg_loss

