import wandb
from pathlib import Path
import torch
import torch.nn.functional as F

class WandBLogger:
    def __init__(self, config, model):
        self.config = config
        self.run = wandb.init(
            project=config['wandb_project'],
            entity=config['wandb_entity'],
            config=config,
            tags=config.get('tags', ['super-resolution', 'lensing']),
            dir=str(Path.cwd())
        )

        wandb.watch(
            model,
            log='all',
            log_freq=config.get('log_interval', 50),
            log_graph=True
        )
        
    def log_metrics(self, metrics, step=None, commit=True):
        wandb.log(metrics, step=step, commit=commit)
        
    def log_images(self, lr, sr, hr, caption="LR/SR/HR Comparison"):
        # Resize LR to match HR/SR dimensions
        lr_upscaled = F.interpolate(lr, scale_factor=2, mode='bilinear')
        
        # Denormalize images
        lr_upscaled = (lr_upscaled * 0.5 + 0.5).clamp(0, 1)
        sr = (sr * 0.5 + 0.5).clamp(0, 1)
        hr = (hr * 0.5 + 0.5).clamp(0, 1)

        grid = torch.cat([lr_upscaled, sr, hr], dim=-1)  # Concatenate along width

        images = wandb.Image(grid, caption=caption)
        wandb.log({"Examples": images})
        
    def log_model(self, model_path, metadata=None):
        artifact = wandb.Artifact(
            name=f"model-{wandb.run.id}",
            type="model",
            description="Equiformer super-resolution model",
            metadata=metadata or {}
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        
    def finish(self):
        wandb.finish()
