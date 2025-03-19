### Approach

For this task I trained a Masked Auto Encoder on the no_sub samples to learn a feature representation of strong lensing images.

For both the task the results were surprisingly bad. I tried a few things like changing model architecture, retraining MAE with different configurations but no success. Did not have much time to resolve the issue. Run informations and model details can be found as follows.

  
[Model Reference](https://github.com/IcarusWizard/MAE) 



## Model details 

The **MAE-ViT** model consists of:
- An **Encoder** that:
  - Divides the image into patches.
  - Randomly shuffles and masks a percentage of patches.
  - Processes visible patches through a Transformer encoder.

- A **Decoder** that:
  - Reconstructs the original image from the encoded features and masked tokens.
  - Uses a Transformer decoder to predict missing patches.

---

### Components

1. PatchShuffle
   - Shuffles patch indices with a specified mask_ratio.
   - Keeps only unmasked patches and remembers shuffle order for reconstruction.

2. MAE_Encoder
   - **Patchify layer:** Conv2d to turn image into patch embeddings.
   - **Positional Embeddings:** Added to patch embeddings.
   - **Patch Shuffle:** Randomly removes a fraction of patches.
   - **Transformer Encoder:** num_layer layers, emb_dim embedding size, num_head heads each.
   - **LayerNorm** at the end.
   - Outputs feature embeddings and shuffle indices.

3. MAE_Decoder
   - Takes encoded features and reintroduces masked tokens.
   - Adds positional embeddings.
   - **Transformer Decoder:** num_layer layers, emb_dim embedding size, num_head heads.
   - Fully connected head to predict pixel values for each patch.
   - Rearranges predicted patches back into the image.

4. MAE_ViT
   - Combines the encoder and decoder.
   - End-to-end training for self-supervised learning and reconstruction.

### Key Hyperparameters

| Parameter             | Encoder / Decoder      | Default Value |
|-----------------------|------------------------|---------------|
| image_size          | Both                   | 64            |
| patch_size          | Both                   | 4             |
| emb_dim             | Both                   | 192           |
| encoder_layer       | Encoder                | 12            |
| encoder_head        | Encoder                | 3             |
| decoder_layer       | Decoder                | 4             |
| decoder_head        | Decoder                | 3             |
| mask_ratio          | Encoder                | 0.75          |


### Data Flow Graph (Conceptual)

Input Image (3x64x64)
   │
   ├─> Patchify (Conv2D with stride=patch_size)
   │
   ├─> Flatten + Add Positional Embeddings
   │
   ├─> Shuffle & Mask (PatchShuffle)
   │
   ├─> Transformer Encoder (12 layers)
   │
   ├─> Features + Backward indexes
   │
   ├─> Add Mask Tokens & Unshuffle (Decoder)
   │
   ├─> Add Positional Embeddings
   │
   ├─> Transformer Decoder (4 layers)
   │
   ├─> Linear head → Patch predictions
   │
   └─> Rearrange → Reconstructed image

## Results
[MAE Train](https://api.wandb.ai/links/samkitshah1262-warner-bros-discovery/zdlkmwqd) \
[Task A Classification](https://api.wandb.ai/links/samkitshah1262-warner-bros-discovery/noelv2qu) \
[Task B Super Resolution](https://api.wandb.ai/links/samkitshah1262-warner-bros-discovery/mfjq83m4)

## Analysis

Masked Auto Encoder Training

| Model | Loss | epochs |
| --- | --- | --- |
| v0 |  0.006 |	 200 |
| v1 | 0.015  |  75 |

---

Downstream Classification Task

| Model | Loss | AUC | epochs |
| --- | --- | --- | --- |
| best |  0.0017 |	0.599  |  200 |

---

Downstream Super Resolution Task

| Model | MSE | SSIM  | PSNR | epochs |
| --- | --- | --- | --- | --- |
| best |  0.034 |	0.634  | 14.81 |  20 |

