# UNet for Segmented Image Identification

## Description
This project implements a U-Net convolutional neural network for semantic image segmentation using PyTorch. The network is trained on a dataset of coin images with corresponding masks for segmentation.

## Requirements

To run this project, you need the following dependencies:

- Python 3.x  
- PyTorch  
- torchvision  
- numpy  
- matplotlib  
- wandb (for training tracking)  

You can install them using the following command:
```bash
pip install torch torchvision numpy matplotlib wandb
```

## Usage

### 1. Mount Google Drive
The project is designed to run on Google Colab, so Google Drive needs to be mounted:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Load Data
Ensure that images are organized inside `ImagenesMonedas/` and then execute the code to load and preprocess the data.

```python
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as T

class CoinDataset(Dataset):
    def __init__(self, data_dir, label_dir):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.files = [f.split('.')[0] for f in os.listdir(data_dir)]
        self.transform = T.Resize((512, 512))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        image = read_image(f"{self.data_dir}/{file_name}.png", mode=ImageReadMode.RGB)
        mask = read_image(f"{self.label_dir}/{file_name}_mask.png", mode=ImageReadMode.GRAY)
        image = self.transform(image) / 255.0
        mask = torch.round(self.transform(mask))
        return image, mask
```

### 3. Define the U-Net Architecture
The U-Net model is defined in `SegmentacionSemanticaVF.ipynb`, including encoding, decoding, and final classifier layers.

```python
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Define encoding and decoding blocks

    def forward(self, x):
        # Implement forward pass
        return x
```

### 4. Training
Training is performed using the `wandb` library to visualize the loss over epochs.

```python
import wandb
wandb.init(project="coin-segmentation")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(40):
    for images, labels in DataLoader(ds, batch_size=4):
        outputs = model(images.to(device))
        loss = criterion(outputs, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        wandb.log({"loss": loss.item()})
```

## Results
Training and segmentation results can be viewed on [Weights & Biases](https://wandb.ai/).

## Contribution
If you'd like to contribute, fork the repository, create a branch, and submit a pull request.
