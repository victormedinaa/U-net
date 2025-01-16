# UNet para la Identificación de Imágenes Segmentadas

## Descripción
Este proyecto implementa una red neuronal convolucional U-Net para la segmentación semántica de imágenes, utilizando PyTorch. La red ha sido entrenada en un conjunto de datos de imágenes de monedas, con máscaras correspondientes para la segmentación.

## Requisitos

Para ejecutar este proyecto, se necesitan las siguientes dependencias:

- Python 3.x
- PyTorch
- torchvision
- numpy
- matplotlib
- wandb (para el seguimiento del entrenamiento)

Puedes instalarlas utilizando el siguiente comando:
```bash
pip install torch torchvision numpy matplotlib wandb
```


## Uso
### 1. Montar Google Drive
El proyecto está diseñado para ejecutarse en Google Colab, por lo que se necesita montar Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Cargar Datos
Asegúrate de tener las imágenes organizadas dentro de `ImagenesMonedas/` y luego ejecuta el código para cargar los datos y preprocesarlos.

```python
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
import torchvision.transforms as T

class Dataset(Dataset):
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

### 3. Definir la Arquitectura de U-Net
El modelo U-Net se define en `SegmentacionSemanticaVF.ipynb`, con capas de codificación, decodificación y clasificador final.

```python
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Definición de bloques de codificación y decodificación

    def forward(self, x):
        # Implementación del paso hacia adelante
        return x
```

### 4. Entrenamiento
El entrenamiento se realiza con la librería `wandb` para visualizar la pérdida durante las épocas.
```python
import wandb
wandb.init(project="segmentacion-monedas")

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

## Resultados
Los resultados del entrenamiento y la segmentación pueden visualizarse en [Weights & Biases](https://wandb.ai/).

## Contribución
Si deseas contribuir, realiza un fork del repositorio, crea una rama y envía un pull request.



