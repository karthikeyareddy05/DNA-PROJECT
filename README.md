# Plant Species Detection Project

A deep learning project for classifying plant species using TensorFlow/Keras and MobileNetV2 architecture.

## Overview

This project uses a pre-trained MobileNetV2 model to classify different plant species from images. The model has been trained on a diverse dataset of 30 different plant species including fruits, vegetables, and medicinal plants.

## Features

- **30 Plant Species Classification**: Including aloe vera, banana, coconut, mango, papaya, and many more
- **Transfer Learning**: Uses MobileNetV2 pre-trained on ImageNet
- **Data Augmentation**: Includes rotation, zoom, and horizontal flip for better generalization
- **Easy Prediction**: Simple script to predict plant species from new images

## Dataset

The dataset is organized into three folders:
- `Train_Set_Folder/`: Training images (~800 images per class)
- `Validation_Set_Folder/`: Validation images (~100 images per class)  
- `Test_Set_Folder/`: Test images (~100 images per class)

### Supported Plant Species

1. Aloe Vera
2. Banana
3. Bilimbi
4. Cantaloupe
5. Cassava
6. Coconut
7. Corn
8. Cucumber
9. Curcuma
10. Eggplant
11. Galangal
12. Ginger
13. Guava
14. Kale
15. Long Beans
16. Mango
17. Melon
18. Orange
19. Paddy
20. Papaya
21. Pepper Chili
22. Pineapple
23. Pomelo
24. Shallot
25. Soybeans
26. Spinach
27. Sweet Potatoes
28. Tobacco
29. Water Apple
30. Watermelon

## Requirements

```bash
tensorflow>=2.0.0
numpy
PIL (Pillow)
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/karthikeyareddy05/DNA-PROJECT.git
cd DNA-PROJECT
```

2. Install required packages:
```bash
pip install tensorflow numpy pillow
```

## Usage

### Training the Model

To train the model on your dataset:

```bash
python train_plant_classifier.py
```

This script will:
- Load and preprocess the dataset
- Create data generators with augmentation
- Train a MobileNetV2-based model
- Save the best model as `best_model.h5`
- Evaluate on the test set

### Making Predictions

To predict the species of a plant image:

```bash
python predict_plant.py
```

Then enter the path to your plant image when prompted.

### Example Usage in Code

```python
from predict_plant import predict_species

# Predict species from image
predict_species("path/to/your/plant_image.jpg")
```

## Model Architecture

- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Input Size**: 224x224x3
- **Custom Head**: GlobalAveragePooling2D + Dropout(0.3) + Dense(128, ReLU) + Dense(num_classes, Softmax)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

## Training Configuration

- **Batch Size**: 64
- **Epochs**: 5 (with early stopping)
- **Data Augmentation**: Rotation (20°), Zoom (0.2), Horizontal Flip
- **Callbacks**: EarlyStopping (patience=3), ModelCheckpoint

## File Structure

```
DNA-PROJECT/
├── dataset/
│   ├── Train_Set_Folder/
│   ├── Validation_Set_Folder/
│   └── Test_Set_Folder/
├── train_plant_classifier.py
├── predict_plant.py
├── best_model.h5
├── .gitignore
└── README.md
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

## Acknowledgments

- Dataset collected from various sources
- MobileNetV2 architecture by Google
- TensorFlow/Keras framework
