# Logo Segmentation

This project performs semantic segmentation to detect masks in images using a custom U-Net model. It includes data preprocessing, training, validation, inference, and post-processing pipelines.

## 📁 Project Structure

```
.
│
├── train/            # Provided Dataset (70% USED FOR TRAINING 15% FOR VALIDATION AND 15% FOR TEST)
│   ├── img/          # Groundtruth images
│   └── mask/         # Groundtruth masks
│
├── test_set/         # Test set from random 42 data split of Provided Dataset (reproduced in train.py)
│   ├── img/          # Groundtruth images
│   └── mask/         # Groundtruth masks
│
├── inference_testdir/
│   ├── img/          # Random samples to test inference
│   └── mask/         # Directory for inference output
│
├── checkpoints/      # Saved model checkpoints
├── logs/             # Training logs (CSV format)
├── dataset.py        # Custom Dataset class with Albumentations transforms
├── model.py          # Custom U-Net model implementation
├── train.py          # Training and validation logic
├── utils.py          # Utility functions (IoU, Dice Loss, BCEDice Loss, post-processing)
├── infer.py          # Inference on new images
├── test_setup.py     # Testing script for model and data loading
├── evaluation.py     # Standalone evaluation script for benchmarking
└── README.md         # Project documentation
```

## 🧠 Model Architecture

- **Custom U-Net** 
- **No pre-trained weights** - trained from scratch
- **13.4M parameters** total
- **Encoder-Decoder** architecture with skip connections
- **Batch Normalization** and **ReLU** activations

## ⚙️ Requirements

Install dependencies via pip:

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `torch>=1.9.0` - PyTorch deep learning framework
- `torchvision>=0.10.0` - Computer vision utilities
- `albumentations>=1.0.0` - Image augmentation
- `opencv-python>=4.5.0` - Computer vision operations
- `pillow>=8.0.0` - Image processing
- `numpy>=1.21.0` - Numerical computing
- `scikit-learn>=1.0.0` - Data splitting
- `tqdm>=4.60.0` - Progress bars

## 🧪 Data Preparation

Your `train/` folder should contain:

- `img/`: Input RGB `.jpg` files
- `mask/`: Corresponding binary `.bmp` masks (0 for background, 255 for logo)

**Data Split:**
- **70% Training** (15,790 images)
- **15% Validation** (3,383 images)  
- **15% Test** (3,383 images)

## 🚀 Training

Train the model using:

```bash
python train.py
```

**Features:**
- **Automatic data splitting** into train/val/test
- **Albumentations augmentations**: Horizontal flip, brightness/contrast, shift/scale/rotate
- **BCEDiceLoss**: Combines Binary Cross Entropy and Dice Loss for better performance
- **Learning rate scheduling**: ReduceLROnPlateau based on validation IoU
- **Gradient clipping**: Prevents gradient explosion
- **Early stopping**: Stops when validation IoU doesn't improve for 5 epochs
- **Checkpoint saving**: Saves best model based on validation IoU
- **CSV logging**: Tracks training metrics over time

**Training Results:**
- **Final Validation IoU**: 84.71%
- **Training Loss**: 0.1220
- **Model Performance**: Excellent segmentation quality 81.93% IoU on test set

## 📊 Validation

- **IoU calculation** after each epoch
- **Learning rate reduction** when validation IoU plateaus
- **Best model selection** based on validation performance
- **Early stopping** to prevent overfitting

## 🔍 Inference

Run inference on a folder of images:

```bash
python infer.py <input_folder> <output_folder>
```

**Example:**
```bash
python infer.py /inference_test_samples/imgs /inference_test_samples/mask
```

**Features:**
- **Batch processing** of multiple images
- **Automatic resizing** to model input size (340×512)
- **Binary mask output** with same filenames as input
- **Device detection**: Automatically uses MPS (Apple Silicon), CUDA, or CPU

**Output:**
- Binary segmentation masks saved as `.png` files
- Filenames match input images exactly

## 🧪 Testing

Test the model setup and data loading:

```bash
python test_setup.py
```

**Tests:**
- Data loading and preprocessing
- Model forward pass
- Loss function calculations
- IoU computation
- Trained model performance (if checkpoint exists)

## 📈 Evaluation

Evaluate your model on test datasets using:

```bash
python evaluation.py --testset_path ./test/ --prediction_path ./predictions/
```

**Features:**
- Automated inference generation if predictions don't exist
- Batch evaluation on multiple test directories
- Computes mean IoU scores for each test set

## 📦 Loss Functions

### **DiceLoss**
- Measures overlap between predicted and ground truth masks
- Effective for class imbalance (small objects)
- Formula: `Dice = (2 × |X ∩ Y|) / (|X| + |Y|)`

### **BCEDiceLoss**
- Combines Binary Cross Entropy and Dice Loss
- BCE: Pixel-wise classification loss
- Dice: Shape overlap loss
- Often provides better convergence than either loss alone

## ✨ Key Features

- **Custom U-Net** implementation (no external dependencies)
- **Data augmentation** using Albumentations
- **Advanced loss functions** (BCEDiceLoss)
- **Learning rate scheduling** based on validation performance
- **Gradient clipping** for training stability
- **Early stopping** to prevent overfitting
- **Comprehensive logging** and checkpointing
- **Cross-platform support** (MPS, CUDA, CPU)
- **Clean, modular codebase** for easy experimentation

## 📚 Performance

- **Validation IoU**: 84.71%
- **Model Parameters**: 13.4M
- **Training Time**: ~30 minutes per epoch (MPS)
- **Memory Usage**: ~8GB (batch size 8)

## 📚 Citations

- **U-Net**: Ronneberger et al., 2015
- **Dice Loss**: Milletari et al., 2016
- **Albumentations**: https://github.com/albumentations-team/albumentations
