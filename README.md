# Hyperspectral-Deep-Learning
## 🔍 Hyperspectral Image Classification and Segmentation
A comprehensive implementation of deep learning models for hyperspectral image classification and segmentation using PyTorch. The project implements various architectures and techniques in a single streamlined pipeline.

## 📚 Table of Contents
1. **🔄 Data Acquisition and Preprocessing**
   - 📥 Downloading Datasets
   - 🔃 Data Format Conversion and Organization
   - ✂️ Dataset Splitting (80/10/10)
   - 📊 Band Quality Assessment
   - 🧹 Noise Removal

2. **📈 Data Exploration and Visualization**
   - 📉 Spectral Signatures Visualization
   - 🔗 Band Correlation Analysis
   - 📊 Class Distribution Analysis
   - 🎲 3D Datacube Visualizations

3. **🎯 Dimensionality Reduction and Feature Selection**
   - 🧮 PCA, ICA, LDA Analysis
   - 🎯 Band Selection Methods
   - ✨ Reduced Representations Evaluation

4. **🔄 Data Augmentation Strategies**
   - 🌊 Spectral Augmentation
   - 🌍 Spatial Augmentation
   - 🔄 Combined Spectral-Spatial Augmentation

5. **🏗️ Model Architecture Design**
   - 📋 Classification Models
     - 🔥 ResNet18 with Spectral Attention
     - 🤖 Vision Transformer (ViT)
     - 🧊 3D CNN
     - 🔄 Hybrid CNN-Transformer
   - 🎨 Segmentation Models
     - 🌈 U-Net with Spectral Attention
     - 🎲 3D U-Net
     - 🎯 FCN with Spectral Attention

6. **⚙️ Training Pipeline Implementation**
   - 📉 Loss Functions
     - 💫 Cross-entropy with Class Weights
     - 🎯 Dice Loss for Segmentation
   - 🔧 Optimization
     - ⚡ Adam Optimizer
     - 📈 Learning Rate Scheduling
   - 🔒 Regularization
     - 🎭 Spectral Dropout
     - 🏋️ L1/L2 Regularization

7. **📊 Model Training and Monitoring**
   - 🎯 Model Selection
   - 🏃‍♂️ Training and Validation
   - 💾 Checkpoint Management
   - 🛡️ Overfitting Handling

8. **📊 Results and Visualization**
   - 📈 Performance Charts
   - 🎨 Color-graded and Normalized Visualizations

## ✨ Features
### 📚 Supported Datasets
- 🌾 Indian Pines
- 🏛️ Pavia University
- 🌱 Salinas Scene

### 🛠️ Core Functionality
- 🔄 Automatic dataset downloading and processing
- 📊 Band quality assessment and noise removal
- 🔄 Spectral and spatial data augmentation
- 📉 Dimensionality reduction techniques
- ⚡ Advanced training features with regularization
- 📊 Comprehensive visualization and evaluation tools

## 📋 Requirements
```text
torch>=1.8.0
numpy>=1.19.2
pandas>=1.2.0
scipy>=1.6.0
scikit-learn>=0.24.0
matplotlib>=3.3.4
seaborn>=0.11.1
tqdm>=4.59.0
plotly>=4.14.0
```

## 🚀 Usage
1. Install the required dependencies:
```bash
pip install torch numpy pandas scipy scikit-learn matplotlib seaborn tqdm plotly
```

2. Run the main training pipeline:
```python
# Select dataset and models
SELECTED_DATASET = ['indian_pines']  # Options: 'indian_pines', 'pavia_university', 'salinas'
models = ['fcn']  # Options: 'resnet', 'vit', '3dcnn', 'hybrid', 'unet', 'fcn'
# Training will automatically:
# - Download and process the dataset
# - Train the selected models
# - Generate visualizations and metrics
```

## 🔑 Key Components
### 📊 Data Processing
```python
# Load and preprocess dataset
loader = HyperspectralDataLoader(dataset_name)
data, ground_truth = loader.load_dataset()
# Analyze band quality
analyzer = BandQualityAnalyzer(data, dataset_name)
noisy_bands = analyzer.identify_noisy_bands()
```

### 🏃‍♂️ Model Training
```python
# Create model
model = create_model(dataset_name)  # For classification
# or
model = create_fcn_model(dataset_name)  # For segmentation
# Train
trainer = Trainer(config, task_config)
trainer.train()
```

### 📈 Visualization
```python
# Plot results
plot_confusion_matrix(dataset_name, classification_models, device)
plot_segmentation_maps(dataset_name, segmentation_models, device)
plot_training_curves(metrics, dataset_name, model_type)
```

## 📊 Results
The code includes comprehensive evaluation tools that generate:
- 📈 Classification accuracy metrics
- 🎯 Segmentation IoU/Dice scores
- 📊 Confusion matrices
- 📉 ROC curves
- 🔍 Error analysis
- 📊 Band quality visualization
- 📈 Training progress curves
