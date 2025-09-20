# 🖼️ CIFAR-10 Image Classification with Deep Learning

A deep learning project that classifies 10 categories of real-world images from the **CIFAR-10** dataset using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

---

## 📌 Project Overview

This project demonstrates how to build and train a convolutional neural network to classify **color images** from the **CIFAR-10 dataset**, which contains **60,000 images (32x32 pixels, RGB)** across **10 classes**:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

The dataset is balanced, with 6,000 images per class.

---

## 🧠 Model Architecture

The CNN used in this project follows a simple and effective architecture:

- **Input**: 32x32x3 RGB images  
- **Conv2D + MaxPooling** layers  
- **Flatten**  
- **Dense layers** with ReLU activation  
- **Output**: 10 neurons with Softmax activation (one for each class)

> ✅ **Optimizer**: Adam  
> 📉 **Loss Function**: Sparse Categorical Crossentropy  
> 🧪 **Evaluation Metric**: Accuracy

---

## ⚙️ Training Strategy

To optimize training and prevent overfitting, the following **Keras callbacks** were used:

- **EarlyStopping**: Stops training early if validation accuracy doesn't improve for several epochs.
- **ReduceLROnPlateau**: Reduces learning rate by a factor when the model stops improving, allowing for finer convergence.

---

## 📊 Results

- **Training Accuracy**: ~94%  
- **Test Accuracy**: ~72%  
- **Observations**:
  - Overfitting exists.

Sample accuracy/loss graphs are included in the notebook.

---

## 📁 Project Structure

```bash
.
├── cnn_cifar10.ipynb
├── README.md
└── requirements.txt
```

## 🔮 Future Work

- Improve accuracy using deeper CNN architectures (e.g., ResNet, VGG)

- Add data augmentation for better generalization

- Integrate model explainability (e.g., Grad-CAM)

- Deploy model using Streamlit or Flask

- Add hyperparameter tuning with Keras Tuner or Optuna