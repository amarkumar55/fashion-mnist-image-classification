# Fashion-MNIST Image Classification using CNN (PyTorch)

This project implements a Convolutional Neural Network (CNN) using PyTorch to classify grayscale images from the Fashion-MNIST dataset into 10 clothing categories. The project demonstrates an end-to-end deep learning workflow including data preprocessing, model design, training, evaluation, and model persistence.

---

## 📌 Project Overview

Fashion-MNIST is a popular benchmark dataset consisting of 70,000 grayscale images of fashion products such as shirts, shoes, bags, and coats. Each image is 28×28 pixels.  
In this project, the images are resized and used to train a CNN that learns spatial patterns for accurate classification.

---

## 🧠 Model Architecture

The CNN architecture includes:

- Two convolutional layers with ReLU activation
- Max pooling layers for spatial downsampling
- Fully connected layers for classification
- Dropout for regularization

**Architecture Summary:**
- Input: 1 × 16 × 16 grayscale image
- Conv2D → ReLU → MaxPool
- Conv2D → ReLU → MaxPool
- Fully Connected → Dropout
- Output: 10-class prediction

---

## 🗂 Dataset

- **Name:** Fashion-MNIST
- **Source:** Public dataset by Zalando
- **Classes:** 10
- **Total Images:** 70,000
  - Training: 60,000
  - Testing: 10,000

The dataset is automatically downloaded using `torchvision.datasets.FashionMNIST`.

---

## 🔧 Data Preprocessing

- Image resizing to 16×16
- Conversion to PyTorch tensors
- Normalization to improve training stability

```python
transforms.Normalize((0.5,), (0.5,))
```

### Training Details
    
    Framework: PyTorch
    Loss Function: CrossEntropyLoss
    Optimizer: Adam
    Learning Rate: 0.001
    Epochs: 10
    Batch Size: 64
    Device: CPU / GPU (CUDA if available)


### Evaluation

    The trained model is evaluated on a held-out test set using classification accuracy.
    Test Accuracy: ~88%


### Model Saving

    The trained model weights are saved for reuse or deployment:    
    torch.save(model.state_dict(), "cnn_fashion_mnist.pth")

### Project Structure
```text

fashion-mnist-cnn/
│
├── data/                    
├── cnn_fashion_mnist.pth     
├── fashion_mnist_cnn.ipynb  
├── README.md

```

### Key Learnings

    Building CNNs from scratch in PyTorch
    Image preprocessing and normalization
    Training and evaluating deep learning models
    Saving trained models for future use

### Future Improvements

    Increase image resolution    
    Add batch normalization
    Hyperparameter tuning
    Confusion matrix & per-class accuracy
    Model deployment using FastAPI


📜 License

    This project uses a publicly available dataset and contains original implementation code for educational and portfolio purposes.
