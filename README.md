# CIFAR-10 Image Classification Project

##  Overview

This project demonstrates image classification on the **CIFAR-10 dataset** using two different  models:  
- A **custom Convolutional Neural Network (CNN)**  
- A **ResNet50** model  

The goal is to classify images into one of the following 10 classes:  
`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

The project is implemented in **Python** using **TensorFlow/Keras** .

---

##  Project Structure

- `CV.ipynb`: The main Jupyter Notebook containing:
  - Loading and preprocessing the CIFAR-10 dataset.
  - Building and training a custom CNN model.
  - Evaluating the CNN model with metrics and a confusion matrix.
  - Implementing and evaluating a ResNet50-based model.

- `README.md`: This file, providing an overview and instructions for the project.

---

##  Dataset

The **CIFAR-10 dataset** consists of **60,000 32x32 color images**, split into:
- **50,000 training images**
- **10,000 test images**

It includes **10 classes**, and is loaded using TensorFlow’s `cifar10.load_data()` function.

---

##  Models

### Custom CNN:

- **Architecture**: Three convolutional blocks with increasing filters (32, 64, 128), each followed by max-pooling, then dense layers with dropout (0.3) to prevent overfitting.
- **Optimizer**: Adam  
- **Loss Function**: Categorical Crossentropy  
- **Training**: 10 epochs, batch size of 64  
- **Performance**: Achieves approximately **74% accuracy** on the test set.

---

### ResNet50:

- A pre-trained **ResNet50** model adapted for CIFAR-10 classification.
- **Performance**: Achieves approximately **91% accuracy** on the test set, with detailed metrics provided in a classification report and confusion matrix.

---

##  Results & Conclusion

The CIFAR-10 image classification project demonstrates the power of deep learning and transfer learning:

- The **custom CNN** achieved ~74% accuracy, providing a solid baseline for experimentation. Its simplicity makes it suitable for understanding CNN fundamentals.
- The **ResNet50**, leveraging pre-trained weights and deeper architecture, significantly outperformed the custom model with ~91% test accuracy. It demonstrated robust classification results across all 10 classes.

**Conclusion**:

- Transfer learning is highly effective for image classification tasks like CIFAR-10.
- Advanced architectures such as ResNet50 drastically improve model generalization and performance.
- Further improvements can be explored through data augmentation, model fine-tuning, and trying other modern architectures such as EfficientNet or Vision Transformers.


---

## ⚙️ Requirements

To run this project, ensure you have the following dependencies installed:

- Python 3.10  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

### Install using pip:

```bash
pip install tensorflow torch numpy matplotlib seaborn scikit-learn
