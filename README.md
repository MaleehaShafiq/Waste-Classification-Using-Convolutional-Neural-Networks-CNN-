# Organic vs Recyclable Waste Classification using PyTorch

## Overview
This project focuses on classifying waste images into two categories: **Organic (O)** and **Recyclable (R)**. The model is built **from scratch using PyTorch**, and trained on a custom dataset containing fruits, vegetables, food waste, plastic bottles, wrappers, and similar items.  
The purpose is to explore how deep learning models can be applied to promote automated waste segregation and efficient recycling processes.

---

## Dataset
The dataset is divided into two main folders: `train` and `test`, each containing subfolders for the two categories — `O` (Organic) and `R` (Recyclable).  
A few AI-generated and augmented images were added to increase data diversity and improve model generalization.

**Structure:**
```
DATASET/
│
├── train/
│   ├── O/
│   └── R/
│
└── test/
    ├── O/
    └── R/
```

---

## Data Preprocessing and Augmentation
Before training, all images were resized and normalized. Data augmentation was applied to enhance model robustness and help prevent overfitting.

**Transformations:**
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.6892, 0.6416, 0.5668],
                         [0.2081, 0.2178, 0.2390])
])
```

---

## Model Architecture
A **Convolutional Neural Network (CNN)** was implemented from scratch using PyTorch.  
The architecture includes:
- Multiple convolutional and pooling layers for feature extraction  
- ReLU activation functions  
- Fully connected layers for classification  
- Softmax output layer for prediction

---

## Training Details
- **Framework:** PyTorch  
- **Optimizer:** Adam  
- **Loss Function:** CrossEntropyLoss  
- **Epochs:** 20  
- **Batch Size:** 32  

The model was trained on the custom dataset and evaluated on unseen test images.

---

## Results
The model demonstrated strong classification performance on clear and distinct images, such as:
- Correctly identifying vegetables, fruits, and food items as **Organic**
- Correctly classifying bottles, plastic, and paper items as **Recyclable**

However, some misclassifications were observed when the images contained ambiguous textures or colors (for example, crumpled colored paper being classified as Organic).

---

## Future Improvements
- Increase dataset size with more diverse, real-world images  
- Experiment with pretrained CNN architectures such as ResNet or EfficientNet  
- Use generative models for synthetic image creation  
- Deploy the trained model using Streamlit or Flask for interactive classification

---

## How to Run

1. Clone this repository:
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train the model:
```bash
python train.py
```

4. Test with a new image:
```bash
python predict.py --image path/to/image.jpg
```

---
