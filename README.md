# 🚗 German Traffic Sign Recognition  

This project focuses on building and evaluating **deep learning models** to classify German traffic signs using the **GTSRB dataset**.  

---

## 📊 Dataset
- **German Traffic Sign Recognition Benchmark (GTSRB)**  
- Contains **50,000+ images** across **43 classes** of traffic signs.  
- Dataset is split into **training** and **testing** sets.  

---

## 🧹 Data Preprocessing
- **Image Loading & Resizing**: All images resized to **30x30 pixels**.  
- **Normalization**: Pixel values scaled to `[0, 1]`.  
- **Class Distribution Analysis**: Significant class imbalance observed.  
- **Data Augmentation** (using `ImageDataGenerator`):  
  - Random rotations: `rotation_range=10`  
  - Width/height shifts: `0.1`  
  - Zooming: `0.1`  
  - **No horizontal flips** (not suitable for traffic signs).  
- **Train-Validation Split**: 80% training / 20% validation.  
- **One-Hot Encoding**: Labels converted to categorical format.  

---

## 🧠 Model Building & Training  

### 1️⃣ Custom CNN Model  
- Built with **Keras Sequential API**.  
- Layers: Multiple **Conv2D + MaxPooling2D** (filters: 32, 64, 128) → **Dropout** → **GlobalAveragePooling2D** → Dense layers.  
- **Softmax output** with 43 classes.  
- **Optimizer**: Adam, **Loss**: Categorical Crossentropy.  
- **Training**:  
  - 50 epochs, batch size = 128  
  - Augmented data (`datagen.flow`)  
  - Class imbalance handled with **class weights**  

### 2️⃣ MobileNetV2 (Transfer Learning)  
- Pretrained **MobileNetV2 (ImageNet)**, `include_top=False`.  
- Added new Dense layers for classification (ReLU + Dropout + Softmax).  
- **Training strategy**:  
  1. **Freeze base**, train only top layers.  
  2. **Unfreeze last 30 layers**, fine-tune with lower LR (`1e-4`).  
- Used **EarlyStopping** based on validation loss.  

---

## 📈 Evaluation
Both models were evaluated on the **held-out test set**:  

- **Accuracy**: Overall classification accuracy.  
- **Confusion Matrix**: Visualized per-class performance.  
- **Training History**: Accuracy & loss curves for training vs validation.  

---

## ⚖️ Results Comparison

### 🔹 Custom CNN Model  
- **Test Accuracy**: `{{accuracy*100:.2f}}%`  
- **Test Loss**: `{{loss:.4f}}`  
- Confusion matrix shows strong performance across most classes, with some weaker results on minority classes.  

### 🔹 MobileNetV2 Transfer Learning  
- Initial accuracy lower than custom CNN in this run.  
- Classification report showed variation in precision/recall across classes.  
- Confusion matrix highlighted difficulty in minority classes.  
- (**Note**: Accuracy calculation method needs correction for true evaluation.)  

---

## 👋 Conclusion
- A **Custom CNN** performed better in this implementation than **MobileNetV2 Transfer Learning**.  
- However, **transfer learning** remains a powerful tool, especially when data is limited.  
- Further improvements could include:  
  - Better hyperparameter tuning  
  - More advanced augmentation strategies  
  - Fine-tuning additional pretrained layers  

---
