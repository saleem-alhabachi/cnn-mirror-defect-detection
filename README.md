# CNN-Based Mirror Defect Detection System

This project is a computer vision solution designed to automate defect detection in mirror surfaces using Convolutional Neural Networks (CNN). The model distinguishes between defective and non-defective mirror images to enhance quality control in industrial environments.

## 🔍 Project Goals

- Detect surface defects in mirrors after the painting process.
- Reduce human error and speed up quality inspection.
- Improve defect classification accuracy using CNN models.

## 🛠️ Tools & Technologies

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Matplotlib
- Jupyter Notebook or PyCharm (for development)

## 📁 Dataset

The dataset consists of two main classes:
- **Defect images** (e.g. scratches, smudges, paint issues)
- **Non-defect images** (clean mirror surfaces)

> Note: The full dataset is not included in this repository due to file size. Contact the author for access if needed.

## 🧠 Model Overview

- A custom CNN architecture trained on 1000 defect and 1000 non-defect images.
- Data was split into training, validation, and test sets.
- Model evaluation includes accuracy, precision, recall, and F1-score.

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mirror-defect-detection.git
