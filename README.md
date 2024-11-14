# COVID-19-Disease-Image-Classification-using-CNN-AI-Project

# Project Overview:
This project involves building a Convolutional Neural Network (CNN) model to classify chest X-ray images into three categories: Covid, Normal, and Viral Pneumonia Diseases. The goal is to create an automated tool that can assist healthcare professionals in the rapid and accurate identification of COVID-19 infections, helping to manage resources effectively during critical times.

# Project Details:
## Dataset:
The dataset consists of chest X-ray images for three categories: Covid, Normal, and Viral Pneumonia. The images are divided into training, validation, and testing sets to ensure that the model generalizes well to new, unseen data. Data augmentation techniques are applied to increase the diversity of the training set and reduce overfitting.

## Data Preprocessing and Augmentation:
The model uses a data pipeline with extensive preprocessing steps, including resizing images to 512x512 pixels and applying data augmentation. Augmentation techniques include random rotations, shifts, zooms, and horizontal flips, enhancing model robustness and generalization.

## Model Architecture:
A custom CNN model was developed using TensorFlow and Keras libraries. The model architecture includes:

- Convolutional Layers with ReLU activation functions for feature extraction.
- MaxPooling Layers for dimensionality reduction.
- Fully Connected Layers (Dense Layers) for decision-making and classification.
- Softmax Output Layer with three neurons for multiclass classification, outputting the probabilities for each class (Covid, Normal, Viral Pneumonia).

## Model Evaluation:
The model's performance was evaluated on separate testing and validation datasets to ensure accuracy and reliability. Metrics such as accuracy score were calculated to assess the model’s efficacy in classifying each category.

## Interactive Jupyter Notebook:
The project includes a detailed Jupyter Notebook (Project.ipynb) that contains the full pipeline—from data loading and preprocessing to model training, evaluation, and visualization of results. The notebook is organized and well-documented to provide insights into each step of the process, making it easy to understand and replicate.

# Usage:
- Clone the repository.
- Load the dataset into the specified folder structure.
- Run the Jupyter Notebook to preprocess data, train the model, and evaluate its performance.
- (Optional) Export the trained model using joblib to deploy or integrate with other applications.
- after load the joblib file use following code for prediction
  ```
  classes = ['Covid' , 'Normal' , 'Viral Pneumonia']
  def predict(path):
    img = load_img(path , target_size = (512,512,3))
    img_arr = img_to_array(img)
    norm= img_arr/255.0
    flat = np.expand_dims(norm , axis=0)
    pred = model.predict(flat)[0]
    clas = classes[np.argmax(pred)]
    return clas

  # for prediction
  predict('path of the image')
  
  ```

# Tools and Technologies:
Deep Learning Frameworks: TensorFlow, Keras
Data Augmentation and Preprocessing: ImageDataGenerator, Resizing, Augmentation
Model Saving: Joblib
Programming Language: Python

# Repository Structure:

Project.ipynb: Main notebook with code for data preprocessing, model training, evaluation, and visualizations.
Model dump file Covid_19_Disease_Image_classification_project.joblib is too large to upload but can be recreated using the provided code.
```
import joblib
from joblib import dump
dump(model, 'Covid_19_image_classification_project.joblib')
```
Dataset: Folder structure expected to have separate directories for training, validation, and testing images for each class.


This project demonstrates the application of deep learning in healthcare, showing how CNNs can effectively classify X-ray images to aid in COVID-19 diagnosis.
