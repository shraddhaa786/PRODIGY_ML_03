# PRODIGY_ML_03


# Cat and Dog Image Classification Using Support Vector Machine (SVM)

## Project Overview

This project focuses on classifying images of cats and dogs using a Support Vector Machine (SVM) algorithm. The dataset used for this task is sourced from Kaggle and consists of labeled images of cats and dogs.

### Dataset
The dataset used for training and testing the model is from Kaggle: [Cats and Dogs Images Dataset](https://www.kaggle.com/datasets/chetankv/dogs-cats-images/data).

- **Training set**: Contains images of cats and dogs stored in `training_set/cats` and `training_set/dogs`.
- **Test set**: Contains images of cats and dogs stored in `test_set/cats` and `test_set/dogs`.

### Problem Statement
The goal of this task is to train a model using a Support Vector Machine (SVM) algorithm to classify whether an image is of a cat or a dog. 

## Project Structure

- `archive (12).zip`: The original zipped dataset.
- `Dataset/`: Extracted dataset, containing training and test images.
- `svm_classifier.py`: Python code for preprocessing the images, training the SVM model, and making predictions.
- `README.md`: This file, explaining the project and its components.

## Approach

1. **Data Preprocessing**:
   - The images were loaded and resized to a uniform size of `50x50` pixels.
   - Each image was normalized by scaling pixel values between 0 and 1.
   - Labels were assigned to the images: `0` for cats and `1` for dogs.
   
2. **Feature Extraction**:
   - The image data was flattened into one-dimensional arrays to be used as input for the SVM model.

3. **Dimensionality Reduction**:
   - Principal Component Analysis (PCA) was applied to reduce the dimensionality of the feature space to 2 components, making it easier for visualization and computation.

4. **Model Training**:
   - An SVM model with a linear kernel was trained on the PCA-reduced data.
   - The model was trained using a portion of the dataset, with 20% of the data reserved for testing.

5. **Model Evaluation**:
   - The performance of the SVM model was evaluated using a classification report and a confusion matrix.
   - A decision boundary plot was generated to visualize how well the SVM separates the two classes (cats vs. dogs).

## Installation & Usage

### Requirements
- Python 3.x
- Libraries: 
  - `numpy`
  - `opencv-python`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `tqdm`

### Steps to Run the Code
1. Clone the repository and download the dataset from the [Kaggle link](https://www.kaggle.com/datasets/chetankv/dogs-cats-images/data).
   
2. Install the required Python libraries by running:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the SVM classification script:
   ```bash
   python svm_classifier.py
   ```

## Results

- The SVM model achieved good accuracy in classifying the test images.
- The confusion matrix and classification report indicated the model’s performance in distinguishing between the two classes.
- The decision boundary plot visualized the effectiveness of the SVM model on the PCA-reduced data.

## Conclusion

This project demonstrates the application of SVM for image classification, along with dimensionality reduction techniques like PCA to enhance the performance and interpretability of the model. It showcases the importance of proper data preprocessing, feature extraction, and model evaluation in building machine learning models for image classification tasks.
