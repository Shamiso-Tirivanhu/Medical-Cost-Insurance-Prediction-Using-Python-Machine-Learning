# Medical-Cost-Insurance-Prediction-Using-Python-Machine-Learning

## Table of Contents

1. [Overview](#overviw)

2. [Dataset](#dataset)

3. [Data Preprocessing](#data-progressing)

4. [Model Selection](#model-selection)

5. [Model Training](#model-training)

6. [Model Evaluation](#model-evaluation)

7. [Technologies Used](#technologies-used)

8. [How to Run the Project](#how-to-run-the-project)

9. [Future Improvements](#future-improvements)

## 1. Overview

This project aims to predict medical insurance costs based on various factors such as age, gender, BMI, number of children, smoking status, and region. By leveraging machine learning techniques, this model provides a reliable estimate of insurance charges, which can be useful for both insurance companies and individuals seeking to understand how different factors influence their insurance costs.

## 2. Dataset

The dataset used in this project consists of 1,338 entries with the following features:

- Age: Age of the individual

- Sex: Gender (Male/Female)

- BMI: Body Mass Index


[!image_alt](https://github.com/Shamiso-Tirivanhu/Medical-Cost-Insurance-Prediction-Using-Python-Machine-Learning/blob/372a222f60bdf1f98871417bd53749e49897dd08/BML%20Distribution%20screenshot.png)


- Children: Number of children covered by health insurance

[!image_alt]()


- Smoker: Whether the individual is a smoker or not

- Region: The individual's residential region

- Charges: The medical insurance cost (Target variable)

## 3. Data Preprocessing

To prepare the dataset for model training, the following preprocessing steps were performed:

- Checked for missing values (none found)

## Encoded categorical variables:

- Sex: Male (0), Female (1)

[!image_alt]()



- Smoker: Yes (0), No (1)

[!Imagee_alt]()



- Region: Southwest (0), Northwest (1), Southeast (2), Northeast (3)

[!image_alt]()



- Split the dataset into features (X) and target variable (Y)


- BML & Children were not encoded as they contained numerical values
  
Divided the dataset into training (80%) and testing (20%) subsets

## 4. Model Selection

A Linear Regression model was used for this predictive task. Linear regression was chosen due to its simplicity and interpretability when analyzing relationships between insurance charges and independent variables.

## 5. Model Training

The model was trained using the scikit-learn library, specifically:

- LinearRegression() from sklearn.linear_model

- Training the model on X_train and Y_train

## 6. Model Evaluation

The performance of the model was evaluated using the R-squared (R²) score:

- Training data R² Score: 0.7516

- Testing data R² Score: 0.7428

Since both scores are close, the model generalizes well without overfitting.

## 7. Technologies Used

Python

- Pandas for data manipulation

- NumPy for numerical operations

- Matplotlib & Seaborn for data visualization

- scikit-learn for model building and evaluation

## 8. How to Run the Project

Clone the repository:


| git clone https://github.com/your-username/medical-insurance-cost-prediction.git |
| ---------------------------------------------------------------------------------|

Navigate to the project folder:

| cd medical-insurance-cost-prediction |
| -------------------------------------|

Install required dependencies:

| pip install -r requirements.txt |
|---------------------------------|

Run the Jupyter Notebook or Python script:

jupyter notebook

OR

| python medical_insurance_prediction.py |
|----------------------------------------|

## 9. Future Improvements

- Implement additional machine learning models such as Decision Trees and Random Forests for comparison

- Hyperparameter tuning to improve model accuracy

- Deployment of the model using Flask or FastAPI for real-world use cases
