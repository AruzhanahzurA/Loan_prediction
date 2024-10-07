# Loan Approval Prediction - Binary Classification

This project aims to predict the loan approval status (approved or not) based on several applicant features using Logistic Regression and Neural Network. The dataset includes features such as applicant income, coapplicant income, loan amount, credit history, and more.

## Content Plan
- [Dataset](#dataset)
- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [How to Use](#how-to-use)
- [Results](#results)
- [Future Work](#future-work)
- [License](#license)

## Dataset

The dataset consists of 381 samples with the following features:

| Feature            | Type    | Description                                              |
|--------------------|---------|----------------------------------------------------------|
| Loan_ID            | Object  | Unique identifier for each loan                          |
| Gender             | Object  | Gender of the applicant (Male, Female)                   |
| Married            | Object  | Marital status of the applicant                          |
| Dependents         | Object  | Number of dependents (0, 1, 2, 3+)                       |
| Education          | Object  | Education level of the applicant (Graduate, Not Graduate)|
| Self_Employed      | Object  | Whether the applicant is self-employed (Yes, No)         |
| ApplicantIncome    | Float   | Monthly income of the applicant                          |
| CoapplicantIncome  | Float   | Monthly income of the co-applicant                       |
| LoanAmount         | Float   | Loan amount requested (in thousands)                     |
| Loan_Amount_Term   | Float   | Term of the loan (in months)                             |
| Credit_History     | Float   | Credit history (1 = Good, 0 = Bad)                       |
| Property_Area      | Object  | Property location (Urban, Rural, Semiurban)              |

## Project Overview

The goal is to develop a binary classification model that predicts whether a loan will be approved (Loan_Status: 0 = No, 1 = Yes) based on the features above. Both logistic regression and neural network models have been implemented and evaluated.

### Steps Followed:
1. **Data Cleaning and Visualization:** 
    - Handled outliers in columns like ApplicantIncome, LoanAmount, and CoapplicantIncome.
    - Applied PowerTransformer to numeric columns to stabilize variance and minimize skewness.
2. **Feature Engineering:**
    - Converted categorical features like Gender, Education, and Property_Area into numerical values using encoding techniques.
    - Handled missing values using imputation techniques.
3. **Modeling:**
    - Built a Logistic Regression model.
    - Developed a Neural Network with Batch Normalization and Dropout layers to prevent overfitting.
4. **Evaluation:**
    - Logistic Regression Model achieved approximately 82% accuracy on both training and test sets accuracy while Neural Network achieved slightly higher accuracy of 85% on train set and 83% on test set.
   
## Dependencies

- Python 3.x
- pandas, numpy
- scikit-learn
- pytorch
- matplotlib/plotly for visualizations

## How to Use

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/loan-approval-prediction.git
    ```
2. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the notebook or script to train the models and evaluate the results:
    ```bash
    python loan_approval.py
    ```

## Results

The models were evaluated using accuracy, precision, recall, and F1 score. The neural network architecture includes batch normalization and dropout layers to reduce overfitting.

| Model               | Train Accuracy | Test Accuracy |
|---------------------|----------------|---------------|
| Logistic Regression | 81.8%          | 81.5%         |
| Neural Network      | 84.8%          | 82.9%         |

## Future Work

- Fine-tuning the hyperparameters of the neural network to improve test accuracy.
- Implementing additional feature engineering techniques such as polynomial features.
- Trying other models such as Random Forest or XGBoost for comparison.

## License

This project is licensed under the MIT License.
