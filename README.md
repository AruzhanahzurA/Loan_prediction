# Loan Approval Predictor

This project aims to predict loan approval or disapproval using regression models with Convolutional Neural Networks (CNN). The dataset used for this project contains various features related to loan applications, and the goal is to predict whether a loan will be approved or not based on these features.

## Requirements

- Python 3.x
- pandas
- PyTorch
- scikit-learn
- NumPy

## Installation

1. Clone the repository:

```
git clone https://github.com/yourusername/loan-approval-predictor.git
```

2. Install the required dependencies:

```
pip install pandas torch scikit-learn numpy
```

## Usage

1. Navigate to the project directory:

```
cd loan-approval-predictor
```

2. Run the EDA (Exploratory Data Analysis) script to explore the dataset:

```
python eda.py
```

3. Once you've gained insights from the EDA, you can proceed to train and evaluate the regression models. There are two regression models implemented in this project:

   - Simple CNN model
   - CNN model with Dropout, Batchnorm, and max pooling for optimization

4. To train and evaluate the models, run the following script:

```
python train_and_evaluate.py
```

This script will train both models and evaluate their performance using Mean Squared Error (MSE) as the evaluation metric.

## Model Architecture

### Simple CNN Model

The simple CNN model has a basic architecture comprising convolutional layers followed by fully connected layers. It is designed to learn features directly from the input data without any additional optimizations.

### Optimized CNN Model

The optimized CNN model incorporates Dropout, Batch Normalization, and max pooling layers for better generalization and faster convergence during training. Dropout helps prevent overfitting by randomly dropping neurons during training, while Batch Normalization ensures stable training by normalizing the activations. Max pooling reduces the spatial dimensions of the feature maps, helping to reduce computational complexity.

## Results

After training both models, the Mean Squared Error (MSE) values are compared to assess their performance. Lower MSE values indicate better predictive performance.

## Contributors

- Aruzhan Oshakbayeva
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
