# 🦾 MachineLearning

Machine Learning (COMP130172) course of Fudan University, Fall 2023. Instructed by Professor [Mingmin Chi](https://datascience.fudan.edu.cn/e1/6f/c13398a123247/page.htm)

## Assignment 1: Wine Quality Classification using K-Nearest Neighbours

- Using the k-NN for classification on wine quality dataset available at [Kaggle](https://www.kaggle.com/shelvigarg/wine-quality-dataset/)

## Assignment 2: Regression for Housing Price Predictions

- Design different regression models to predict housing prices based on Boston price dataset, which can be downloaded directly on [Kaggle](https://www.kaggle.com/datasets/vikrishnan/boston-house-prices) or loaded on [scikit-learn](https://scikit-learn.org/1.0/modules/generated/sklearn.datasets.load_boston.html)
- You should at least try Ridge Regression & Lasso regression

## Assignment 3: Support Vector Machines

- Based on [MNIST dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data), use SVM classifier to classify the handwritten digits.
- Requirements:
  - Try different kernels, and find the best hyperparameter settings (including kernel parameters and the regularization parameter) for each of the kernel types
  - Visualize SVM boundary
  - Try other methods to classify MNIST dataset, such as least squares with regularization / Fisher discriminant analysis (with kernels) / Perceptron (with kernels) / logistic regression / MLP-NN with two different error functions.

## Assignment 4: Support Vector Regression
- Using support vector regression to predict housing prices. The data is available on [Kaggle](https://www.kaggle.com/vikrishnan/boston-house-prices)

## Assignment 5: MultiLayer Perceptron

- Use Torchvision API to get the [MNIST Dataset](http://yann.lecun.com/exdb/mnist/http://yann.lecun.com/exdb/mnist/)(some handwriting figures), convert them from images to vector tensors. Then, try to build `nn.Linear()` layers, (which equals to W & b). Try to feed the vector tensors to Linear Layer and Activation Functions, to get the predict label. Use the loss function to compute the difference between predict label and the ground truth label, use `loss.backward()` function to get the gradient of each Linear Layer's parameters. Finally, use the optimizers (SGD or others) to make the model converge. 
