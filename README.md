# Clothing Reommendation Model
This is my data science project on image recommendation for e-commerce websit here i made recommendation using [pca + kmeans](recommendation-system-using-pca-and-kmeans.ipynb), [pca + resnet](recommendation-system-using-pca-and-resnet.ipynb) out of which pca+ resent model give good result. Find more detail infomation about model in [Models](#models) and other procedure below. 

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Data](#data)
8. [Models](#models)
9. [Evaluation](#evaluation)
10. [Contributing](#contributing)
11. [License](#license)
12. [Acknowledgments](#acknowledgments)
13. [Timeline](#timeline)


## Introduction

The recommender system is the most profound research area for e-commerce product recommendations. Currently, many e-commerce platforms use a text-based product search, which has limitations to fetch the most similar products. An image-based similarity search for recommendations had considerable gains in popularity for many areas, especially for the e-commerce platforms giving a better visual search experience by the users.

Checkout my kaggle notebook 
* [pca + resnet](https://www.kaggle.com/code/shreyashchacharkar/recommendation-system-using-pca-and-resnet)
* [pca + kmeans](https://www.kaggle.com/code/shreyashchacharkar/recommendation-system-using-pca-and-kmeans)

## Features

List the key features or functionalities of the project.

## Requirements
All software/ liberies mention in [requirements.txt](requirements.txt) and you  can use Kaggle Notebooks GPU for model training or prediction to make process fast. Can found readymade file on kaggle notebook output to download either directly download or from kaggle kenerl command

## Installation

Provide step-by-step instructions on how to install the project and its dependencies. This may include setting up a virtual environment, installing libraries, or configuring specific settings.

```bash
git clone https://github.com/ShreyashChacharkar/Clothing_Model.git
cd Clothing_Model
pip install -r requirements.txt
```

## Usage

Explain how to use the project. Include examples of command-line commands or code snippets.

```bash
# Example usage
python main.py --input data/input.csv --output results/output.csv
```

## Project Structure

Explain the organization of the project's directories and files. Highlight the purpose of important files or folders.

```
project-root/
|-- data/
|   |-- raw/
|   |-- processed/
|-- notebooks/
|-- src/
|-- README.md
|-- requirements.txt
```

## Data
* About Dataset
* Context
Thr growing e-commerce industry presents us with a large dataset waiting to be scraped and researched upon. In addition to professionally shot high resolution product images, we also have multiple label attributes describing the product which was manually entered while cataloging. To add to this, we also have descriptive text that comments on the product characteristics.

* Content
Each product is identified by an ID like 42431. You will find a map to all the products in styles.csv. From here, you can fetch the image for this product from images/42431.jpg. To get started easily, we also have exposed some of the key product categories and it's display name in styles.csv.

## Models

1. **Image Recommendation using Convolutional Neural Network (CNN):**
Here I Use the product images in the "images" directory.Extract the product categories from the "masterCategory" column in the "styles.csv" file and Build a CNN model pretrained model ResNet50 to recommend images into different master categories base dimage click. I Trained the model using the images and their corresponding master categories.


2. **NLP-based Recommendation using Product Descriptions:**
I Extracted product descriptions from the "styles" directory, for example, from "styles/42431.json." Used Natural Language Processing(NLP) techniques to process and vectorize the product descriptions.Trained a classifier, such as a text classification model, to predict the masterCategory based on product descriptions.


3. **Multi-Label Classification:**
Extend your classification task to predict other category labels in addition to the masterCategory. Modify your model to handle multi-label classification, where each product can belong to multiple categories. Update your dataset and model accordingly.

## Evaluation
Explain how the project's performance is evaluated. Include metrics or criteria used to measure success.

## License
This project can be used for reserach or learning purpose not commercial purpose

## Timeline
**How to set this project timeline?**

| Days                 | Tasks    | Description|
|-----------------------------------------|----------|----------|
|Day 1 | Topic introdcution  |read reserch paper, youtube, article|
|Day 1 to 2 | Data wrangling | Data Preprocessing(pyredr lib.), Cleaing, ETL activities, Data analysis, Data visualistion(matplotlib, seaborn)|
|Day 3 |Model Trainaing | Training with data(sklearn, tensorflow, classification algorithm), Feature extracting, Hyperparammeter Tuning|
|Day 4 | Communication Result | Explainable AI Shaply and Lime, Real time fault analysis|
|Day 5 to 8 | Web dashboard | Web dashboard(basline with templates, style, app.py, utlis.py) |
|Day 9 | Deplying on cloud | Deploying selected ML model on GCP and AWS, connect apis|

## Acknowledgments 
* [fashion recommender project by campusx](https://github.com/campusx-official/fashion-recommender-system)
* Probabilistic Unsupervised Machine Learning Approach for a Similar Image Recommender System for E-Commerce Ssvr Kumar Addagarla and Anthoniraj Amalanathan *


## Other Clothing dataset
* [People Clothing Segmentation](https://www.kaggle.com/datasets/rajkumarl/people-clothing-segmentation)  
* [Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews)
* [Nlp with women's clothing](https://www.kaggle.com/code/granjithkumar/nlp-with-women-clothing-reviews)
* [Fashion Clothing Products Dataset](https://www.kaggle.com/datasets/shivamb/fashion-clothing-products-catalog)
* [SIze recommendation](https://www.kaggle.com/datasets/rmisra/clothing-fit-dataset-for-size-recommendation) json format
* [Clothing dataset (full, high resolution)](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full)  5000 images and csv files
* [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data)
* [Colthing, Models, Posing](https://www.kaggle.com/datasets/dqmonn/zalando-store-crawl)
* [Clothing Co-Parsing Dataset Clothes Segmentation Dataset](https://www.kaggle.com/datasets/balraj98/clothing-coparsing-dataset) images and csv files
* [Ajio Fashion Clothing](https://www.kaggle.com/datasets/manishmathias/ajio-clothing-fashion)
  
