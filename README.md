# Age Gender Prediction 

## Python Modules 
* Pandas
* Numpy
* Tensorflow
* Scikit Learn
* Seaborn
* Matplotlib

## Data  
Images were acquired from [UTKFace dataset](https://susanqq.github.io/UTKFace/). Dataset contains around 27,000 images of people with different gender, ethnicity (race) and age. Data has been aready preprocesed into cvs format and this project starts from investigating the preprocessed data. Each image is already reshaped 48 by 48 pixels size. 

<img src="images/random_images_1.png" border="2">

## Exploratory Data Analysis
There are 3 different targets to classify for this dataset, which are described below. 

### Age Distribution
Dataset has long age span ranging from 0 to 116. DIstribution is demostrated below. 
![Age EDA](images/Ages_EDA.png)

### Gender Distribution
Number of male individuals is little higher female, however difference can be neglected for this analysis. Also it should be noted that in original data Female category is mapped as 1, and Male as 0. Some simple changes were done using pandas mapping feature. 
![Gender EDA](images/Gender_EDA.png)


### Ethnicity Distribution
Ethnicity category is mapped as an integer from 0 to 4, denoting  White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern) respectively. 
![Ethnicity EDA](images/Ethnicity_EDA.png)

## None Deep Learning Models


### PCA Decomposition
PCA decomposition was performed on 90 % of variance. Which came out 73 componets.
![Scree Plot](images/Scree.png)
![Gender decomp Plot](images/Decompose_gender.png)
![Ethnicity decomp Plot](images/Decompose_ethno.png)
![Age decomp Plot](images/Decompose_age.png)
From decomposistion plot for each target is it obvious that there is no any visible pattern.

### Comparison of Classification Models
Support Vector Machine Classifier, KNeighborsClassifier were used to classify Gender and Ethnicity, but the accuracy was not high. As an input Decomposed Matrix of image data was used. 

Afterwards ensemble Voting Classifier was used to combine SVM and KNN. Again low accuracy was observed. Out three methods KNN demostrated higher accuracy. Comparison of outputs is demonstrated below. 

![Accurac Comparison none deep learning](images/Accuracy_comparision_.png)

## Deep Learning Models
As it was expected none Deep Learning models did not demostrate feasible results. So Dl models were used for classification

### Gender CNN Prediction Model
1. Input data is passed through 4 Convolution Layers with Max Pooling between each
2. Data is flattened
3. Flattened data is passed through 2 dense layers
4. Model is compiled with adam optimizer and binary crossentropy loss function

![Accuracy CNN gender](images/gender_cnn.png)

After 10 epochs accuracy of model is around 90%, which is higher than for non-deep learning models.
![Accuracy CNN gender](images/Accuracy_cnn_gender.png)

### Ethnicity Prediction Model
Two different models have been used for ethnicity predictions
1. Input data is passed through 5 Convolution Layers with 3 Max Pooling between 
2. Data is  flattened
3. Flattened data is passed through 2 dense layers
4. Model is compiled with rmsprop optimizer and sparse categorical crossentropy loss function

![Ethnicity CNN 1](images/Ethno_cnn_1.png)

After 20 epochs accuracy of model is around 70%, which is higher than for non-deep learning models.
![Accuracy CNN Ethno 1](images/Accuracy_cnn_ethno1.png)

1. Input data is passed through 4 Convolution Layers with Max Pooling between 
2. Data is  flattened
3. Flattened data is passed through 2 dense layers
4. Model is compiled with adam optimizer and sparse categorical crossentropy loss function

![Ethnicity CNN 2](images/Ethno_cnn_2.png)

It appears that model with less parameters performed better and test accuracy is arround 75%. 
![Accuracy CNN Ethno 2](images/Accuracy_cnn_ethno2.png)

### Age Prediction Model