# RIFLE: Robust Inference from Low Order Marginals
RIFLE is a general framework for learning parameters of a statistical learning model on datasets containing a large amount of missing values. Supervised learning approaches, including linear regression, logistic regression, neural networks, and support vector machine (SVM) initially assume that the input data is complete, containing no missing values. However, availability of such clean datasets is scarce in many practical problems, especially in electronic medical records and clinical research. Integration of multiple datasets may also increase the size of data available to researchers as different organizations collect data from similar population in different areas. The obtained datasets can contain large blocks of missing values as they may not share exactly the same features (See below figure).

![Alt text](Merged_Datasets.png?raw=true "Title")
*Figure 1: Consider the problem of predicting the trait y from feature vector (x<sub>1</sub>, ..., x<sub>100</sub>). Suppose that we have access to three data sets: The first dataset includes the measurements of (x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>40</sub>, y) for n<sub>1</sub> individuals. The second data set collects data from another n<sub>2</sub> individuals by measuring (x<sub>30</sub>, \dots, x<sub>80</sub>) with no measurements of the target variable y in it; and the third data set contains the measurements from the variables (x<sub>70</sub>, ..., x<sub>100</sub>, y) for n<sub>3</sub> number of individuals; How one should learn the predictor y = δ (x<sub>1</sub>, ..., x<sub>100</sub>) from these three datasets?*

In practice, a pre-processing step is performed on a given dataset to remove or impute missing values. Removing the rows containing missing values is not an option when the percentage of missingness in a dataset is high, or the distribution of missing values is MNAR (Missing Not At Random). On the other hand, the state-of-the-art approaches for data
imputation are not robust to the large amount of missing values. Moreover, the error in the imputation phase can drastically affect the performance of the statistical model applied to the imputed dataset. 

An alternative approach for learning statistical models when the data is not complete is to learn the parameters of the model directly based on the available data. RIFLE is a distributionally robust optimization framework for the direct (without imputation) inference of a target variable based on a set of features containing missing values. The proposed framework does not require the data to be imputed as a pre-processing stage, although, it can be used as a pre-processing tool for imputing data as well. The main idea of RIFLE is to estimate appropriate confidence intervals for the first and second-order moments of the data by bootstrapping on the available data matrix entries and then finding the optimal parameters of the statistical model for the worst-case distribution with the low-order moments (mean and variance) within the estimated confidence intervals. 

Mathematically speaking, RIFLE solves the following min-max problem:

<div align='center'> 
<img src="general_framework.jpg" width="350" align='center'>
</div>

## RIFLE as an Imputation Tool
To run the robust linear regression imputer on a given dataset containing missing values, execute the following command on a terminal (linux) or command line (windows):

```
python run.py input_file.csv output_file.csv 
```


## ADMM Version:
To accelerate the convergence and considering the PSD condition, we apply ADMM schema to the robust linear regression problem in the presence of missing values. Algorithm 6 in the paper is implemented and can be used via the following command:

```
python ADMM.py input_file.csv output_file.csv 
```


## Robust LDA and RIFLE for Classification Tasks
Two algorithms are proposed for handling classification task via the framework described above: Thresholding RIFLE, Robust LDA. To run Robust LDA, use the following command:

```
python RobustLDA.py input_file.csv output_file.csv 
```

To use the thresholding algorithm, run the following command:

```
python Thresholding.py input_file.csv output_file.csv 
```

## Performance of RIFLE
In the following figure, we compare the performance of RIFLE with MissForest and MICE (two popular imputers) in terms of Normalized Root Mean-Squared Error (NMRSE). We count the number of features each one of these methods have a better performance for imputing them. RIFLE has a better performance compared to two other methods, especially when the proportion of missing values is higher.

![Alt text](Counts.png?raw=true "Counts")
*Figure 2: Performance Comparison of RIFLE, MICE and MissForest on four UCI datasets: [Parkinson](https://archive.ics.uci.edu/ml/datasets/parkinsons), [Spam](https://archive.ics.uci.edu/ml/datasets/spambase), [Wave Energy Converter](https://archive.ics.uci.edu/ml/datasets/Wave+Energy+Converters), and [Breast Cancer](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Prognostic)). For each dataset we count the number of features that each method has a better performance over other two methods.*

## Sensitivity of RIFLE to Proportion of Missing Values and Number of Samples:
In the next figure, we can observe that performance of RIFLE is less sensitive to the proportion of missing values compared to other state-of-the-art approaches including MIDA, KNN Imputer, Amelia, MICE, MissForest and Mean Imputer.

![Alt text](Sensitivity_Missing_Value_Proportion.png?raw=true "S_MP")
*Figure 3: Sensitivity of RIFLE, MissForest, Mice, Amelia, KNN Imputer, MIDA and Mean Imputer to the percentage of missing values On [Drive Dataset](https://archive.ics.uci.edu/ml/datasets/dataset+for+sensorless+drive+diagnosis). Increasing the percentage of missing value entries degrades the performance of the benchmarks as compared to RIFLE. KNN-imputer implementation cannot be executed on datasets containing  80% (or more) missing entries. Moreover, Amelia and MIDA do not converge to a solution when the percentage of missing value entries is higher than 70%.*

Moreover, we examine the sensitivity of RIFLE and other approaches to the number of samples. As it can be observed, when the number of samples is limited (relative to the number of features) RIFLE shows a better performance compared to other methods. When we increase the number of samples, still RIFLE has a good performance comparable to the non-linear imputer MissForest.

![Alt text](Sensitivity_Sample.png?raw=true "S_N")
*Figure 4: Sensitivity of RIFLE, MissForest, Mice, Amelia, Mean Imputer, KNN Imputer, and MIDA to the number of samples for the imputations of [Drive dataset](https://archive.ics.uci.edu/ml/datasets/dataset+for+sensorless+drive+diagnosis) containing 40% of MCAR missing values. When the number of samples is limited, RIFLE demonstrates better performance compared to other methods and its performance is very close to the non-linear imputer MissForest for larger samples.*
