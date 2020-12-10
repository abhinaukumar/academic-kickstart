---
title: "Analyzing Model Tradeoffs in Predicting Length of Stay (LOS) in eICU Patients"
subtitle: ""
summary: ""
authors: ["admin"]
tags: ["Academic"]
categories: ["Project"]
date: 2020-12-10
lastmod: 2020-12-10
featured: false
draft: true
math: true
# Featured image
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: 'Source: [**TwistedSifter**](https://twistedsifter.com/2015/04/the-banyan-tree/)'
  focal_point: ""
  preview_only: false

projects: []
---

# Introduction and Background
Patients with prolonged stay, easily account for high resource consumption [1] and are thus a potential loss of revenue from a hospital management perspective. This is especially important with Intensive Care Unit (ICU) and electronic ICU patients due to their intensive need for care resources such as nurses, drugs, surgeons, X-ray machines etc., for continuous monitoring.  Data mining and predicting/projecting the remaining length of stay (rLOS) for a patient will help allocate resources efficiently and provide timely services to patients.  

The challenge for developing such a predictive model arises from the incomplete understanding of what complex clinical factors are involved and their interactions or relationships that lead to a specific LOS. This is a potential applicability for neural networks, which are considered universal function approximators. Since we are handling time-series data, in the form of patient records, we use RNNs (and related architectures) to predict rLOS.  

In addition to having a predictive model, an accurate understanding of the factors that are closely associated with LOS can allow for efficient hospital resource management, savings to a healthcare system and better patient care by reducing readmissions. In this regard, neural networks fall short, as they are not interpretable. This “black box” behavior of neural networks leads us to the first tradeoff being considered in this blog – performance vs. interpretability. 

In addition, the data set is sensitive for the patients and it deserves to be protected. All countries have laws to protect privacy leading to difficulties in the data collection phase. Many hospitals may refuse to share healthcare data without any Health Insurance Portability and Accountability Act (HIPAA) agreements or data governance in place. Therefore, we also investigated a federated Learning framework to address such privacy issues.  This distributed learning framework leads us to the second tradeoff being considered – performance vs. privacy security. 

# Database
The eICU Collaborative Research Database [2] is a multi-center intensive care unit database with high granularity data for over 200,000 admissions to ICUs monitored by eICU programs across the United States. The eICU database comprises 200,859 patient unit encounters for 139,367 unique patients admitted between 2014 and 2015 to 208 hospitals located throughout the US. 

The database is de-identified, and includes vital sign measurements, care plan documentation, severity of illness measures, diagnosis information, treatment information, and more. Data are publicly available after registration, including completion of a training course in research with human subjects and signing of a data use agreement mandating responsible handling of the data and adhering to the principle of collaborative research.

# Data preprocessing 

To clean and preprocess the data, we use the same procedure used in [benchmarking ML paper]. To clean the data, we use several exclusion criteria.  

1. We first only consider patients who are “adults”, i.e., patients who are between 18 and 89 years old.  

2. We only consider patients who have one ICU stay on record, because we are interested in ICU length of stay  

3. We exclude patients who have less than 15 or more than 200 records, to keep the length of inputs significant, but not excessive.  

4. We exclude patients whose gender or hospital/ICU discharge status (alive vs. expired) are unknown.  

5. Finally, we only include records which were recorded between admission and discharge, which corresponds to having positive offset and rLOS. 

As a side note, we use the term “record” to refer to a set of 21 features collected at a time step, which is called an “offset”. After cleaning the data, we bin the offsets into hours. Within each hour, we consider the first record, and we impute its missing values with the average value over the hour. Often, features are missing over the entire hour, so we will need to impute values again at the hour level. We will discuss this “sparsity” of the database in further detail below. To handle these missing values, we follow the literature and use “typical” values of each feature. 

After cleaning and binning the data, we have a total of 74686 patients, and a total of just over 3.1 million records. We then split the data randomly to use 80% of the patients for training and 20% for testing. It is important to note that we do not split the data of any patient. 

Because we have several categorical variables, and some can take several values (GCS Total can take 15 values, admission diagnosis can take over 100), we cannot use the typical one-hot encoding to represent these variables without adding a large number of features. 

Instead, we handle categorical variables by a method called “Target Encoding”. In this method, each value of a categorical variable is replaced by the average value of the target variable corresponding to all examples which have this value. More formally, we replace value $v$ of a categorical variable $X_j$ by 

$E[Y | X_j = v]$ 

In some cases, we may see new values of the categorical variable in the test data that we have not encountered. In this case, we simply use the average value of the target variable instead of such a conditional average. 

In our code, we created a class TargetEncoder to carry this out, which adheres to the Scikit-Learn API format. So, in practice, target encoding variables in a dataset is as simple as 

```
encoder = TargetEncoder()
encoder.fit(self.X, self.y, ['apacheadmissiondx', 'ethnicity', 'gender', 'GCS Total', 'Eyes', 'Motor', 'Verbal'])
self.X = encoder.transform(self.X)
```

Finally, we applied a “Min-Max scaler”, which maps the values of each feature to the range `[-1, 1]`. This kind of scaling helps improve the convergence of many machine learning models, especially those that use gradient-based optimization methods.

# Understanding the Data
Before we begin training the models, it is useful to understand the nature of the database. As we have mentioned above, a notable feature of the database is its sparsity. This is to be expected, because not every feature is recorded at each time step. Often the frequency at which, say, oxygen saturation is measured may be quite higher than that at which blood glucose level or pH are measured. 

In the following table, we show the percentage of missing values of each feature. Features which have no missing values have been omitted. 

| Feature | Percentage missing|
| :-------: | :--------: |
| Eyes         | 15.451178 |
| GCS Total    | 15.45 |
| Heart Rate   | 15.451178 |
| Diastolic BP | 15.451178 |
| Systolic BP  | 15.451178 |
| MAP (mmHg)   | 15.451178 |
| Motor        | 15.451178 |
| O2 Sat.      | 15.451178 |
| Resp. Rate   | 15.451178 |
| Temp. (C)    | 15.451178 |
| Verbal       | 15.451178 |
| FiO2         | 98.319722 |
| Glucose      | 71.088619 |
| pH           | 97.839852 |
| Height       | 1.209786  |
| Weight       | 2.290426  |

In this plot, we plot the histogram of the number of missing features in each record. 

[Missing features plot] 

From these two plots, it’s clear that the dataset is quite sparse, with almost no records having all features. Also, the features that are missing most often are the pH, glucose and FiO2, which are likely measured only in specific patients, when required. So, we don’t expect them to be missing completely at random (MCAR), and we cannot omit any records or features. As a result, imputation becomes a very important data processing step. Also, because the data is time-series, imputing using the mean or median, which are common choices, can give undue weight to patients who have more records. So, to avoid these complex considerations, using “typical” values is a good practical choice. 

After target encoding, we have converted all variables into numerical variables, so we can visualize their dependencies using the correlation plot below. 

[correlation plot] 

Looking at the plot, it is not surprising that the height and the weight are highly correlated. Another interesting observation is that the features obtained from the Glasgow Coma Score (GCS) suite - GCS Total, Eyes, Verbal, Motor are all highly correlated. It is important to note that these variables are originally categorical variables, so these correlations are over a small set of unique values. Other than these, the variables are weakly correlated, so we can expect that they provide diverse information about the LOS/rLOS targets. 

To round out the data visualization exercise, we show the histogram of lengths of stays below. Since most values lie in the range 0-20, we restrict the histogram to this range. 

[los_histogram.png] 

In the histogram, we can notice several local modes corresponding to integer values of LOS. We think this shows a natural bias of hospitals to discharge patients in whole numbers of days, perhaps due to administrative convenience. 

Beyond these summaries, we want to look a little deeper into the data. Specifically, we are interested in finding the effect of gender and ethnicity on the LOS and the discharge status - alive vs. expired. Such an analysis will help us uncover biases (conscious or otherwise) in the healthcare system which need to be addressed. 

To find the effect of gender on the LOS, we will compare the average LOS between male and female patients. Visually, we can see the values in the form of the following box plot. 

[gender_los_boxplot.png] 

Analytically, because we want to compare the means of two groups, we will use a statistical test called the “independent t-test”. A good resource to learn about the t-test is [https://conjointly.com/kb/statistical-student-t-test/ ], and we will skip the details here. The t-test returned a “p-value” of about 10^{-3}. Such a low p-value means that the difference in means is “significant”, i.e., the mean LOS in ICUs depends on gender. We see from the box plot that men have a larger average LOS, and are also more likely to have abnormally high LOS, when compared to women.  

We can ask the same question of ethnicity. The dataset contains patients of the following ethnicities - Asian, African American, Caucasian, Hispanic, Native American, Other/Unknown. Once again, we can visualize the data using a box plot. 

[ethnicity_los_boxplot.png] 

But now that we have more than two groups, we use an F-test (also called a one-way analysis of variance), which compares the means of several groups. The p-value returned by the F-test was 10^{-17}, so once again, the test is “significant”, and the mean LOS depends on ethnicity. 

Investigating the effect of gender on discharge status, we can tabulate the number of men and women who were discharged alive or expired. Such a table is called a “contingency table”. 