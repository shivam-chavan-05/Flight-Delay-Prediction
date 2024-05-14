# Enhancing Traveler Experience through flight delay prediction

### Motivation
Flight delays present substantial issues for
aviation stakeholders, resulting in interruptions,
decreased operational efficiency and financial
losses. These delays have an influence on
airlines, airports, passengers' timetables,
resource allocation and consumer satisfaction.
Predicting flight delays accurately is critical for
optimizing operations, managing resources
effectively and improving the overall travel
experience. By tackling this issue, stakeholders
can reduce the negative effects of delays while
also improving the efficiency and dependability
of air travel.

### Objective
The objective is to develop machine learning
models which will leverage a variety of input
features such as airline data, airport congestion
metrics and flight schedules to achieve its
predictive accuracy.
Hypothesis
1. Null Hypothesis (H0): There is no significant
relationship between the input features and flight
delays.
2. Alternative Hypothesis (H1): There is a
significant relationship between the input
features & flight delays.

### Data Set Specifications
Size: Our dataset contains a total of 3 million
data points, with each data point having 32
features.
Rows & Columns: The dataset has 3,000,000
rows and 32 columns.
Years: 2020- 2023
Target Variable: Flight Status
Data Types: (Before data pre-processing)
datetime64, object, float64 & int64

![Image1](https://github.com/shivam-chavan-05/Flight-Delay-Prediction/assets/144063863/64a2bfbf-d324-4a67-9b4c-1af10bb23b3b)

### Data Preprocessing Steps
#### Eliminating Duplicates
Duplicate data is deleted as it could
cause overfitting.
#### Handling Missing values
A common criterion for replacing
missing values with means is between 5
and 10% of the entire dataset size.
#### Handling Null & Nan Values
In our dataset, columns such as
[DEP_DELAY, TAXI_OUT,
WHEELS_OFF ,WHEELS_ON, etc]
contained Null (or NA) / NaN (Not a
Number) values. These are special
examples of missing values that must be
handled separately. To address this issue
we replaced it with mean values.
Not all machine learning algorithms
support Null/Nan values. While
evaluating our models, we dropped the
records with those values.
#### Handling encoding Issues
This step addressed data encoding
discrepancies and difficulties, such as
different character encodings and data
formats. We used one-hot encoding to
convert categorical variables into
numerical format, making them
compatible with machine learning
approaches. It's also worth mentioning
that correlation matrices cannot be used
directly with categorical data.
#### Standardizing Data Format
Data may exist in a variety of forms or
units, which can have an impact on
analysis and modeling results.
Standardizing data format entails
transforming it to a standard format or
unit, which improves comparison and
analysis.
As part of this process, we converted
FL_DATE to a more appropriate format,
ensuring coherence and compatibility
across multiple features.
#### Performing Feature Engineering
Feature engineering involves creating
new features or modifying existing ones
to improve machine learning model
performance.
We created a feature Flight Status using
binary classification which form two
classes 0 and 1.
1: Delayed and 0: On time

### Correlation Matrix

![Image2](https://github.com/shivam-chavan-05/Flight-Delay-Prediction/assets/144063863/1e550a6b-07a8-4058-9597-830d3a46aed2)

● The graphic depicts a heatmap of the
correlation coefficients between numeric
variables in the dataset.
● The diagonal elements, which represent
a variable's correlation with itself, are
always 1.0 and shown in dark blue.
● The heatmap provides insights into
variable correlations, which helps to
uncover multicollinearity concerns and
guides feature selection and engineering
procedures.

### Exploratory Data Analysis
EDA is a crucial stage in the data analysis
process since it offers insightful information
that helps with modeling, analysis,
decision-making, etc. It aids in
comprehending the data, seeing trends and
abnormalities, evaluating the quality of the
data, directing feature engineering and
selection, validating models and effectively
presenting findings.
• In the first barchart Frontier Airlines has
the highest average arrival delay of around 11
minutes.
• In the second bar chart American
Airlines has the highest total delayed
minutes, indicating a significant overall
impact of delays on their operations.

![Image3](https://github.com/shivam-chavan-05/Flight-Delay-Prediction/assets/144063863/e75729bf-e1d7-4e24-b6a6-c9ce875f0095)

![Image4](https://github.com/shivam-chavan-05/Flight-Delay-Prediction/assets/144063863/be17b415-a060-49d6-b60f-e36af1c28431)

● CRS_DEP_TIME: It shows a
multimodal distribution with distinct
peaks.
● DEP_DELAY: It shows a right-skewed
distribution, with most flights having
short delays and others having
significant delays.
● TAXI_OUT: It shows a distribution that
is right-skewed, with most flights
having shorter taxi-out times and a small
percentage having longer ones.
● WHEELS_OFF: It indicates different
takeoff times or schedules with a
multimodal pattern similar to
CRS_DEP_TIME.

### Baseline model : Decision Tree
The decision tree serves as our baseline model.
To establish a benchmark for performance
evaluation, we generated baseline metrics such as
accuracy, precision, recall, and F1-score. These
metrics provide information about the decision
tree's ability to make accurate predictions across
different categories.

![Image5](https://github.com/shivam-chavan-05/Flight-Delay-Prediction/assets/144063863/e0193d5e-533e-4505-ba57-f6bd1fa8fb2d)

● The bar chart shows a small decline in
accuracy from the training set (84.25%) to
the test set (84.08%), indicating the model's
capacity to generalize on previously
encountered data.
● The model outperforms random guessing, as
seen by its curve being high above the
diagonal.
● The decision tree baseline model does
reasonably well on this binary classification
job, with an overall test accuracy of
approximately 82%.

### Model Selection
Evaluation Criteria
1. Accuracy: It is the fraction of accurately
predicted instances in a dataset.
2. Precision: It is the ratio of true positive
predictions to all positive predictions made
by the model.
3. Recall: It is the percentage of true positive
predictions among all real positive cases in
the dataset.
4. F1 Score: It is the harmonic average of
precision & recall, resulting in a single term
that balances both.
5. ROC Curves: ROC (Receiver Operating
Characteristic) curves are graphical
representations of a binary classification
model's performance at different thresholds.
6. AUC ROC Curve: The AUC (Area Under
the Curve) of the ROC curve assesses the
model's ability to distinguish between
positive and negative instances at all
thresholds.

### Model Evaluation
#### Adaboost
It is an ensemble learning method that
combines multiple weak learners, often
decision trees, to create a strong learner. It
iteratively adjusts weights to focus on
misclassified instances, effectively
improving classification accuracy. It was
selected as our initial model due to its
prowess in handling unbalanced data, a
prevalent characteristic in our flight delay
detection task where the number of delayed
flights is substantially lower compared to
non-delayed ones.

As anticipated, Adaboost exhibited strong
performance, achieving an accuracy of
approximately 87.8% and an F1-score of
around 81.3% on both the training and
testing sets. These results underscored
Adaboost's proficiency in accurately
classifying flight delays while navigating the
challenges posed by imbalanced data
distributions.

#### Gradient Boost
Following Adaboost's commendable
performance, our exploration led us to
Gradient Boosted Trees (GBT), renowned
for its capacity to handle complex
relationships within the data and its
superior predictive accuracy. GBT
operates by sequentially fitting decision
trees to the residuals of the preceding
trees, thereby capturing intricate patterns
and relationships present in the data.

It achieves an accuracy of approximately 90.4%
& an F1-score of around 85.2% on the training
set, and maintaining its excellence with an
accuracy of about 90.3% and an F1-score of
approximately 85.1% on the testing set.

#### XG Boost
It improves standard gradient boosting by
adding various new features and improvements,
making it extremely efficient and effective for
both regression and classification problems.

### Challenges
Processing large datasets takes a significant
amount of time and resources due to their
size and complexity, especially when
dealing with intricate, nonlinear interactions
that make correct modeling difficult.
