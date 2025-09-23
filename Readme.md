# Purwadhika JCDSOL19 - Final Project - Kelompok Alpha
- Hendra Ronaldi
- Dhiya Ilham Trihatmaja

# 🛒 Customer Churn Prediction

- [Streamlit](https://ecommercecustomerchurn-alpha-jcdsol19.streamlit.app/)
- [Dashboard](https://lookerstudio.google.com/reporting/10509b68-b21c-4cd9-afe0-e264abbb2272/page/NO6YF)

![](./assets/dashboard.png)

## 📌 Business Problem Statement
The Kaggle dataset represents a **general e-commerce retail model**, where churn means customers became inactive.  

In retail e-commerce, annual churn rates of **60–80%** are common  
([Ecommerce Fastlane](https://ecommercefastlane.com/ecommerce-churn-rates-measure-and-reduce-lost-customers-and-revenue/?utm_source=chatgpt.com);  
[Sobot.io](https://www.sobot.io/article/average-churn-rate-for-ecommerce-stores-trends-2024-2025/?utm_source=chatgpt.com)).  

For this project, we **assume the dataset reflects one month of customer activity**.  
Under this assumption, the ~16% churn rate is far above healthy monthly benchmarks of **5–10%** seen in subscription-style e-commerce  
([ScaleMath](https://scalemath.com/blog/what-is-a-good-monthly-churn-rate/?utm_source=chatgpt.com);  
[Opensend](https://www.opensend.com/post/churn-rate-ecommerce?utm_source=chatgpt.com)).  

This makes churn reduction a **serious business problem** and a strong candidate for predictive modeling and retention strategies.

---

## 🎯 Project Objectives
- Build a predictive model to **identify customers at risk of churn**.
- Enable **what-if scenario simulation** (e.g., improving satisfaction, resolving complaints).
- Provide **actionable recommendations** to reduce churn.
- Evaluate the **business impact** of churn reduction strategies.

---

## Evaluation Metrics

For this churn dataset, the following evaluation setup will be used:

1. **Main Metric → F2-Score**  
   - Prioritizes Recall 4× more than Precision, reflecting the fact that **acquiring a new customer is 5–25× more expensive than retaining an existing one** ([Harvard Business Review, 2014](https://hbr.org/2014/10/the-value-of-keeping-the-right-customers?utm_source=chatgpt.com)).  
   - This makes missing churners (false negatives) far more costly than mistakenly targeting loyal customers (false positives).  
   - Supported by churn prediction literature, where Recall-oriented metrics (e.g., F2) are recommended for imbalanced datasets ([Verbeke et al., 2012](https://doi.org/10.1016/j.dss.2012.05.005)).  

2. **Supporting Metric → ROC-AUC**  
   - Evaluates model discriminative power independent of thresholds.  
   - Widely used in churn research as a benchmark for classification quality.  

3. **Supporting Tool → Precision & Recall**  
   - Reports Precision and Recall for each class.  
   - Provides transparency into trade-offs, allowing business stakeholders to see exactly how many churners are caught versus how many loyal customers are misclassified.  

**Summary:**  
- **F2-Score** will be the headline evaluation metric.  
- **ROC-AUC** provides a threshold-independent comparison across models.  
- **Precision & Recall** ensures interpretability and business clarity.

---

## Exploratory Data Analysis

Data columns (total 20 columns) & 5630 rows:

| # | Column Name | Non-Null Count | Data Type |
| :--- | :--- | :--- | :--- |
| 0 | CustomerID | 5630 | `int64` |
| 1 | Churn | 5630 | `int64` |
| 2 | Tenure | 5366 | `float64` |
| 3 | PreferredLoginDevice | 5630 | `object` |
| 4 | CityTier | 5630 | `int64` |
| 5 | WarehouseToHome | 5379 | `float64` |
| 6 | PreferredPaymentMode | 5630 | `object` |
| 7 | Gender | 5630 | `object` |
| 8 | HourSpendOnApp | 5375 | `float64` |
| 9 | NumberOfDeviceRegistered | 5630 | `int64` |
| 10 | PreferedOrderCat | 5630 | `object` |
| 11 | SatisfactionScore | 5630 | `int64` |
| 12 | MaritalStatus | 5630 | `object` |
| 13 | NumberOfAddress | 5630 | `int64` |
| 14 | Complain | 5630 | `int64` |
| 15 | OrderAmountHikeFromlastYear | 5365 | `float64` |
| 16 | CouponUsed | 5374 | `float64` |
| 17 | OrderCount | 5372 | `float64` |
| 18 | DaySinceLastOrder | 5323 | `float64` |
| 19 | CashbackAmount | 5630 | `float64` |

- Missing Values Exist
![](./assets/missing_values.png)

- Most of the numerical features are not normal distributed
![](./assets/bivariate_tenure.png)

---

## Methodology Analysis

### Preprocessing
- Imputer (Median) Numerical Features
- Robust Scaler
- OneHot Encoding
- Ordinal Encoding

### Benchmarking

**Feature Selection**
- Experiment show `70` percentile (from 31 to 21 features) perform good enough with `F2 Score > 0.94`
## 📊 FS_70 Train vs Test Results

| Model                   | Features Used | Train F2 | Test F2 | Train ROC-AUC | Test ROC-AUC | Train Precision | Test Precision | Train Recall | Test Recall |
|--------------------------|---------------|----------|---------|---------------|--------------|-----------------|----------------|--------------|-------------|
| DecisionTreeClassifier   | 21            | 1.000    | 0.960   | 1.000         | 0.977        | 1.000           | 0.907          | 1.000        | 0.974       |
| XGBClassifier            | 21            | 1.000    | 0.944   | 1.000         | 0.998        | 1.000           | 0.952          | 1.000        | 0.942       |
| RandomForestClassifier   | 21            | 1.000    | 0.934   | 1.000         | 0.998        | 1.000           | 0.967          | 1.000        | 0.926       |
| KNeighborsClassifier     | 21            | 0.860    | 0.681   | 0.992         | 0.964        | 0.959           | 0.897          | 0.838        | 0.642       |
| LogisticRegression       | 21            | 0.562    | 0.578   | 0.891         | 0.878        | 0.785           | 0.786          | 0.525        | 0.542       |

**Oversampling**
- With addition `70` percentile feature selection

## 📊 F2 Score Comparison (Test Set, Sorted by FS_70)

| Model                   | FS_70  | FS_70_ROS | FS_70_SMOTE |
|--------------------------|--------|-----------|-------------|
| DecisionTreeClassifier   | 0.960  | 0.899     | 0.862       |
| XGBClassifier            | 0.944  | 0.958     | 0.943       |
| RandomForestClassifier   | 0.934  | 0.960     | 0.918       |
| KNeighborsClassifier     | 0.681  | 0.903     | 0.889       |
| LogisticRegression       | 0.578  | 0.686     | 0.678       |

`XGBoostClassifier` and `RandomForestClassifier` chosen as benchmark models using oversampling `Random Oversampling`

### Hyperparameter Tuning
## 🏆 Final Results

| Experiment | F2-Score | Precision | Recall  | ROC-AUC |
|------------|----------|-----------|---------|---------|
| XGB + ROS  | **0.960** | 0.948     | 0.963   | 0.998   |
| RF + ROS   | 0.952    | 0.929     | 0.958   | 0.999   |

![](./assets/hyperparameter_tuning_cm.png)
`XGBoostClassifier` using `ROS` is the best preprocess + model pipeline

**Feature Importances**
![](./assets/feature_importances.png)
![](./assets/shap.png)

## Conclusion
A high-performing **XGBoost** model is successfully developed to predict customer churn.  
Using a **SelectPercentile feature selection (70%)**, the number of features was reduced from **31 to 21** while maintaining strong model performance.  

The primary metric, **F2-Score**, which prioritizes recall (catching churners), remains excellent, with the feature-selected model achieving a score of **0.960**.  

✅ This demonstrates that the model is highly effective at identifying customers at risk of churning while operating with a more compact feature set.

***

* **Tenure:** This is the **most significant predictor** of churn. New customers (`low Tenure`) are far more likely to churn than long-term customers. This is a common pattern and suggests that the initial customer experience is critical.
* **Customer Complaints:** Having a complaint on file (`Complain_0`) is the **second most important factor** and a very strong indicator of churn risk. Customers who have complained are highly likely to leave.
* **Payment and Login Methods:** The preferred payment mode (`PreferredPaymentMode_Credit Card`, `PreferredPaymentMode_E wallet`) and login device (`PreferredLoginDevice_Computer`) are important signals. This may suggest that customers who use specific methods or devices have different engagement patterns.
* **Marital Status:** Being single (`MaritalStatus_Single`) is a notable predictor of churn, while being married has a smaller impact. This finding aligns with the observation that different customer demographics have different churn probabilities.
* **Order and Engagement Metrics:** Features like `OrderAmountHikeFromlastYear`, `NumberOfAddress`, and `CashbackAmount` all have a strong negative correlation with churn. Customers who show have more addresses, receive higher cashback are much less likely to churn. The `SatisfactionScore` may not have clear measure in what context it is since it has a strong positive correlation with churn. (Satisfaction scores maybe measured not from purchase reviews, but from app reviews or customer service feedback.) 

## Recommendation Actions
1.  **Focus on New Customer Retention:** Since `Tenure` is the top predictor, create a proactive retention strategy specifically for new customers in their first few months. This could include personalized onboarding, exclusive offers, or check-in surveys to ensure they have a positive experience.
2.  **Establish a Complaint Resolution Task Force:** Given the strong link between complaints and churn, implement a high-priority system to handle customer complaints swiftly and effectively. The goal should be to resolve issues to the customer's satisfaction within a specific timeframe and monitor their engagement afterward.
3.  **Launch a Customer Engagement Program:** Use the model to identify customers with low `SatisfactionScore` or low `CashbackAmount` and target them with personalized campaigns. For example, offer a loyalty program that rewards higher cashback or a survey with a discount incentive to improve their satisfaction.
4.  **Develop Targeted Campaigns for Specific Demographics:** Use the insights from the `MaritalStatus` feature to create tailored marketing campaigns. For example, offer benefits or products that appeal to single customers to increase their engagement and loyalty.

## Measurable Impact
![](./assets/final_fs_confusion_matrix.png)

### Assumptions for this Simulation 💰

We’ll calculate costs using the general formula:  

**Total Cost = (FP + TP) × CRC + FN × CAC**

* **Customer Retention Cost (CRC):** $17  
* **Customer Acquisition Cost (CAC):** $85 [Reference](https://www.upcounting.com/blog/average-ecommerce-customer-acquisition-cost?utm_source=chatgpt.com)   
* **Sample Size:** 1,126 customers  

---

### 1. Cost With Model (Best Pipeline Confusion Matrix)

- **TP = 183**  
- **FP = 10**  
- **FN = 7**  

**Calculation:**  
- (TP + FP) × CRC = (183 + 10) × 50 = 193 × 17 = **$3,281**  
- FN × CAC = 7 × 85 = **$595**  
- **Total Cost (With Model) = $3,281 + $595 = $3,876**  

---

### 2. Cost Without Model (Naive: Treat All as At-Risk)

- **TP = 190** (all churners)  
- **FP = 936** (all non-churners treated as at-risk)  
- **FN = 0**  

**Calculation:**  
- (TP + FP) × CRC = (190 + 936) × 17 = 1,126 × 17 = **$19,142**  
- FN × CAC = 0 × 85 = **$0**  
- **Total Cost (Without Model) = $19,142**  

---

### 3. Cost Comparison  

| Scenario         | Formula Applied                       | Total Cost |
|------------------|---------------------------------------|------------|
| **With Model**   | (TP + FP) × CRC + FN × CAC = 193×17 + 7×85 | **$3,876** |
| **Without Model**| (TP + FP) × CRC + FN × CAC = 1,126×17 + 0   | **$19,142** |
| **Savings**      | —                                     | **$15,266** |

---

✅ By deploying this model, the business reduces costs from **$19,142** down to **$3,876**, achieving a net saving of **$15,266** — which is roughly **79.7% lower cost** compared to the naive approach.
