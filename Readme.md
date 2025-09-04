# 🛒 Customer Churn Prediction – E-commerce Case Study

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

## 📊 Evaluation Metrics

### 1. **Main Metric → F2-Score**  
- Prioritizes Recall **4× more** than Precision.  
- Reflects that **acquiring a new customer is 5–25× more expensive than retaining an existing one**  
([Harvard Business Review, 2014](https://hbr.org/2014/10/the-value-of-keeping-the-right-customers?utm_source=chatgpt.com)).  
- Missing churners (**false negatives**) is far more costly than mistakenly targeting loyal customers (**false positives**).  
- Supported by churn prediction literature, where Recall-oriented metrics (e.g., F2) are recommended for imbalanced datasets  
([Verbeke et al., 2012](https://doi.org/10.1016/j.dss.2012.05.005)).  

### 2. **Supporting Metric → ROC-AUC**  
- Evaluates model discriminative power independent of thresholds.  
- Widely used in churn research as a benchmark for classification quality.  

### 3. **Supporting Tool → Classification Report**  
- Reports Precision, Recall, and F1-score for each class.  
- Provides transparency into trade-offs, allowing business stakeholders to see **how many churners are caught vs. how many loyal customers are misclassified**.  

---

## ✅ Summary
- **F2-Score** will be the **headline evaluation metric**.  
- **ROC-AUC** provides a **threshold-independent comparison** across models.  
- **Classification Report** ensures **interpretability and business clarity**.  

---

## 🚀 Next Steps
- Data preprocessing & feature engineering.  
- Model training and hyperparameter tuning.  
- Deploying **Streamlit simulation tool** for business users.  
- Business impact analysis of retention strategies.  
