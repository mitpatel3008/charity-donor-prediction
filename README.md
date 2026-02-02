# 💰 Charity Donor Prediction System

A machine learning-based donor prediction system that identifies potential high-value donors for charity organizations. By analyzing demographic, professional, and financial data, this system predicts whether individuals earn more than $50K annually—helping charities optimize their fundraising campaigns and maximize donor outreach efficiency.

---

## 🌐 Live Demo
** https://donor-finder.streamlit.app/

---

## 🚀 Features

```
•	Achieves 93% recall in identifying potential donors (>$50K income earners)
•	Successfully identifies 1,412 out of 1,518 potential donors
•	Analyzes 13 key demographic and financial features
•	Provides real-time predictions with confidence scores
•	Interactive Streamlit web interface for easy data input
•	Optimized using XGBoost with GridSearchCV hyperparameter tuning
•	Clear probability distributions and actionable recommendations
•	Handles imbalanced dataset with class weighting strategies
```

---

## 🧠 Tech Stack

```
•	Language: Python 3.10
•	Core Libraries: 
    - Pandas, NumPy (Data Processing)
    - Matplotlib, Seaborn (Visualization)
    - Scikit-learn (ML Pipeline)
    - XGBoost (Gradient Boosting)
•	Machine Learning:
    - RandomForestClassifier
    - AdaBoostClassifier
    - XGBClassifier (Final Model)
•	Optimization: GridSearchCV with 5-fold Cross-Validation
•	Feature Engineering: LabelEncoder for categorical variables
•	Deployment: Streamlit Framework
•	Environment: Conda (donation env)
•	Development Tools: Jupyter Notebook, VS Code
```

---

## 📊 Dataset

```
•	Source: UCI Adult Income Dataset (adult.data)
•	Original Records: 32,561 entries
•	After Cleaning: 30,693 records
•	Features: 14 columns (13 predictors + 1 target)
•	Target Variable: Income (<=50K or >50K)
•	Class Distribution: 
    - Class 0 (<=50K): 75.1% (4,621 test samples)
    - Class 1 (>50K): 24.9% (1,518 test samples)
•	Challenge: Imbalanced dataset requiring recall optimization
```

### Dataset Features:
```
Numerical Features:
- age: Age of the individual
- education-num: Years of education (numerical encoding)
- capital-gain: Capital gains from investments
- capital-loss: Capital losses
- hours-per-week: Working hours per week

Categorical Features:
- workclass: Employment type (Private, Self-employed, Government, etc.)
- education: Education level (Bachelors, Masters, HS-grad, etc.)
- marital-status: Marital status
- occupation: Job category
- relationship: Family relationship role
- race: Ethnicity
- sex: Gender
- native-country: Country of origin

Target:
- Income: Binary classification (<=50K or >50K)
```

---

## 🛠️ Project Workflow

### 1. Data Loading & Exploration
```
•	Loaded adult.data using Pandas
•	Explored data structure with .info(), .describe()
•	Analyzed feature distributions and correlations
•	Identified target variable class imbalance (75/25 split)
```

### 2. Data Cleaning & Preprocessing
```
•	Handled missing values (removed entries with '?')
•	Removed duplicate records
•	Detected outliers in capital-gain and hours-per-week
•	Decision: Kept outliers as they represent legitimate high-earner segments
•	Final clean dataset: 30,693 records
```

### 3. Exploratory Data Analysis (EDA)
```
•	Visualized distributions using histograms and box plots
•	Created scatter plots for capital-gain and hours-per-week outliers
•	Analyzed feature importance for income prediction
•	Key findings:
    - Education strongly correlates with high income
    - Married individuals more likely to earn >50K
    - Capital gains are strong indicators of >50K income
    - Executive/Professional occupations associated with higher earnings
```

### 4. Feature Engineering
```
•	Label Encoding applied to all categorical features:
    - workclass, education, marital-status
    - occupation, relationship, race, sex
    - native-country
•	All features converted to numerical format
•	No feature scaling applied (tree-based models don't require it)
•	Preserved outliers for meaningful predictions
```

### 5. Model Development
```
Train-Test Split:
- Training Set: 80% (24,554 samples)
- Test Set: 20% (6,139 samples)
- Random State: 42 (for reproducibility)
- Stratified split to maintain class distribution

Models Tested:
1. Random Forest (Baseline)
2. Random Forest (GridSearchCV Optimized)
3. AdaBoost (Default)
4. XGBoost (Default)
5. XGBoost (GridSearchCV Optimized) ✅ FINAL MODEL
```

### 6. Hyperparameter Tuning
```
XGBoost GridSearchCV Parameters:
- n_estimators: [100, 200, 300]
- max_depth: [4, 6, 8]
- learning_rate: [0.05, 0.1, 0.2]
- scale_pos_weight: [2, 3, 4, 5] (for class imbalance)
- subsample: [0.8, 1.0]
- colsample_bytree: [0.8, 1.0]
- min_child_weight: [1, 3, 5]

Optimization Metric: Recall for Class 1 (>50K donors)
Cross-Validation: 5-fold CV
Scoring: Custom recall scorer for minority class
```

---

## 📈 Model Performance Comparison

| Model | Accuracy | Recall (>50K) | Precision (>50K) | F1-Score | Donors Found | Donors Missed |
|-------|----------|---------------|------------------|----------|--------------|---------------|
| Random Forest (Baseline) | 85.1% | 63% | 73% | 0.68 | 961/1518 | 557 |
| Random Forest (Optimized) | 81.1% | 87% | 57% | 0.69 | 1325/1518 | 193 |
| AdaBoost (Default) | 85.4% | 60% | 76% | 0.67 | 913/1518 | 605 |
| XGBoost (Default) | 87.4% | 69% | 78% | 0.73 | 1046/1518 | 472 |
| **XGBoost (Optimized)** ✅ | **77.0%** | **93%** | **52%** | **0.67** | **1412/1518** | **106** |

### Why XGBoost (Optimized) is the Winner:
```
✅ Highest Recall (93%): Finds the maximum number of potential donors
✅ Lowest Miss Rate: Only misses 106 out of 1518 donors (7%)
✅ Business Impact: $8,700+ additional revenue compared to next best model
✅ Optimization Goal Achieved: Maximized donor identification over accuracy
```

### Confusion Matrix (XGBoost Optimized):
```
                 Predicted
                <=50K    >50K
Actual <=50K    3630     991
Actual >50K     106      1412
```

### Classification Report (XGBoost Optimized):
```
              precision    recall  f1-score   support
      <=50K       0.97      0.72      0.83      4621
      >50K        0.52      0.93      0.67      1518
   accuracy                           0.77      6139
```

---

## 💡 Key Insights from Analysis

### Feature Importance:
```
1. education-num: Most predictive feature
2. age: Second strongest indicator
3. capital-gain: Critical for high-income identification
4. hours-per-week: Working hours correlate with income
5. occupation: Job type significantly impacts earnings
```

### Donor Profile Characteristics:
```
High Potential Donors (>50K):
- Age: 35-60 years (experienced professionals)
- Education: Bachelor's degree or higher
- Marital Status: Married
- Occupation: Executive, Professional, Technical roles
- Hours per Week: 40-60 hours (full-time commitment)
- Capital Gains: Present (indicates investments)
```

### Outlier Analysis:
```
Capital Gain Outliers:
- Range: $0 to $99,999
- Most values: $0 (no investments)
- Outliers: Represent high-income investors
- Decision: KEPT (valuable donor indicators)

Hours per Week Outliers:
- Range: 1-100 hours
- Outliers: 80-100 hours (entrepreneurs, multiple jobs)
- Decision: KEPT (real work patterns)
```

---

## 🎯 Business Impact

### Revenue Projection (Assuming $100 per donor):
```
Original Random Forest:
- Donors Found: 961
- Estimated Revenue: $96,100
- Lost Revenue: $55,700

XGBoost Optimized:
- Donors Found: 1,412
- Estimated Revenue: $141,200
- Lost Revenue: Only $10,600

Net Improvement: +$45,100 in potential donations
Additional Donors: +451 compared to baseline
```

### Trade-off Analysis:
```
What We Gained:
✅ 93% recall - finds almost all donors
✅ Only 106 donors missed (vs 557 in baseline)
✅ Maximum fundraising potential unlocked

What We Sacrificed:
⚠️ Lower precision (52%) - more false positives
⚠️ More outreach needed (~2,700 contacts)
⚠️ Overall accuracy reduced to 77%

Verdict: Worth it! For charity fundraising, finding 
more real donors is more valuable than avoiding 
false positives.
```

---

## 🔧 Installation & Setup

### Prerequisites
```bash
# Python 3.10+ required
# Conda recommended for environment management
```

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/charity-donor-prediction.git
cd charity-donor-prediction
```

### Step 2: Create Conda Environment
```bash
conda create -n donation python=3.10
conda activate donation
```

### Step 3: Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### Step 4: Run Jupyter Notebook (Optional)
```bash
# Explore the analysis
jupyter notebook file.ipynb
```

### Step 5: Launch Streamlit App
```bash
# Run the web application
streamlit run app/app.py
```

### Access the Application
```
Open your browser and navigate to: http://localhost:8501
```

---

## 🎮 Using the Application

### Input Fields:
```
Personal Information:
- Age (17-90)
- Sex (Male/Female)
- Race (5 categories)
- Marital Status (7 options)
- Relationship (6 family roles)

Professional Information:
- Work Class (8 employment types)
- Education (16 levels)
- Occupation (14 job categories)
- Hours per Week (1-100)

Financial Information:
- Capital Gain ($0-$100,000)
- Capital Loss ($0-$5,000)

Other:
- Native Country (41 countries)
```

### Sample Prediction:

**High Potential Donor Example:**
```
Input:
- Age: 50
- Sex: Male
- Education: Bachelors
- Occupation: Exec-managerial
- Marital Status: Married-civ-spouse
- Hours per Week: 50
- Capital Gain: $15,000

Output:
✅ HIGH POTENTIAL DONOR (Income >$50K)
Confidence: 99.2%
Recommendation: Add to priority donor contact list
```

**Low Potential Donor Example:**
```
Input:
- Age: 22
- Sex: Female
- Education: HS-grad
- Occupation: Other-service
- Marital Status: Never-married
- Hours per Week: 25
- Capital Gain: $0

Output:
❌ LOW POTENTIAL DONOR (Income ≤$50K)
Confidence: 98.5%
Recommendation: May not be ideal for high-value campaigns
```

---

## 📊 Model Interpretability

### Why the Model Works:
```
1. Ensemble Learning: XGBoost combines multiple decision trees
2. Class Balancing: scale_pos_weight addresses imbalance
3. Feature Interactions: Captures complex relationships
4. Gradient Boosting: Learns from previous mistakes
5. Regularization: Prevents overfitting with L1/L2 penalties
```

### Decision Logic:
```
The model considers combinations like:
- High education + Executive role + Married → High donor probability
- Young age + Part-time + No gains → Low donor probability
- Capital gains > $5000 → Strong donor signal
- Hours > 45 + Professional occupation → Donor indicator
```

---

## 🎓 Learning Outcomes

```
Technical Skills:
✅ Handling imbalanced datasets
✅ Optimizing for business metrics (recall over accuracy)
✅ Hyperparameter tuning with GridSearchCV
✅ Understanding precision-recall trade-offs
✅ Feature engineering with label encoding
✅ Model evaluation and comparison
✅ Building production-ready ML applications

Business Understanding:
✅ Translating ML metrics to business value
✅ Making strategic trade-offs (precision vs recall)
✅ ROI calculation for ML projects
✅ Stakeholder-focused model optimization
```

---

## 🚀 Future Enhancements

```
Feature Additions:
•	Batch prediction via CSV upload
•	SHAP values for model explainability
•	Feature importance visualization
•	Donor segmentation clustering
•	Historical tracking dashboard

Technical Improvements:
•	REST API for external integrations
•	A/B testing framework
•	Real-time model retraining
•	Integration with CRM systems
•	Mobile-responsive design

Advanced Analytics:
•	Donor lifetime value prediction
•	Churn risk analysis
•	Donation amount prediction
•	Campaign effectiveness tracking
•	Geographic donor mapping
```

---

## 📚 Dependencies

### Core Libraries:
```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
streamlit>=1.30.0
```

### Complete requirements.txt:
```
streamlit
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
pickle5
```

---

## 🤝 Use Cases

```
Charity Organizations:
- Identify high-potential donors for annual campaigns
- Optimize fundraising budget allocation
- Prioritize donor outreach efforts

Non-Profit Institutions:
- Segment donor base by income likelihood
- Target marketing campaigns effectively
- Maximize ROI on fundraising activities

Fundraising Teams:
- Build priority contact lists
- Reduce wasted outreach efforts
- Track campaign effectiveness
- Predict donation likelihood
```

---

## ⚠️ Important Notes

### Model Limitations:
```
•	Trained on US census data (may not generalize globally)
•	Binary classification (<=50K or >50K only)
•	Lower precision (52%) means more false positives
•	Requires manual input for each prediction
•	No temporal data (income trends over time)
```

### Ethical Considerations:
```
•	Model uses demographic data - ensure fair use
•	Avoid discriminatory targeting based on race/gender
•	Comply with data privacy regulations (GDPR, etc.)
•	Transparent communication with potential donors
•	Regular audits for bias and fairness
```

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

```
•	Dataset: UCI Machine Learning Repository
•	Inspiration: Real-world charity fundraising optimization
•	Community: Streamlit, XGBoost, Scikit-learn contributors
•	Environment: Anaconda Distribution
```

---

## 📧 Contact & Support

For questions, suggestions, or collaboration:
- GitHub Issues: [Create an issue](https://github.com/yourusername/charity-donor-prediction/issues)
- Email: your.email@example.com
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)

---

## 🌟 Star This Repository

If you found this project helpful, please consider giving it a ⭐ on GitHub!

---

**Built with ❤️ for effective charity donor identification and fundraising optimization**

---

### Quick Links:
- [Live Demo](#)
- [Documentation](#)
- [API Reference](#)
- [Contributing Guidelines](#)
- [Changelog](#)

---

*Last Updated: February 2026*
