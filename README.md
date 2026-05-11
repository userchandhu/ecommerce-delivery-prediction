# 📦 E-Commerce Delivery Prediction App

🚀 Project Overview

In the fast-growing e-commerce industry, delivery delays directly impact customer satisfaction and business efficiency.
This project aims to build a predictive system that can identify high-risk deliveries before dispatch.

The complete pipeline includes:

✔ Data Cleaning & Preprocessing
✔ Feature Engineering
✔ Exploratory Data Analysis (EDA)
✔ Model Building & Evaluation
✔ Cross Validation & Hyperparameter Tuning
✔ Feature Importance Analysis
✔ Streamlit Deployment
✔ Power BI Dashboard Visualization

🎯 Business Problem

Late deliveries can lead to:

Poor customer experience
Increased operational costs
Higher customer complaints
Reduced trust in logistics systems

This project helps businesses proactively identify delayed shipments and improve delivery operations.

📊 Dataset Features

The dataset contains logistics and customer-related information such as:

Warehouse Block
Mode of Shipment
Customer Care Calls
Customer Rating
Product Cost
Prior Purchases
Product Importance
Gender
Discount Offered
Product Weight
Delivery Status (Target Variable)
⚙️ Feature Engineering

Created additional meaningful features to improve model learning:

Feature	Purpose
Cost_per_gram	Identifies expensive lightweight products
Discount_percent	Measures actual discount impact
Call_intensity	Detects operational/customer dissatisfaction
🔍 Exploratory Data Analysis (EDA)

Performed detailed analysis using:

Univariate Analysis
Bivariate Analysis
Correlation Analysis
Distribution Analysis
Key Insights:
Higher discounts increased delay probability
Heavy products faced more delivery delays
More customer care calls indicated delivery issues
Shipment mode strongly affected delivery performance
🤖 Models Implemented

The following models were trained and evaluated:

Logistic Regression
Decision Tree
Random Forest
AdaBoost
Gradient Boosting
XGBoost
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)
Artificial Neural Network (ANN)
📈 Model Evaluation Metrics

Models were evaluated using:

Accuracy
Precision
Recall
F1-Score
ROC-AUC Score
Confusion Matrix
K-Fold Cross Validation
🏆 Final Model Selection

After comparing all models, Gradient Boosting Classifier was selected as the final model based on balanced performance and business suitability.

Final Performance:
Accuracy: ~66%
Recall: ~68%
Precision: ~73%
F1-Score: ~70%
🧠 Feature Importance Analysis

Top influential features:

Discount Offered
Cost per Gram
Product Weight
Discount Percentage
Product Cost

These features had the highest impact on delivery prediction.

🌐 Deployment

The model was deployed using:

Streamlit for real-time prediction
Power BI for interactive dashboard visualization
Deployment Features:

✔ Real-time order prediction
✔ User-friendly interface
✔ Probability-based prediction output
✔ Dynamic feature processing

📊 Dashboard Highlights

The Power BI dashboard provides insights into:

Delivery performance trends
Shipment analysis
Discount impact
Weight distribution
Customer behavior analysis
🛠 Tech Stack
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost
TensorFlow / Keras
Streamlit
Power BI
📚 Key Learning Outcomes

Through this project, I learned:

End-to-end ML pipeline development
Real-world feature engineering
Model comparison techniques
Hyperparameter tuning
Deployment workflow
Business-oriented data storytelling
📌 Conclusion

This project demonstrates how Machine Learning can transform raw logistics data into actionable business intelligence.

The final solution helps businesses:

Predict delivery delays early
Improve shipment planning
Enhance customer satisfaction
Reduce operational inefficiencies
👨‍💻 Author

Prem Chand
Aspiring Data Scientist | AI & Machine Learning Enthusiast

LinkedIn: [Add Your LinkedIn Link]
GitHub: [Add Your GitHub Link]

⭐ If you found this project useful

Feel free to:

Star this repository
Fork the project
Connect with me on LinkedIn
