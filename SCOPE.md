# Customer Lifetime Value and Churn Prediction

## Problem Statement

Retailers cannot distingish high-value customers who will return from lapsed customers who will not. Without this, marketing spend is uniform when it should be targeted

## Target User

CRM or Marketing Analyst who wants a ranked list of customers by:
- Expected Future Value
- Churn Risk

## Approach

Two-stage modelling:

1. Customer Lifetime Value:
    - BG/NBD: Given a customer's purchase history, predicts how many purchases they will make in the next N weeks and what the expected revenue will be.
    - Gamma + Gamma: Given a customer's purchase history, predicts the expected average transaction value for a customer's future purchases.
2. Churn Classification: Given features and CLV predictions, predicts whether a customer will churn

## Design Decisions

- Decision 1: Churn definition 
  - No purchase in the last 90 days of the dataset. 
  - Rationale: non-contractual retail setting; 90 days is standard in literature; matches the dataset's date range (Dec 2010 – Dec 2011). 
 
- Decision 2: Train/test split 
  - Split at 75% of the time dimension (not random). 
  - Training period: Dec 2010 – Sep 2011 
  - Test period: Sep 2011 – Dec 2011 (Q4 holdout — same as Project #1) 
  - Rationale: time-ordered split prevents data leakage. 
 
- Decision 3: CLV as a churn feature 
  - The BG/NBD 'alive' probability (P(customer is still active)) is used as a feature in the churn classifier. This links the two models. 
  - Rationale: alive probability captures recency-frequency interaction more efficiently than raw RFM features alone. 
 
- Decision 4: Classifier choice 
  - XGBoost as primary, with logistic regression as interpretable baseline. 
  - Rationale: XGBoost handles class imbalance well, fast to train on this dataset size, and feature importance is directly interpretable. 

## Architecture

- Data layer:    PostgreSQL 
    - New tables: customer_features, clv_predictions, churn_predictions 
    - Alembic migrations
 
- Feature layer: 
    - src/features/rfm.py — RFM aggregation from transactions 
    - src/features/cohort.py — train/test split by date 
 
- Model layer:   
    - src/models/clv.py — BG/NBD + Gamma-Gamma (lifetimes library) 
    - src/models/churn.py — XGBoost classifier + threshold optimisation 
    - src/models/evaluation.py — AUC, PR curve, calibration 
 
- Serving layer: FastAPI app in app/api.py — POST /predict endpoint 
    - Input: CustomerID (or raw RFM values) 
    - Output: {clv_90d, churn_probability, segment} 
 
- App layer:     Streamlit dashboard

## Definition of Done
- BG/NBD model achieves MAE on predicted vs actual purchases < 2.0
- Churn classifier achieves AUC-ROC >= 0.75 on test set
- Precision at top 20% of predicted churners >= 60%
- Fast API/predict endpoint returns a response in < 500ms
- Streamlit dashboard live with public URL
- Methodology Document
- Test suite passed > 30 tests, CI Green

## Out of Scope

- Deep learning models (LSTM for purchase sequence) — deferred to FUTURE.md 
- Real-time feature computation — batch pipeline only in v1 
- Customer segmentation clustering — CLV segments sufficient for v1 
- A/B test framework for intervention measurement 
- Multi-channel attribution (only one channel in dataset) 

## Dataset

UCI online retail

https://archive.ics.uci.edu/dataset/352/online+retail

This is a transactional data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.

**Variable Information**
- InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation. 
- StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
Description: Product (item) name. Nominal.
- Quantity: The quantities of each product (item) per transaction. Numeric.	
- InvoiceDate: Invoice Date and time. Numeric, the day and time when each transaction was generated.
- UnitPrice: Unit price. Numeric, Product price per unit in sterling.
- CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
- Country: Country name. Nominal, the name of the country where each customer resides. 

**Known limitation:** 
- One year of data
- UK-only online retailer
- No customer demographics - RFM features only
- 25% of transactions have no CustomerID