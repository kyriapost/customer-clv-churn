# CHANGELOG — Customer CLV & Churn Prediction 
  
## [Unreleased] 
  
## [0.1.0] — [11/3/2026] 
### Added 
- SCOPE.md: full project definition — problem, dataset, architecture, 
  success metrics (M1–M4), design decisions 
  - Initial project scaffold: folder structure, CI, requirements.txt 
  - notebooks/00_EDA.ipynb: dataset first look 
  
### Dataset first look ([11/3/2026]) 
- Source: UCI Online Retail
- Customers with valid CustomerID: 4,338 
- After cleaning (no returns, no cancellations): 4,338 customers 
- Date range: Dec 2010 – Dec 2011 
- Churn rate (90-day threshold): 33,6% 
- One-time buyers: 34,4% 
- Median frequency (repeat purchases): 3,38 
- Median monetary value: £2031.16 
  
### Design decisions locked 
- Churn defined as no purchase in last 90 days 
- Train period: Dec 2010 – Sep 2011 
- Test period: Sep 2011 – Dec 2011 
- Two-stage model: BG/NBD CLV → XGBoost churn classifier 