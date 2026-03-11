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

## [0.2.0] — [11/3/2026] 
### Added 
- src/data/models.py: RawTransaction, CustomerFeatures, ChurnResult tables 
- Segment logic documented: Champions / At Risk / Promising / Lost 
- UniqueConstraints on (customer_id, run_date) for upsert safety - database/migrations/: Alembic migration 'Add CLV and churn tables' 
- scripts/ingest_data.py: loads UCI Excel, cleans, inserts ~400k rows 
- Idempotent via ON CONFLICT DO NOTHING on uq_invoice_line 
- src/data/loader.py: load_transactions, load_customer_features load_churn_results, save_customer_features, save_churn_results 
- All load functions raise ValueError on empty result 
- Both save functions use upsert - tests/test_data_layer.py: 8 tests covering models and churn logic 
  
### Data 
- clv_raw_transactions: ~400000 rows ingested 
- Customers with valid CustomerID: ~4,338 