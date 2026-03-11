# Three tables: raw_transactions, customer_features, churn_results 
  
from sqlalchemy import ( 
    Column, Integer, String, Float, Boolean, Date, 
    DateTime, Index, UniqueConstraint 
) 
from sqlalchemy.orm import declarative_base 
from datetime import datetime 
  
Base = declarative_base() 
  
  
class RawTransaction(Base): 
    """ 
    One row per invoice line from the UCI dataset. 
    Stores cleaned transactions only (no cancellations, no nulls). 
    """ 
    __tablename__ = 'clv_raw_transactions' 
  
    id             = Column(Integer, primary_key=True, autoincrement=True) 
    invoice_no     = Column(String(20),  nullable=False) 
    customer_id    = Column(String(10),  nullable=False, index=True) 
    stock_code     = Column(String(20),  nullable=True) 
    description    = Column(String(255), nullable=True) 
    quantity       = Column(Integer,     nullable=False) 
    invoice_date   = Column(DateTime,    nullable=False, index=True) 
    unit_price     = Column(Float,       nullable=False) 
    country        = Column(String(50),  nullable=True) 
    revenue        = Column(Float,       nullable=False) 
  
    __table_args__ = ( 
        UniqueConstraint('invoice_no', 'stock_code', name='uq_invoice_line'), 
    ) 
  
  
class CustomerFeatures(Base): 
    """ 
    One row per customer. RFM features + BG/NBD inputs. 
    Recomputed each time the pipeline runs. 
    run_date allows comparing pipeline runs over time. 
    """ 
    __tablename__ = 'customer_features' 
  
    id               = Column(Integer, primary_key=True, autoincrement=True) 
    customer_id      = Column(String(10), nullable=False) 
    run_date         = Column(Date,       nullable=False) 
  
    # Raw RFM 
    frequency        = Column(Integer, nullable=False) 
    recency_days     = Column(Integer, nullable=False) 
    t_days           = Column(Integer, nullable=False)
    monetary         = Column(Float,   nullable=False) 
    avg_order_val    = Column(Float,   nullable=False) 
  
    # BG/NBD inputs (weeks) 
    frequency_bgn    = Column(Integer, nullable=False) 
    recency_weeks    = Column(Float,   nullable=False) 
    t_weeks          = Column(Float,   nullable=False) 
  
    # Derived flags 
    one_time_buyer   = Column(Boolean, nullable=False) 
    churned          = Column(Boolean, nullable=False) 
  
    __table_args__ = ( 
        UniqueConstraint('customer_id', 'run_date', name='uq_customer_run'), 
        Index('ix_customer_features_customer', 'customer_id'), 
        Index('ix_customer_features_run_date', 'run_date'), 
    ) 
  
  
class ChurnResult(Base): 
    """ 
    One row per customer per pipeline run. 
    Stores CLV predictions, churn probability, and segment. 
    """ 
    __tablename__ = 'churn_results' 
  
    id                    = Column(Integer, primary_key=True, autoincrement=True) 
    customer_id           = Column(String(10), nullable=False) 
    run_date              = Column(Date,       nullable=False) 
  
    # BG/NBD outputs 
    predicted_purchases   = Column(Float,   nullable=True) 
    p_alive               = Column(Float,   nullable=True) 
  
    # Gamma-Gamma outputs 
    expected_avg_order    = Column(Float,   nullable=True) 
    clv_12w               = Column(Float,   nullable=True) 
  
    # Churn classifier outputs 
    churn_probability     = Column(Float,   nullable=True) 
    churn_predicted       = Column(Boolean, nullable=True) 
    actual_churned        = Column(Boolean, nullable=True) 
  
    # Segment (computed from CLV + churn_probability) 
    # High CLV + Low churn  → 'Champions' 
    # High CLV + High churn → 'At Risk' 
    # Low CLV  + Low churn  → 'Promising' 
    # Low CLV  + High churn → 'Lost' 
    segment               = Column(String(20), nullable=True) 
  
    __table_args__ = ( 
        UniqueConstraint('customer_id', 'run_date', name='uq_churn_result_run'), 
        Index('ix_churn_results_customer', 'customer_id'), 
    ) 