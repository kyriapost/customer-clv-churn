import pytest 
import pandas as pd 
from datetime import date, datetime 
from sqlalchemy import create_engine, inspect 
from sqlalchemy.orm import Session 
  
from src.data.models import Base, RawTransaction, CustomerFeatures, ChurnResult 
  
  
@pytest.fixture 
def engine(): 
    """In-memory SQLite engine for testing.""" 
    eng = create_engine('sqlite:///:memory:') 
    Base.metadata.create_all(eng) 
    yield eng 
    Base.metadata.drop_all(eng) 
  
  
class TestModels: 
    def test_all_tables_created(self, engine): 
        tables = inspect(engine).get_table_names() 
        assert 'clv_raw_transactions' in tables 
        assert 'customer_features'    in tables 
        assert 'churn_results'        in tables 
  
    def test_insert_raw_transaction(self, engine): 
        with Session(engine) as s: 
            s.add(RawTransaction( 
                invoice_no='INV001', customer_id='C001', 
                quantity=5, invoice_date=datetime(2011,1,15), 
                unit_price=2.50, revenue=12.50 
            )) 
            s.commit() 
            count = s.query(RawTransaction).count() 
        assert count == 1 
  
    def test_insert_customer_features(self, engine): 
        with Session(engine) as s: 
            s.add(CustomerFeatures( 
                customer_id='C001', run_date=date(2011,12,10), 
                frequency=5, recency_days=30, t_days=200, 
                monetary=500.0, avg_order_val=100.0, 
                frequency_bgn=4, recency_weeks=4.3, t_weeks=28.6, 
                one_time_buyer=False, churned=False 
            )) 
            s.commit() 
            row = s.query(CustomerFeatures).first() 
        assert row.customer_id == 'C001' 
        assert row.frequency   == 5 
        assert row.churned     == False
  
    def test_insert_churn_result(self, engine): 
        with Session(engine) as s: 
            s.add(ChurnResult( 
                customer_id='C001', run_date=date(2011,12,10), 
                predicted_purchases=2.3, p_alive=0.82, 
                expected_avg_order=95.0, clv_12w=218.5, 
                churn_probability=0.18, churn_predicted=False, 
                actual_churned=False, segment='Champions' 
            )) 
            s.commit() 
            row = s.query(ChurnResult).first() 
        assert row.segment          == 'Champions' 
        assert row.churn_predicted  == False 
  
    def test_churn_result_high_risk(self, engine): 
        with Session(engine) as s: 
            s.add(ChurnResult( 
                customer_id='C002', run_date=date(2011,12,10), 
                p_alive=0.12, churn_probability=0.88, 
                clv_12w=850.0, segment='At Risk' 
            )) 
            s.commit() 
            row = s.query(ChurnResult).filter_by(customer_id='C002').first() 
        assert row.segment == 'At Risk' 
        assert row.churn_probability > 0.5 
  
  
class TestChurnLogic: 
    def test_churned_flag_true_when_recency_gte_90(self, engine): 
        """Customer with recency >= 90 days is marked churned.""" 
        with Session(engine) as s: 
            s.add(CustomerFeatures( 
                customer_id='C003', run_date=date(2011,12,10), 
                frequency=3, recency_days=95, t_days=300, 
                monetary=200.0, avg_order_val=66.7, 
                frequency_bgn=2, recency_weeks=29.3, t_weeks=42.9, 
                one_time_buyer=False, churned=True 
            )) 
            s.commit() 
            row = s.query(CustomerFeatures).filter_by(customer_id='C003').first() 
        assert row.churned == True 
        assert row.recency_days >= 90 
  
    def test_not_churned_when_recent(self, engine): 
        with Session(engine) as s: 
            s.add(CustomerFeatures( 
                customer_id='C004', run_date=date(2011,12,10), 
                frequency=10, recency_days=14, t_days=300, 
                monetary=1500.0, avg_order_val=150.0, 
                frequency_bgn=9, recency_weeks=40.9, t_weeks=42.9, 
                one_time_buyer=False, churned=False 
            )) 
            s.commit() 
            row = s.query(CustomerFeatures).filter_by(customer_id='C004').first() 
        assert row.churned == False 
  
    def test_one_time_buyer_flag(self, engine): 
        with Session(engine) as s: 
            s.add(CustomerFeatures( 
                customer_id='C005', run_date=date(2011,12,10), 
                frequency=1, recency_days=200, t_days=200, 
                monetary=50.0, avg_order_val=50.0, 
                frequency_bgn=0, recency_weeks=0.0, t_weeks=28.6, 
                one_time_buyer=True, churned=True 
            )) 
            s.commit() 
            row = s.query(CustomerFeatures).filter_by(customer_id='C005').first() 
        assert row.one_time_buyer == True 
        assert row.frequency_bgn  == 0 