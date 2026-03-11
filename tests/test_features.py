import pytest 
import pandas as pd 
import numpy as np 
from src.features.rfm import build_rfm, split_train_test, CHURN_THRESHOLD_DAYS 
  
  
@pytest.fixture 
def sample_transactions(): 
    """Minimal transaction DataFrame for testing.""" 
    return pd.DataFrame({ 
        'customer_id':  ['C1','C1','C1','C2','C3'], 
        'invoice_no':   ['I1','I2','I3','I4','I5'], 
        'invoice_date': pd.to_datetime([ 
            '2011-01-01','2011-03-01','2011-06-01',  # C1: 3 orders 
            '2011-01-15',                            # C2: 1 order (one-time buyer) 
            '2011-11-01',                            # C3: recent, not churned 
        ]), 
        'revenue': [100.0, 200.0, 150.0, 80.0, 300.0], 
    }) 
  
  
class TestBuildRFM: 
    def test_returns_rfm_result(self, sample_transactions): 
        result = build_rfm(sample_transactions) 
        assert result.rfm is not None 
        assert result.gg_df is not None 
        assert result.snapshot_date is not None 
  
    def test_customer_count(self, sample_transactions): 
        result = build_rfm(sample_transactions) 
        assert len(result.rfm) == 3 
  
    def test_frequency_correct(self, sample_transactions): 
        result = build_rfm(sample_transactions) 
        c1 = result.rfm[result.rfm['customer_id'] == 'C1'].iloc[0] 
        assert c1['frequency'] == 3 
        assert c1['frequency_bgn'] == 2 
  
    def test_one_time_buyer_flag(self, sample_transactions): 
        result = build_rfm(sample_transactions) 
        c2 = result.rfm[result.rfm['customer_id'] == 'C2'].iloc[0] 
        assert c2['one_time_buyer'] == True 
        assert c2['frequency_bgn'] == 0 
  
    def test_churn_flag_respects_threshold(self, sample_transactions): 
        result = build_rfm(sample_transactions) 
        # C3 bought on 2011-11-01, snapshot is 2011-11-02 
        # recency = 1 day — should NOT be churned 
        c3 = result.rfm[result.rfm['customer_id'] == 'C3'].iloc[0] 
        assert c3['recency_days'] < CHURN_THRESHOLD_DAYS 
        assert c3['churned'] == False 
  
    def test_gg_df_excludes_one_time_buyers(self, sample_transactions): 
        result = build_rfm(sample_transactions) 
        assert 'C2' not in result.gg_df['customer_id'].values 
  
    def test_empty_raises(self): 
        with pytest.raises(ValueError): 
            build_rfm(pd.DataFrame()) 
  
  
class TestSplitTrainTest: 
    def test_split_preserves_all_rows(self, sample_transactions): 
        train, test = split_train_test(sample_transactions, test_fraction=0.25) 
        assert len(train) + len(test) == len(sample_transactions) 
  
    def test_train_comes_before_test(self, sample_transactions): 
        train, test = split_train_test(sample_transactions, test_fraction=0.25) 
        assert train['invoice_date'].max() <= test['invoice_date'].min() 