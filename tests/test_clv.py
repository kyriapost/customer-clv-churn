import pytest 
import pandas as pd 
import numpy as np 
from src.models.clv import fit_bgnbd, fit_gamma_gamma, _assign_segment 
  
  
@pytest.fixture 
def rfm_df(): 
    """Minimal RFM DataFrame for model testing.""" 
    return pd.DataFrame({ 
        'customer_id':   ['C1','C2','C3'], 
        'frequency_bgn': [4,   2,   1  ], 
        'recency_weeks': [20,  10,  5  ], 
        't_weeks':       [52,  30,  20 ], 
        'avg_order_val': [100, 200, 50 ], 
    }) 
  
  
class TestBGNBD: 
    def test_fit_returns_model(self, rfm_df): 
        bgf = fit_bgnbd(rfm_df) 
        assert bgf is not None 
        assert hasattr(bgf, 'params_') 
  
    def test_params_are_positive(self, rfm_df): 
        bgf = fit_bgnbd(rfm_df) 
        for param, val in bgf.params_.items(): 
            assert val > 0, f'Param {param} should be positive' 
  
  
class TestGammaGamma: 
    def test_fit_returns_model(self, rfm_df): 
        ggf = fit_gamma_gamma(rfm_df) 
        assert ggf is not None 
        assert hasattr(ggf, 'params_') 
  
    def test_raises_on_zero_frequency(self): 
        bad_df = pd.DataFrame({ 
            'frequency_bgn': [0, 1], 
            'avg_order_val': [100, 200], 
        }) 
        with pytest.raises(ValueError): 
            fit_gamma_gamma(bad_df) 
  
  
class TestSegmentAssignment: 
    def test_champions(self): 
        assert _assign_segment(1000, 0.9, 500) == 'Champions' 
  
    def test_at_risk(self): 
        assert _assign_segment(1000, 0.1, 500) == 'At Risk' 
  
    def test_promising(self): 
        assert _assign_segment(100, 0.9, 500) == 'Promising' 
  
    def test_lost(self): 
        assert _assign_segment(100, 0.1, 500) == 'Lost' 