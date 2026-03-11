import pandas as pd 
import numpy as np 
from dataclasses import dataclass 
from lifetimes import BetaGeoFitter, GammaGammaFitter 
from lifetimes.utils import summary_data_from_transaction_data 
  
  
@dataclass 
class CLVResult: 
    """ 
    Container for CLV model outputs. 
    predictions:  DataFrame with one row per customer 
    bgf:          fitted BetaGeoFitter (for diagnostics) 
    ggf:          fitted GammaGammaFitter (for diagnostics) 
    mae_purchases: MAE on predicted vs actual purchases (test set) 
    """ 
    predictions:   pd.DataFrame 
    bgf:           BetaGeoFitter 
    ggf:           GammaGammaFitter 
    mae_purchases: float 
  
  
def fit_bgnbd(rfm: pd.DataFrame, penalizer: float = 0.01) -> BetaGeoFitter: 
    """ 
    Fits a BG/NBD model to the RFM DataFrame. 
    Requires columns: frequency_bgn, recency_weeks, t_weeks 
    penalizer: L2 regularisation — 0.01 is standard starting point. 
    """ 
    bgf = BetaGeoFitter(penalizer_coef=penalizer) 
    bgf.fit( 
        frequency = rfm['frequency_bgn'], 
        recency   = rfm['recency_weeks'], 
        T         = rfm['t_weeks'], 
    ) 
    print(f'BG/NBD fitted — params: {bgf.params_}') 
    return bgf 
  
  
def fit_gamma_gamma(gg_df: pd.DataFrame, penalizer: float = 0.01) -> GammaGammaFitter: 
    """ 
    Fits a Gamma-Gamma model for expected monetary value. 
    Requires columns: frequency_bgn, avg_order_val 
    Only customers with frequency_bgn > 0 should be passed. 
    """ 
    if (gg_df['frequency_bgn'] == 0).any(): 
        raise ValueError('gg_df must only contain customers with frequency_bgn > 0') 
  
    ggf = GammaGammaFitter(penalizer_coef=penalizer) 
    ggf.fit( 
        frequency            = gg_df['frequency_bgn'], 
        monetary_value       = gg_df['avg_order_val'], 
    ) 
    print(f'Gamma-Gamma fitted — params: {ggf.params_}') 
    return ggf 
  
  
def predict_clv( 
    rfm:     pd.DataFrame, 
    gg_df:   pd.DataFrame, 
    bgf:     BetaGeoFitter, 
    ggf:     GammaGammaFitter, 
    horizon: int = 12, 
) -> pd.DataFrame: 
    """ 
    Generates CLV predictions for all customers. 
  
    Returns DataFrame with columns: 
      customer_id, predicted_purchases, p_alive, 
      expected_avg_order, clv_12w, segment 
    """ 
    result = rfm[['customer_id', 'frequency_bgn', 'recency_weeks', 
                  't_weeks', 'avg_order_val', 'churned']].copy() 
  
    # BG/NBD predictions 
    result['predicted_purchases'] = bgf.conditional_expected_number_of_purchases_up_to_time( 
        horizon, 
        result['frequency_bgn'], 
        result['recency_weeks'], 
        result['t_weeks'], 
    ) 
    result['p_alive'] = bgf.conditional_probability_alive( 
        result['frequency_bgn'], 
        result['recency_weeks'], 
        result['t_weeks'], 
    ) 
  
    # Gamma-Gamma predictions (only for repeat buyers) 
    gg_preds = ggf.conditional_expected_average_profit( 
        gg_df['frequency_bgn'], 
        gg_df['avg_order_val'], 
    ) 
    gg_map = dict(zip(gg_df['customer_id'], gg_preds)) 
  
    # One-time buyers get their raw avg_order_val as the best estimate 
    result['expected_avg_order'] = result.apply( 
        lambda r: gg_map.get(r['customer_id'], r['avg_order_val']), 
        axis=1, 
    ) 
  
    # CLV = predicted purchases x expected avg order value 
    result['clv_12w'] = result['predicted_purchases'] * result['expected_avg_order'] 
  
    # ── Segment assignment ──────────────────────────────────────── 
    # Uses median CLV and 0.5 churn probability as thresholds 
    clv_median = result['clv_12w'].median() 
    result['segment'] = result.apply( 
        lambda r: _assign_segment(r['clv_12w'], r['p_alive'], clv_median), 
        axis=1, 
    ) 
  
    return result[['customer_id', 'predicted_purchases', 'p_alive', 
                   'expected_avg_order', 'clv_12w', 'segment']] 
  
  
def _assign_segment(clv: float, p_alive: float, clv_median: float) -> str: 
    """ 
    2x2 segmentation matrix: 
    High CLV + High alive  → Champions 
    High CLV + Low alive   → At Risk 
    Low CLV  + High alive  → Promising 
    Low CLV  + Low alive   → Lost 
    """ 
    high_value = clv     >= clv_median 
    high_alive = p_alive >= 0.5 
    if high_value and high_alive:  return 'Champions' 
    if high_value and not high_alive: return 'At Risk' 
    if not high_value and high_alive: return 'Promising' 
    return 'Lost' 
  
  
def evaluate_bgnbd( 
    bgf:             BetaGeoFitter, 
    test_rfm:        pd.DataFrame, 
    horizon:         int = 12, 
) -> float: 
    """ 
    Evaluates BG/NBD predictions against actual test-period purchases. 
    Returns MAE — SCOPE.md metric M1 target: MAE < 2.0 
    test_rfm must have columns: frequency_bgn, recency_weeks, t_weeks, 
    and 'actual_purchases' (purchases made in the test period). 
    """ 
    predicted = bgf.conditional_expected_number_of_purchases_up_to_time( 
        horizon, 
        test_rfm['frequency_bgn'], 
        test_rfm['recency_weeks'], 
        test_rfm['t_weeks'], 
    ) 
    mae = np.abs(predicted - test_rfm['actual_purchases']).mean() 
    print(f'BG/NBD MAE on test set: {mae:.4f} (target < 2.0)') 
    return float(mae) 