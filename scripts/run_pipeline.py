import sys, os 
sys.path.insert(0, 
os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
  
from dotenv import load_dotenv 
load_dotenv() 
  
import pandas as pd 
from datetime import date 
  
from src.data.loader import ( 
    load_transactions, 
    save_customer_features, 
    save_churn_results, 
) 
from src.features.rfm import build_rfm, split_train_test 
from src.models.clv import ( 
    fit_bgnbd, 
    fit_gamma_gamma, 
    predict_clv, 
    evaluate_bgnbd, 
) 
  
RUN_DATE = date.today() 
  
  
def main(): 
    print('=' * 60) 
    print('CLV PIPELINE') 
    print(f'Run date: {RUN_DATE}') 
    print('=' * 60) 
  
    # ── 1. Load transactions ────────────────────────────────────── 
    print('\n[1/6] Loading transactions...') 
    transactions = load_transactions() 
    print(f'      {len(transactions):,} rows, {transactions["customer_id"].nunique():,} customers') 
  
    # ── 2. Train/test split ─────────────────────────────────────── 
    print('\n[2/6] Splitting train/test...') 
    train_tx, test_tx = split_train_test(transactions, test_fraction=0.25) 
  
    # ── 3. Build RFM features ───────────────────────────────────── 
    print('\n[3/6] Building RFM features...') 
    result = build_rfm(train_tx) 
    rfm    = result.rfm 
    gg_df  = result.gg_df 
    print(f'      {len(rfm):,} customers') 
    print(f'      Churn rate ({result.churn_rate*100:.1f}%)') 
    print(f'      One-time buyers: {rfm["one_time_buyer"].mean()*100:.1f}%') 
  
    # ── 4. Fit models ───────────────────────────────────────────── 
    print('\n[4/6] Fitting BG/NBD...') 
    bgf = fit_bgnbd(rfm) 
  
    print('\n      Fitting Gamma-Gamma...') 
    ggf = fit_gamma_gamma(gg_df) 
  
    # ── 5. Evaluate on test set ─────────────────────────────────── 
    print('\n[5/6] Evaluating on test set (M1 metric)...') 
    test_result = build_rfm(test_tx) 
    test_rfm    = test_result.rfm.copy() 
  
    # Actual purchases in test period per customer 
    actual = test_tx.groupby('customer_id')['invoice_no'].nunique().reset_index() 
    actual.columns = ['customer_id', 'actual_purchases'] 
    test_rfm = test_rfm.merge(actual, on='customer_id', how='left') 
    test_rfm['actual_purchases'] = test_rfm['actual_purchases'].fillna(0)
  
    mae = evaluate_bgnbd(bgf, test_rfm) 
    print(f'      MAE = {mae:.4f} | Target: < 2.0 | {"MET" if mae < 2.0 else 
"NOT MET"}') 
  
    # ── 6. Generate predictions and save ───────────────────────── 
    print('\n[6/6] Generating predictions and saving to database...') 
    predictions = predict_clv(rfm, gg_df, bgf, ggf, horizon=12) 
  
    # Save customer features 
    features_to_save = rfm[['customer_id','frequency','recency_days', 
                            't_days','monetary','avg_order_val', 
                            'frequency_bgn','recency_weeks','t_weeks', 
                            'one_time_buyer','churned']].copy() 
    features_to_save['one_time_buyer'] = features_to_save['one_time_buyer'].astype(bool) 
    features_to_save['churned']        = features_to_save['churned'].astype(bool) 
    save_customer_features(features_to_save, RUN_DATE) 
  
    # Save CLV predictions (churn columns filled in Day 4) 
    save_churn_results(predictions, RUN_DATE) 
  
    print('\n' + '=' * 60) 
    print('PIPELINE COMPLETE') 
    print(f'  Customers processed: {len(rfm):,}') 
    print(f'  BG/NBD MAE:          {mae:.4f}') 
    print(f'  Churn rate:          {result.churn_rate*100:.1f}%') 
    segment_counts = predictions['segment'].value_counts() 
    for seg, count in segment_counts.items(): 
        print(f'  {seg:<15} {count:>5} customers ({count/len(predictions)*100:.1f}%)') 
    print('=' * 60) 
  
  
if __name__ == '__main__': 
    main() 