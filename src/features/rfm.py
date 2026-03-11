import pandas as pd 
from dataclasses import dataclass 
from typing import Tuple 
  
  
CHURN_THRESHOLD_DAYS = 90   # SCOPE.md Design Decision 1 
AVG_ORDER_CAP_PERCENTILE = 0.99  # cap outliers before Gamma-Gamma 
  
  
@dataclass 
class RFMResult: 
    """ 
    Container for the RFM feature engineering output. 
    rfm:          full customer-level DataFrame (all customers) 
    gg_df:        subset for Gamma-Gamma (frequency_bgn > 0 only) 
    snapshot_date: the reference date used for recency calculation 
    churn_rate:   fraction of customers churned by the 90-day rule 
    """ 
    rfm:           pd.DataFrame 
    gg_df:         pd.DataFrame 
    snapshot_date: pd.Timestamp 
    churn_rate:    float 
  
  
def build_rfm(transactions: pd.DataFrame) -> RFMResult: 
    """ 
    Build customer-level RFM features from a transactions DataFrame. 
  
    Expected columns: customer_id, invoice_no, invoice_date, revenue 
  
    Returns RFMResult with rfm DataFrame containing: 
      - frequency:      total number of invoices 
      - recency_days:   days since last purchase 
      - t_days:         days since first purchase (customer age) 
      - monetary:       total revenue 
      - avg_order_val:  monetary / frequency (capped at 99th pct) 
      - frequency_bgn:  frequency - 1 (repeat purchases for BG/NBD) 
      - recency_weeks:  weeks between first and last purchase 
      - t_weeks:        customer age in weeks 
      - one_time_buyer: True if frequency == 1 
      - churned:        True if recency_days >= CHURN_THRESHOLD_DAYS 
    """ 
    if transactions.empty: 
        raise ValueError('transactions DataFrame is empty') 
  
    transactions = transactions.copy()
    transactions['invoice_date'] = pd.to_datetime(transactions['invoice_date']) 
  
    snapshot_date = transactions['invoice_date'].max() + pd.Timedelta(days=1) 
  
    # ── Aggregate to customer level ─────────────────────────────── 
    rfm = transactions.groupby('customer_id').agg( 
        last_purchase  = ('invoice_date', 'max'), 
        first_purchase = ('invoice_date', 'min'), 
        frequency      = ('invoice_no',   'nunique'), 
        monetary       = ('revenue',      'sum'), 
    ).reset_index() 
  
    # ── Recency and customer age ────────────────────────────────── 
    rfm['recency_days'] = (snapshot_date - rfm['last_purchase']).dt.days 
    rfm['t_days']       = (snapshot_date - rfm['first_purchase']).dt.days 
  
    # ── Average order value (capped) ────────────────────────────── 
    rfm['avg_order_val'] = rfm['monetary'] / rfm['frequency'] 
    cap = rfm['avg_order_val'].quantile(AVG_ORDER_CAP_PERCENTILE) 
    rfm['avg_order_val'] = rfm['avg_order_val'].clip(upper=cap) 
  
    # ── BG/NBD columns ──────────────────────────────────────────── 
    # frequency_bgn: repeat purchases (first excluded) 
    # recency_weeks: time between first and last purchase in weeks 
    # t_weeks:       customer age in weeks 
    rfm['frequency_bgn'] = rfm['frequency'] - 1 
    rfm['recency_weeks'] = (rfm['t_days'] - rfm['recency_days']) / 7 
    rfm['t_weeks']       = rfm['t_days'] / 7 
  
    # ── Flags ───────────────────────────────────────────────────── 
    rfm['one_time_buyer'] = rfm['frequency'] == 1 
    rfm['churned']        = rfm['recency_days'] >= CHURN_THRESHOLD_DAYS 
  
    churn_rate = rfm['churned'].mean() 
  
    # ── Gamma-Gamma subset ──────────────────────────────────────── 
    # Gamma-Gamma requires at least 1 repeat purchase 
    gg_df = rfm[rfm['frequency_bgn'] > 0].copy() 
  
    # Drop intermediate columns not needed downstream 
    rfm = rfm.drop(columns=['last_purchase', 'first_purchase']) 
  
    return RFMResult( 
        rfm=rfm, 
        gg_df=gg_df, 
        snapshot_date=snapshot_date, 
        churn_rate=churn_rate, 
    ) 
  
  
def split_train_test( 
    transactions: pd.DataFrame, 
    test_fraction: float = 0.25, 
) -> Tuple[pd.DataFrame, pd.DataFrame]: 
    """ 
    Time-ordered train/test split. 
    SCOPE.md Design Decision 2: split at 75% of the time dimension. 
    Returns (train_transactions, test_transactions). 
    """ 
    transactions = transactions.copy() 
    transactions['invoice_date'] = pd.to_datetime(transactions['invoice_date']) 
  
    min_date = transactions['invoice_date'].min() 
    max_date = transactions['invoice_date'].max() 
    total_days = (max_date - min_date).days 
    split_date = min_date + pd.Timedelta(days=int(total_days * (1 - test_fraction))) 
  
    train = transactions[transactions['invoice_date'] <  split_date] 
    test  = transactions[transactions['invoice_date'] >= split_date] 
  
    print(f'Train: {min_date.date()} to {split_date.date()} — {len(train):,} rows') 
    print(f'Test:  {split_date.date()} to {max_date.date()} — {len(test):,} rows') 
  
    return train, test 