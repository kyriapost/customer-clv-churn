import pandas as pd 
from sqlalchemy.orm import Session 
from datetime import date 
from typing import Optional 
  
from src.data.database import get_engine 
from src.data.models import RawTransaction, CustomerFeatures, ChurnResult 
  
  
def load_transactions() -> pd.DataFrame: 
    """ 
    Returns all cleaned transactions from clv_raw_transactions. 
    Used by the feature engineering layer to build RFM features. 
    """ 
    engine = get_engine() 
    with Session(engine) as session: 
        rows = session.query(RawTransaction).all() 
        if not rows: 
            raise ValueError('clv_raw_transactions is empty. Run ingest_data.py first.') 
        data = [{ 
            'customer_id':  r.customer_id, 
            'invoice_no':   r.invoice_no, 
            'invoice_date': r.invoice_date,
            'revenue':      r.revenue, 
            'quantity':     r.quantity, 
            'unit_price':   r.unit_price, 
        } for r in rows] 
    return pd.DataFrame(data) 
  
  
def load_customer_features(run_date: Optional[date] = None) -> pd.DataFrame: 
    """ 
    Returns customer features. 
    If run_date is None, returns the most recent run. 
    Raises ValueError if no data found. 
    """ 
    engine = get_engine() 
    with Session(engine) as session: 
        if run_date is None: 
            latest = session.query( 
                CustomerFeatures.run_date 
            ).order_by(CustomerFeatures.run_date.desc()).first() 
            if latest is None: 
                raise ValueError('No customer features found. Run the pipeline first.') 
            run_date = latest[0] 
  
        rows = session.query(CustomerFeatures).filter( 
            CustomerFeatures.run_date == run_date 
        ).all() 
  
        data = [{ 
            'customer_id':    r.customer_id, 
            'frequency':      r.frequency, 
            'recency_days':   r.recency_days, 
            't_days':         r.t_days, 
            'monetary':       r.monetary, 
            'avg_order_val':  r.avg_order_val, 
            'frequency_bgn':  r.frequency_bgn, 
            'recency_weeks':  r.recency_weeks, 
            't_weeks':        r.t_weeks, 
            'one_time_buyer': r.one_time_buyer, 
            'churned':        r.churned, 
        } for r in rows] 
  
    if not data: 
        raise ValueError(f'No customer features found for run_date={run_date}') 
    return pd.DataFrame(data) 
  
  
def load_churn_results(run_date: Optional[date] = None) -> pd.DataFrame: 
    """ 
    Returns churn and CLV results. 
    If run_date is None, returns the most recent run. 
    Raises ValueError if no data found. 
    """ 
    engine = get_engine() 
    with Session(engine) as session: 
        if run_date is None: 
            latest = session.query( 
                ChurnResult.run_date 
            ).order_by(ChurnResult.run_date.desc()).first() 
            if latest is None: 
                raise ValueError('No churn results found. Run the pipeline first.') 
            run_date = latest[0] 
  
        rows = session.query(ChurnResult).filter( 
            ChurnResult.run_date == run_date 
        ).all() 
  
        data = [{ 
            'customer_id':         r.customer_id, 
            'predicted_purchases': r.predicted_purchases, 
            'p_alive':             r.p_alive, 
            'expected_avg_order':  r.expected_avg_order, 
            'clv_12w':             r.clv_12w, 
            'churn_probability':   r.churn_probability, 
            'churn_predicted':     r.churn_predicted, 
            'actual_churned':      r.actual_churned, 
            'segment':             r.segment, 
        } for r in rows] 
  
    if not data: 
        raise ValueError(f'No churn results found for run_date={run_date}') 
    return pd.DataFrame(data) 
  
  
def save_customer_features(df: pd.DataFrame, run_date: date) -> None: 
    """ 
    Upserts customer features for a given run_date. 
    """ 
    from sqlalchemy.dialects.postgresql import insert 
    engine = get_engine() 
    records = df.to_dict(orient='records') 
    for r in records: 
        r['run_date'] = run_date 
  
    with Session(engine) as session: 
        stmt = insert(CustomerFeatures).values(records) 
        stmt = stmt.on_conflict_do_update( 
            constraint='uq_customer_run', 
            set_={c: stmt.excluded[c] for c in 
                  ['frequency','recency_days','t_days','monetary', 
                   'avg_order_val','frequency_bgn','recency_weeks', 
                   't_weeks','one_time_buyer','churned']} 
        ) 
        session.execute(stmt) 
        session.commit() 
    print(f'Saved {len(records):,} customer features for {run_date}') 
  
  
def save_churn_results(df: pd.DataFrame, run_date: date) -> None: 
    """ 
    Upserts churn results for a given run_date. 
    """ 
    from sqlalchemy.dialects.postgresql import insert 
    engine = get_engine() 
    records = df.to_dict(orient='records') 
    for r in records: 
        r['run_date'] = run_date 
  
    with Session(engine) as session: 
        stmt = insert(ChurnResult).values(records) 
        stmt = stmt.on_conflict_do_update( 
            constraint='uq_churn_result_run', 
            set_={c: stmt.excluded[c] for c in 
                  ['predicted_purchases','p_alive','expected_avg_order', 
                   'clv_12w','churn_probability','churn_predicted', 
                   'actual_churned','segment']} 
        ) 
        session.execute(stmt) 
        session.commit() 
    print(f'Saved {len(records):,} churn results for {run_date}') 