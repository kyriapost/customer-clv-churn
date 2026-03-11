# scripts/ingest_data.py 
# Loads the UCI Online Retail Excel file and inserts cleaned 
# transactions into clv_raw_transactions. 
# Idempotent: uses INSERT ... ON CONFLICT DO NOTHING. 
  
import sys, os 
sys.path.insert(0, 
os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
  
import pandas as pd 
from sqlalchemy.dialects.postgresql import insert 
from sqlalchemy.orm import Session 
from dotenv import load_dotenv 
from tqdm import tqdm 
  
load_dotenv() 
  
from src.data.database import get_engine 
from src.data.models import RawTransaction 
  
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'Online Retail.xlsx') 
CHUNK     = 500 
  
  
def load_and_clean() -> pd.DataFrame: 
    print('Loading Excel file...') 
    df = pd.read_excel(DATA_PATH) 
  
    # Drop rows with no CustomerID 
    df = df.dropna(subset=['CustomerID']) 
    df['CustomerID'] = df['CustomerID'].astype(int).astype(str) 
  
    # Remove cancellations and returns 
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')] 
    df = df[df['Quantity'] > 0] 
    df = df[df['UnitPrice'] > 0] 
  
    # Rename columns to match the model 
    df = df.rename(columns={ 
        'InvoiceNo':     'invoice_no',
        'CustomerID': 'customer_id', 
        'StockCode':   'stock_code', 
        'Description': 'description', 
        'Quantity':    'quantity', 
        'InvoiceDate': 'invoice_date', 
        'UnitPrice':       'unit_price', 
        'Country':     'country', 
    }) 
  
    df['revenue'] = df['quantity'] * df['unit_price'] 
  
    print(f'Cleaned: {len(df):,} rows, {df["customer_id"].nunique():,} customers') 
    return df 
  
  
def ingest(df: pd.DataFrame) -> None: 
    engine = get_engine() 
    records = df.to_dict(orient='records') 
  
    inserted = 0 
    with Session(engine) as session: 
        for i in tqdm(range(0, len(records), CHUNK), desc='Ingesting'): 
            chunk = records[i:i+CHUNK] 
            stmt = insert(RawTransaction).values(chunk) 
            stmt = stmt.on_conflict_do_nothing( 
                constraint='uq_invoice_line' 
            ) 
            result = session.execute(stmt) 
            inserted += result.rowcount 
        session.commit() 
  
    print(f'Inserted {inserted:,} new rows') 
  
  
if __name__ == '__main__': 
    df = load_and_clean() 
    ingest(df) 
    print('Ingestion complete.') 