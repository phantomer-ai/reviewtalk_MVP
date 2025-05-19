import pandas as pd

def load_reviews(csv_path='reviews.csv'):
    df = pd.read_csv(csv_path)
    # 리뷰 내용만 리스트로 반환 (필요시 다른 컬럼도 반환 가능)
    return df['리뷰내용'].astype(str).tolist() 