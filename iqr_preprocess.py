# iqr_preprocess.py

import pandas as pd

def iqr_preprocess(input_path: str, output_path: str,
                   input_sep: str = '|', output_sep: str = '.',
                   multiplier: float = 1.5, filter_cols=None):
    # 1) 데이터 로드
    df = pd.read_csv(input_path, sep=input_sep, dtype={'YMD_ID': str, 'HH_ID': str})
    
    # 2) IQR 필터링할 컬럼 지정 (default: GETON_CNT)
    if filter_cols is None:
        filter_cols = ['GETON_CNT']
    else:
        # 커맨드라인 인자로 받은 문자열을 리스트로 변환
        if isinstance(filter_cols, str):
            filter_cols = filter_cols.split(',')
    
    # 존재하지 않는 컬럼 제거
    filter_cols = [c for c in filter_cols if c in df.columns]
    if not filter_cols:
        print("필터링할 유효한 컬럼이 없습니다:", filter_cols)
        return
    
    # 3) 각 컬럼별로 IQR 계산 & 필터링
    total_mask = pd.Series(True, index=df.index)
    for col in filter_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        
        mask = df[col].between(lower, upper)
        print(f"· {col}: {mask.sum()} / {len(df)} rows 통과 (IQR={IQR:.2f}, [{lower:.2f}, {upper:.2f}])")
        total_mask &= mask
    
    df_clean = df[total_mask].reset_index(drop=True)

    # 4) 결과 저장
    df_clean.to_csv(output_path, sep=output_sep, index=False)
    print(f"\n최종: 원본 {len(df)} rows → 클린 {len(df_clean)} rows")
    print(f"저장: {output_path} (sep='{output_sep}')")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IQR 기반 이상치 제거 (특정 컬럼에만)")
    parser.add_argument('input_csv',  help="원본 CSV 경로")
    parser.add_argument('output_csv', help="결과 CSV 경로")
    parser.add_argument('--input-sep',   default='|',  help="원본 구분자")
    parser.add_argument('--output-sep',  default='.',  help="출력 구분자")
    parser.add_argument('--multiplier',  type=float, default=1.5, help="IQR 계수")
    parser.add_argument('--filter-cols', type=str, help="콤마로 구분된 IQR 필터링 대상 컬럼 (기본 GETON_CNT)")
    args = parser.parse_args()

    iqr_preprocess(
        input_path   = args.input_csv,
        output_path  = args.output_csv,
        input_sep    = args.input_sep,
        output_sep   = args.output_sep,
        multiplier   = args.multiplier,
        filter_cols  = args.filter_cols
    )
