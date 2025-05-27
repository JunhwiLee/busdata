import pandas as pd
from pathlib import Path
import datetime

# 디렉토리 설정
input_dir = Path(r"BasicLAB\bus_data")
output_dir = Path(r"BasicLAB\bus_data_transfer")
output_dir.mkdir(parents=True, exist_ok=True)

# 처리할 날짜 리스트 (MMDD 형식)
dates = ["0414", "0415", "0416", "0417", "0418", "0419", "0420"]

# 요일 맵핑
day_names = {0: '월', 1: '화', 2: '수', 3: '목', 4: '금', 5: '토', 6: '일'}

for mmdd in dates:
    # 입력 파일 경로 및 날짜 처리
    date_str = f"2025{mmdd}"
    input_file = input_dir / f"노선·정류장 지표(노선별 혼잡도)_{date_str}.csv"

    # 파일 읽기 (CP949 / EUC-KR 인코딩)
    try:
        df = pd.read_csv(input_file, encoding='cp949')
    except UnicodeDecodeError:
        df = pd.read_csv(input_file, encoding='euc-kr')

    # 필요한 컬럼명을 영문으로 임시 매핑
    # 가정: 원본 컬럼명은 ['노선','기종점','정류장순번','정류장명','04시',...,'03시'] 형태

    # 시간 컬럼 리스트 추출
    time_cols = [col for col in df.columns if col.endswith('시')]

    # Long format 변환
    df_long = df.melt(
        id_vars=['노선', '기종점', '정류장명'],
        value_vars=time_cols,
        var_name='시간',
        value_name='혼잡도'
    )

    # 날짜와 요일 컬럼 추가
    df_long['날짜'] = pd.to_datetime(date_str, format='%Y%m%d')
    df_long['요일'] = df_long['날짜'].dt.weekday.map(day_names)

    # 순서 재정렬
    df_long = df_long[['노선', '기종점', '정류장명', '날짜', '요일', '시간', '혼잡도']]

    # 출력 파일 저장
    output_file = output_dir / f"bus_congestion_{date_str}.csv"
    df_long.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"Processed and saved: {output_file}")
