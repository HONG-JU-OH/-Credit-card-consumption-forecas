#############################################################################
### 1. vec1 만들기 ###############################################################
#############################################################################

import pandas as pd
import numpy as np

# 소비내역 불러오기
csv_file_path = r'C:\Study\Python\final_proj\data/통합소비내역5_최종.csv'
file = pd.read_csv(csv_file_path) 

consump = file.copy() # 카피데이터

# 원본데이터에서 추출한 소비건수 기준 top10 카테고리 적용
top10 = ['편의점', '할인점/슈퍼마켓', '온라인거래', '기타요식', '결제대행(PG)', '한식', '통신', '커피전문점', '기타음/식료품', '교통']
consump = consump[consump['소분류'].isin(top10)]

# 카드혜택 카테고리 & 소비 카테고리 일치시키기
change = {'카페/베이커리': ['커피전문점', '제과점'],
        '쇼핑': ['생활잡화', '의복/의류', '패션/잡화','백화점'],
        '대중교통': ['교통'],
        '대형마트': ['할인점/슈퍼마켓', '슈퍼마켓', '기타음/식료품'],
        '의료': ['일반병원', '치과병원', '종합병원', '약국'],
        '영화': ['영화/공연'],
        '뷰티': ['화장품', '미용서비스'],
        '외식': ['한식', '일식', '양식', '중식', '패스트푸드', '기타요식'],
        '간편결제': ['결제대행(PG)', '온라인거래']}


# 소비 카테고리 변경
for c in consump.index:
    for i, j in change.items():
        if consump.loc[c,'소분류'] in j:
            consump.loc[c,'소분류'] = i
                    
consump['소분류'].unique()
# ['간편결제', '외식', '대중교통', '편의점', '대형마트', '카페/베이커리', '통신']

# 소분류 빈도 확인
dt_cnt = consump['소분류'].value_counts().sort_index()
'''
간편결제       506
대중교통       561
대형마트        83
외식         570
카페/베이커리    478
통신          41
편의점        356
'''


# 소분류별 금액 합계 구하기
consump['금액'] = consump['금액'].apply(abs)
tot_price = consump.groupby('소분류')['금액'].sum().sort_index()
'''
간편결제       10991632
대중교통        6542023
대형마트        2153300
외식          8904517
카페/베이커리     3237270
통신          1727050
편의점         1457940
'''


# 소분류별 결제 건수, 금액 총합 데이터프레임
top_df = pd.DataFrame({'count':dt_cnt, 'tot_price':tot_price})

# 소분류 빈도, 금액 총합으로 정렬
sorted_top = top_df.sort_values(by=['count', 'tot_price'], ascending=False)
'''
         count  tot_price
소분류                      
외식         570    8904517
대중교통       561    6542023
간편결제       506   10991632
카페/베이커리    478    3237270
편의점        356    1457940
대형마트        83    2153300
통신          41    1727050
'''      
    
#----------------------------------------------------------------------------
# 카드 데이터 불러오기
card_df = pd.read_csv(r'C:\Study\Python\final_proj\data/카드혜택_2.csv')

# 월별 업종빈도의 가중치(CF weight) : 1 + log(업종빈도)
cf_w = [np.log(i) for i in sorted_top['count']]

# 상품빈도
cnt = []

for i in sorted_top.index:
    n = 0
    for j in card_df['category'] : 
        if i in j :
            n += 1
    cnt.append(n)
    

# 역상품 빈도값(IGF) : log(카드 상품 수/상품 빈도)
# 상품 빈도가 0일 경우 IFG = 0 대체
IGF = [0 if i == 0 else np.log(312/ i) for i in cnt]

# 가중치 값(weight) : CFweight * IGF
w = np.array(IGF) * np.array(cf_w)

# 쿼리길이 구하기 sqrt((weight1)^2 + (weight2)^2 + ...)
query_len = np.sqrt(sum(w**2)) # 16.86475823021001

# 정규화값
norm = w / query_len

# vec1 생성
vec1 = pd.DataFrame({'cf_w':cf_w, 'IGF':IGF, 'w':w, 'norm':norm})


#############################################################################
### 2. vec2 만들기 ###############################################################
#############################################################################
# 카드 테이블 (벡터테이블2)
c_name = card_df['name']

# 카드 벡터테이블 dict형 저장 - key:카드이름, values:벡터테이블
vec2 = {}

# 모든 카드에 대한 벡터테이블 생성
for n in range(len(card_df)):    
    
    # 카드 하나에 대한 cf_w 값 생성 : 카테고리 존재 1, 미존재 0
    rows = []
    
    for t in sorted_top.index:         
        if t in card_df['category'][n]:
            rows.append({'cat' : t, 'cf_w' : 1})                
        else :
            rows.append({'cat' : t, 'cf_w' : 0})
    
    df = pd.DataFrame(rows)
            
    # 쿼리길이 = sqrt(sum(w1^2 + ... wn^2))
    query_len2 = np.sqrt(sum([i ** 2 for i in df['cf_w']]))
    
    # 정규화 값(normalization) : 가중치/쿼리길이
    df['norm'] = [i/query_len2 for i in df['cf_w']]
    
    # norm값이 NaN일 경우 0으로 대체
    df['norm'].fillna(0, inplace=True)
     
    vec2[c_name[n]] = df
    
    
# 유사도 값 구하기
sim = []

for i in range(len(c_name)):  
    norm2 = np.array(vec2[c_name[i]]['norm'])
    sim.append(sum(norm * norm2))
    
simm = pd.DataFrame({'simm':sim})    

simm['c_name'] = c_name

# 상위 5개 카드 추천
result = simm.sort_values('simm', ascending=False).head(5)

print(result)
'''
         simm           c_name
174  0.935823   KB국민 청춘대로 톡톡카드
305  0.935823        스타벅스 현대카드
24   0.927277     KB국민 톡톡Pay카드
23   0.884731  NH농협 올바른FLEX 카드
262  0.884731   KB국민 알뜰교통플러스카드
'''


