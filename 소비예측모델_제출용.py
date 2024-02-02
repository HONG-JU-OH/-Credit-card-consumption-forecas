# -*- coding: utf-8 -*-
"""
코드 진행 

파일 불러오기
데이터 전처리
ARIMA 모델에 맞게 입력 데이터 형태 변환 : arima_input()
데이터 정상성 확인 : kpss_test()
ARIMA 모델 생성 : arima_model()
"""

import pandas as pd

import itertools
from tqdm import tqdm  # 진행상태 프로그래스바로 출력
import warnings

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import kpss # 데이터 정상성 검증
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib.pyplot as plt


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

##############################
## 1. 파일 불러오기 및 전처리
##############################

# 업종코드 불러오기
code = pd.read_csv(r'\\\final_project/서울시민 카드소비 업종코드.csv')
code.info()
'''
 #   Column           Non-Null Count  Dtype 
---  ------           --------------  ----- 
 0   UPJONG_CD  75 non-null     object
 1   CLASS1      75 non-null     object
 2   CLASS2      75 non-null     object
 3   CLASS3      75 non-null     object
'''

# 소비내역파일(원본데이터) 불러오기
#file = pd.read_csv(r'C:\Study\Python\final_proj\data/카드소비패턴_sample.csv', encoding='euc-kr', header=0)
file = pd.read_csv(r'\\\final_project/random_sample2.csv')

file.info()
'''
Data columns (total 10 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   서울시민업종코드  39999 non-null  object
 1   대분류       39999 non-null  object
 2   중분류       39999 non-null  object
 3   소분류       39999 non-null  object
 4   기준년월      39999 non-null  int64 
 5   고객주소블록코드  39999 non-null  int64 
 6   성별        39999 non-null  object
 7   연령대별      39999 non-null  object
 8   카드이용금액계   39999 non-null  int64 
 9   카드이용건수계   39999 non-null  int64 
'''

# 소비데이터 copy
card = file.copy()

# code 업종코드 ss001 -> SS001 변환
code['UPJONG_CD'] = code['UPJONG_CD'].apply(str.upper)

# 업종코드, 소분류 각각 범주 확인
up_code = sorted(card['서울시민업종코드'].unique())
class3 = code['CLASS3'].unique()

len(up_code) # 75
len(class3) # 75


# 불필요한 컬럼 제거
card.drop(['고객주소블록코드', '대분류', '중분류', '성별'], axis=1, inplace=True)


# 날짜데이터 기간 확인
card['기준년월'].min() # '202101'
card['기준년월'].max() # '202208'

# 2030 추출
card_2030 = card[(card['연령대별'] == '20대') | (card['연령대별'] == '30대') ] # [15496 rows x 9 columns]

# 소비내역파일 불러오기
df = pd.read_csv(r'C:\Study\Python\final_proj\data/통합소비내역5_최종.csv')
''' 
 <통합소비데이터>
 Data columns (total 6 columns):
  #   Column  Non-Null Count  Dtype 
 ---  ------  --------------  ----- 
  0   날짜      3741 non-null   object
  1   금액      3741 non-null   int64 
  2   소분류     3741 non-null   object
  3   업종코드    3741 non-null   object
  4   대분류     3741 non-null   object
  5   중분류     3741 non-null   object
'''

# 날짜데이터 기간 확인
df['기준년월'].min() # '202101'
df['기준년월'].max() # '202201'

# 불필요 컬럼 삭제 : ['업종코드', '대분류', '중분류']
our_data = df.copy()

our_data.drop(['업종코드', '대분류', '중분류'], axis=1, inplace=True)

# 날짜변환 2024.1.9 -> 202401
our_data['년월'] = pd.to_datetime(df['날짜'])
our_data['날짜'] = our_data['년월'].apply(lambda x: x.year*100 + x.month)

our_data.drop('년월', axis=1, inplace=True)

# 금액 절대값 취하기
our_data['금액'] = our_data['금액'].apply(abs)

# 원본데이터 + 소비데이터
concat_data = pd.concat([card_2030, our_data])

# 소분류, 기준년월에 따라 그룹화 및 sum()
grouped_data = concat_data.groupby(['서울시민업종코드', '기준년월']).sum('카드이용금액계').reset_index()
grouped_data = grouped_data.sort_values('카드이용금액계', ascending=False) # [1301 rows x 4 columns]

# 카드이용건수계 많은 순서대로 top10 선정
n_grp = df.groupby('서울시민업종코드').sum('카드이용건수계').reset_index()
grp_top10 = n_grp.sort_values('카드이용건수계', ascending=False).iloc[:10, 0]
'''
65         편의점
68    할인점/슈퍼마켓
40       온라인거래
7         기타요식
3     결제대행(PG)
66          한식
62          통신
61       커피전문점
10     기타음/식료품
5           교통
'''


##############################
## 2. ARIMA input data 생성
##############################
'''
ARIMA : 시계열 값만 포함하는 1차원 배열, 인덱스는 DatetimeIndex
        -> series, np.array
'''
# 날짜 기준 데이터
date = pd.Series(sorted(concat_data['기준년월'].unique()))
date.name = '기준년월'
len(date) # 20
date.dtype

import numpy as np

# arima input data
def arima_input(cat, y) : # cat : grp_top5, y : 카드이용금액계(AMT_CORR)/카드이용건수계(USECT_CORR)

    # category subset 생성
    grp_data = grouped_data[grouped_data['서울시민업종코드'] == cat]
    
    #grp_data[grp_data['카드이용금액계'] == grp_data['카드이용금액계'].max()] = np.mean(grp_data['카드이용금액계'])
    
    # 날짜 오름차순으로 정렬
    grp_data = grp_data.sort_values('기준년월')
    
    # 카테고리별 target 날짜 기준 데이터랑 merge
    merge_data = grp_data.merge(date, how='right', on='기준년월')
    
    # datetimeindex로 변환
    merge_data.index = pd.to_datetime(merge_data['기준년월'], format='%Y%m')

    # 시계열 데이터만 남기기 
    arima_data = merge_data[y]
    
    # 카테고리별 subset에 관측기간 중 빈 날짜 존재 시 데이터 채우기
    if arima_data.isna().sum() > 0:
        arima_data = arima_data.interpolate(method='time')
        
    '''
    interpolate()

    결측치 보간(interpolation)은 결측치가 발생한 위치의 앞뒤 데이터를 사용하여 적절한 값을 추정하는 방법 
    선형 보간, 시간 보간, 다항 보간 등 다양한 방법이 있으며, 상황에 따라 적절한 방법을 선택해야 한다.
    결측치가 많은 경우나 결측치가 연속으로 발생하는 경우, 보간을 통해 추정한 값이 실제 값과 크게 달라질 수 있.
    '''
    
    # 데이터 채운 후에도 null값 존재 시 0으로 대체 
    if arima_data.isna().sum() > 0 :
        arima_data.fillna(0, inplace=True)
        
    plt.figure(figsize=(12,4))
    plt.xlabel('날짜')
    plt.ylabel('카드이용 금액')
    plt.title(f'{cat} 시계열 데이터')
    arima_data.plot()
       
    
    if '/' in cat:
        cat = cat.replace('/', '_')
        
    plt.savefig(r'\\\final_project\시각화/'+f'{cat}_원본데이터.png')
    plt.show()
    
    return arima_data

#arima1 = arima_input('편의점', '카드이용금액계')


##############################
## 4.  정상성 확인          
##############################         
'''
귀무가설: 해당 시계열은 정상(stationary) 시계열이다.
대립가설: 해당 시계열은 비정상(non-stationary) 시계열이다.

p-value <= 0.05 : 귀무가설 기각/ 대립가설 채택 -> 비정상(non-stationary) 시계열.
p-value > 0.05 : 귀무가설 채택/ 대립가설 기각 -> 정상(stationary) 시계열.
'''
def kpss_test(series, cat):

    stats, p_value, nlags, critical_values = kpss(series)
    
    diff = 0
    # 비정상 시계열일 경우 차분
    while p_value <= 0.05:
        series = series.diff(periods=1).iloc[1:]
        stats, p_value, nlags, critical_values = kpss(series)
        
        print(f"비정상(non-stationary) 시계열 데이터 입니다. 차수: {diff}")
        
        diff += 1

    print(f'KPSS Stat: {stats:.5f}')
    print(f'p-value: {p_value:.2f}')
    print(f'Lags: {nlags}')
    print('검증결과: 정상(stationary) 시계열 데이터 입니다.')
    
    # 한글 지원 : 폰트 설정 
    plt.rcParams['font.family'] = 'Malgun Gothic'
    
    # 마이너스 지원
    plt.rcParams['axes.unicode_minus'] = False

    # 차수 확인
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(12, 4)


    # ACF Plot
    plot_acf(series, lags=15, ax=axes[0])
    # 급격히 값이 감소하여 0과 가까워지는 지점 : 2 -> MA(0)

    # PACF Plot
    plot_pacf(series, lags=8, zero=False, ax=axes[1])
    # 급격히 값이 감소하여 0과 가까워지는 지점 : 4 -> AR(2)

    for ax in axes:
        ax.set_ylim(-0.5, 1.25)
        
    plt.show()  
    
    # 스케일링 - 진행한 후 모델 성능이 더 떨어지는 것으로 확인 : 스케일링 생략
    #scaled_series = (series - series.min()) / (series.max() - series.min())
    
    train = series[:-3]
    test = series[-3:]
    
    
    # 시계열 데이터 시각화
    plt.figure(figsize=(12,4))
    plt.xlabel('날짜')
    plt.ylabel('카드이용 금액')
    plt.title(f'{cat} 시계열 데이터')
    series.plot()
    plt.show()
    
    return series, train, test

#input_data, train, test = kpss_test(arima1, '편의점')


##############################
## 5. arima model
##############################
'''
ARIMA 모델의 freq 인자는 주로 DatetimeIndex에서 활용되기 때문에 
문자열 형태의 인덱스를 DatetimeIndex로 변경해야 함
'''

def arima_model(series, cat, s): # series:input data, cat:카테고리, s:예측기간
    
    warnings.filterwarnings('ignore')
    
    # 차수 관련 best parameter 찾기 
    p = range(0, 4)
    d = range(0, 3)
    q = range(0, 5)
    
    pdq = list(itertools.product(p,d,q))
    
    aic = []
    params = []

        
    with tqdm(total=len(pdq)) as pg:
        for i in pdq:
            pg.update(1)
            try:
                model = ARIMA(series, order=i).fit()
                aic.append(round(model.aic, 2))
                params.append(i)
            except Exception as e:
                print(f"Error for {i}: {e}")
                continue
    
    optimal = [(params[i],j) for i,j in enumerate(aic) if j == min(aic)]
    
    if series.index.dtype != 'period[M]':
        series.index = series.index.to_period('M')
    
    arima = ARIMA(series, order=optimal[0][0], freq='M')
    arima_fit = arima.fit()
    
    # 향후 3달에 대한 예측치 생성
    forecast = arima_fit.forecast(steps=s)
    
    # 시각화
    series.plot(label='real')
    forecast.plot(color='red', linestyle='--', label='predict')
    plt.title(f'{cat} 시계열 데이터 향후 3개월 소비금액 예측')
    plt.xlabel('날짜')
    plt.ylabel('카드이용 금액')
    
    if '/' in cat:
        cat = cat.replace('/', '_')
        
    plt.savefig(r'\\final_project\시각화/'+f'{cat}_예측데이터.png')
    plt.show()
    
    return arima, arima_fit, forecast

#arima, arima_fit, forecast = arima_model(train, '편의점', 3)

# 예측값 저장
pred_y = pd.DataFrame()

# 카테고리 저장
keys = []
pred_y = []
       
import pickle

for i in grp_top10:
    
    arima_data = arima_input(i, '카드이용금액계')
    input_data, train, test = kpss_test(arima_data, i)
    arima, arima_fit, forecast = arima_model(train, i, 3)
    print(arima_fit.summary())
    
    if '/' in i:
        i = i.replace('/', '_')
        
    with open(f'{i}_arima.pkl', 'wb') as model_file:
        pickle.dump(arima_fit, model_file)
    

    keys.append(i)
    pred_y.append(forecast)

'''
# 예측값 저장
result = pd.DataFrame(pred_y)
result.to_csv(r'\final_project/pred_y.csv')
'''