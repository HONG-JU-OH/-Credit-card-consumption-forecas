from selenium import webdriver # driver 
from selenium.webdriver.edge.service import Service # Edge 서비스
from webdriver_manager.microsoft import EdgeChromiumDriverManager # 엣지드라이버 관리자

from selenium.webdriver.common.by import By # By.NAME
import time 

from urllib.request import urlopen 
from bs4 import BeautifulSoup

import pandas as pd
import os
from urllib.request import urlretrieve # server image save


# 카드 혜택 상세 페이지 url 및 혜택 추출

# 1. driver 객체 생성 
driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()))
   
# 2. 카드 혜택 페이지 열기
url = 'https://card-search.naver.com/list?ptn=1&sortMethod=ri&bizType=CPC'
driver.get(url) 


# 3. 스크롤바 내리기
# 현재 스크롤 높이 계산
last_height = driver.execute_script("return document.body.scrollHeight") 

while True: # 무한반복
    # 브라우저 끝까지 스크롤바 내리기
    driver.execute_script("window.scrollTo(5, document.body.scrollHeight);") 
    
    # 2초 대기 - 화면 스크롤 확인
    time.sleep(2) 
    
    # 화면 갱신된 화면의 스크롤 높이 계산
    new_height = driver.execute_script("return document.body.scrollHeight")

    # 새로 계산한 스크롤 높이와 같으면 stop
    if new_height == last_height: 
        try:              
            driver.find_element(By.CLASS_NAME, "more").click() # [결과 더보기] 버튼 클릭 
        except: 
            break  # [결과 더보기] : 없는 경우 - while문 실행 종료
    last_height = new_height # 새로 계산한 스크롤 높이로 대체


# 4. 카드 혜택 상세페이지 urls 추출 
# 선택자 지정 - locator = CSS_SELECTOR
links = driver.find_elements(By.CSS_SELECTOR, 'a.anchor') 
print('수집 a_tags 개수 =', len(links))

# a태그에서 url만 추출    
urls = []

for a in links :     
    # a태그의 href가 가지고 있는 url 추출 -> urls 리스트에 저장
    urls.append(a.get_attribute('href')) 
    
# 중복 url 삭제       
card_url = list(set(urls)) # 중복 url  삭제 

# 'cardAdId' 포함 url만 추출 : 카드 상세페이지
card_url = [i for i in card_url if 'cardAdId' in i]
print(len(card_url)) # 312


# 5. 카드 이름, 혜택 카테고리, 상세내용 추출
'''
# [방법 1] <카드별 카테고리, 상세혜택 합쳐서>
# 카드 이름, [혜택1, 혜택2, ...], [상세혜택1, 상세혜택2, ...]
c_name = []
category = []
benef = []

for url in card_url:
    byte_data = urlopen(url).read()
    text_data = byte_data.decode("utf-8") # 디코딩  
    html = BeautifulSoup(text_data, 'html.parser') # html source 파싱
    
    name = html.select_one("b.txt").text
    cat = [i.text for i in html.select("b.text")]        
    ben = [i.text for i in html.select("i.desc")]
           
    c_name.append(name)
    category.append(cat)
    benef.append(ben)

data = {'name':c_name, 'category':category, 'benefits':benef}
new_bene = pd.DataFrame(data)


driver.close() # 창 닫기 
'''

# [방법 2] 카드 혜택 저장 df
benefits = pd.DataFrame(columns=['name', 'category', 'benefits', 'img']) 

for i in range(len(card_url)):
    # 카드 상세페이지 접속 및 html 문서 파싱
    url = card_url[i]
    byte_data = urlopen(url).read()
    text_data = byte_data.decode("utf-8") # 디코딩  
    html = BeautifulSoup(text_data, 'html.parser') # html source 파싱
    
    # 카드 이름, 혜택 카테고리, 상세 내용, img 내용 추출
    card_name = html.select_one("b.txt").text
    cat = [i.text for i in html.select("b.text")]
    bene = html.select("i.desc")
    img_tag = html.find("img")  # 이미지 태그 추출 
    img_str = str(img_tag) if img_tag else None 
    
    # 카드이름, 카테고리, 상세내용 row형태로 저장
    rows = []
    
    for category, details in zip(cat, bene):        
        conts = details.contents
        texts = [element for element in conts if element.name != 'br']        
        for detail in texts:
            rows.append({'name': card_name, 'category': category, 'benefits': detail, 'img':img_str}) 
    
    # rows -> df 변환
    df = pd.DataFrame(rows)
    
    # benefits에 df concat  
    benefits = pd.concat([benefits, df], ignore_index=True)


'''
# 7. 카드 혜택 csv 파일 저장
benefits.to_csv('카드혜택.csv', encoding='UTF-8', index=False)
dir(benefits)
'''



