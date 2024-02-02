

from selenium import webdriver # driver 
from selenium.webdriver.edge.service import Service # Edge 서비스
from webdriver_manager.microsoft import EdgeChromiumDriverManager # 엣지드라이버 관리자
import time
from selenium.webdriver.common.by import By

# 카드 혜택 페이지 열기
e_driver = webdriver.Edge(service=Service(EdgeChromiumDriverManager().install()))
e_driver.get('https://card-search.naver.com/list?ptn=1&sortMethod=ri&bizType=CPC')


# 스크롤 내려서 더보기 버튼 
last_height = e_driver.execute_script("return document.body.scrollHeight") #현재 스크롤 높이 계산

while True: # 무한반복
    # 브라우저 끝까지 스크롤바 내리기
    e_driver.execute_script("window.scrollTo(5, document.body.scrollHeight);") 
    
    time.sleep(2) # 2초 대기 - 화면 스크롤 확인

    # 화면 갱신된 화면의 스크롤 높이 계산
    new_height = e_driver.execute_script("return document.body.scrollHeight")

    # 새로 계산한 스크롤 높이와 같으면 stop
    if new_height == last_height: 
        try: # [결과 더보기] : 없는 경우 - 예외처리             
            e_driver.find_element(By.CLASS_NAME, "more").click() # [결과 더보기] 버튼 클릭 
        except:
            break
    last_height = new_height # 새로 계산한 스크롤 높이로 대체



















