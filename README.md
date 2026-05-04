# 대한항공, LG화학, 유가 뉴스 기반 주가 등락 예측 웹앱

대한항공과 LG화학 관련 뉴스, 유가 뉴스를 활용해 주가 등락 방향을 예측하는 프로젝트입니다.  
현재 구조는 `Django 프론트엔드 + FastAPI 백엔드 + scikit-learn 예측 모델`로 구성되어 있습니다.

## 주요 기능

- 대한항공, LG화학, 유가 최신 뉴스 크롤링
- `뉴스 업데이트` 버튼으로 최신 뉴스와 예측 결과 갱신
- 최신 뉴스와 유가 뉴스를 기반으로 내일 대한항공/LG화학 주가 방향 표시
- 학습된 모델의 Accuracy, F1 Score 표시
- 최근 데이터의 실제 등락과 예측 등락 비교

## 프로젝트 구조

```text
.
├─ manage.py
├─ requirements.txt
├─ requirements-optional.txt
├─ dataset/
│  ├─ 대한항공 주가 타겟.csv
│  ├─ LG주가타겟.csv
│  ├─ 대한항공 전처리 (완) 인베딩 (미완).xlsx
│  ├─ 대한항공 + 유가 전처리(완) 인베딩(미완).xlsx
│  ├─ LG화학 전처리(완) 인베딩(미완).xlsx
│  └─ lg화학 + 유가 전처리 (완)  인베딩(미완).xlsx
├─ src/stock_news_prediction/
│  ├─ config.py              # 회사별 데이터 경로/컬럼 설정
│  ├─ dataset.py             # 주가 타겟과 뉴스 데이터 결합
│  ├─ features.py            # 텍스트 정제, 감성 키워드, TF-IDF 피처
│  ├─ train.py               # 모델 학습 및 평가
│  ├─ predict.py             # 저장 모델 예측 확인
│  ├─ crawl_news.py          # 뉴스 크롤링 CLI
│  └─ embed_sentiment.py     # 선택적 BERT 임베딩/감성 피처 생성
├─ web_api/
│  ├─ app.py                 # FastAPI 엔트리포인트
│  ├─ news_service.py        # 최신 뉴스 크롤링
│  └─ prediction_service.py  # 모델 리포트/예측 API
├─ web_frontend/
│  ├─ settings.py
│  ├─ urls.py
│  └─ wsgi.py
├─ dashboard/
│  └─ views.py               # Django 페이지 렌더링
├─ templates/dashboard/
│  └─ index.html
├─ static/dashboard/
│  ├─ app.js
│  └─ styles.css
├─ models/                   # 학습된 모델 파일
├─ outputs/                  # 평가 리포트
└─ scripts/
   └─ run_web.ps1            # Django + FastAPI 동시 실행
```

## 데이터 흐름

1. `dataset/`의 주가 타겟 CSV와 전처리된 뉴스 엑셀을 날짜 기준으로 결합합니다.
2. 뉴스 본문을 정제하고 감성 키워드 점수와 TF-IDF 피처를 생성합니다.
3. 시간순으로 train/test를 나눈 뒤 Logistic Regression 모델을 학습합니다.
4. 학습된 모델과 TF-IDF 벡터라이저를 `models/`에 저장합니다.
5. FastAPI가 최신 뉴스를 크롤링하고, 저장된 모델을 사용해 내일 상승/하락을 예측합니다.
6. Django 웹페이지가 FastAPI를 호출해 뉴스와 예측 결과를 보여줍니다.

## 설치

```bash
pip install -r requirements.txt
```

BERT 임베딩 실험이 필요할 때만 선택 의존성을 설치합니다.

```bash
pip install -r requirements-optional.txt
```

## 모델 학습

기본 모델은 TF-IDF + 감성 키워드 점수 + 최근 종가를 사용합니다.

```bash
python -m src.stock_news_prediction.train --company koreanair --feature-mode tfidf --model logistic
python -m src.stock_news_prediction.train --company lgchem --feature-mode tfidf --model logistic
```

학습이 끝나면 다음 파일이 생성됩니다.

```text
models/koreanair_tfidf_logistic.joblib
models/lgchem_tfidf_logistic.joblib
outputs/koreanair_tfidf_logistic_report.json
outputs/lgchem_tfidf_logistic_report.json
```

## 웹 실행

PowerShell에서 한 번에 실행할 수 있습니다.

```powershell
.\scripts\run_web.ps1
```

또는 터미널 2개에서 각각 실행합니다.

```bash
python -m uvicorn web_api.app:app --host 127.0.0.1 --port 8001
```

```bash
python manage.py runserver 127.0.0.1:8000 --noreload
```

접속 주소:

- 웹페이지: http://127.0.0.1:8000
- FastAPI 문서: http://127.0.0.1:8001/docs

## API

```text
GET /api/health
GET /api/news/latest?limit=5
GET /api/predictions/summary
GET /api/predictions/tomorrow?limit=5
GET /api/predictions/koreanair?limit=10
GET /api/predictions/lgchem?limit=10
```

`/api/predictions/tomorrow`는 최신 대한항공 뉴스 + 최신 LG화학 뉴스 + 최신 유가 뉴스를 크롤링한 뒤, 저장된 모델로 내일 주가 방향을 예측합니다.

## 현재 모델 성능

최근 실행 기준:

| 종목 | Test Accuracy | Test F1 |
| --- | ---: | ---: |
| 대한항공 | 0.5714 | 0.5000 |
| LG화학 | 0.5000 | 0.5679 |

이 모델은 웹 기능 시연과 연구용 프로토타입입니다. 실제 투자 판단에는 사용하지 마세요.

## 문제 해결

웹에서 `데이터 로드 실패`가 보이면 다음을 확인하세요.

1. FastAPI가 실행 중인지 확인합니다: http://127.0.0.1:8001/api/health
2. Django가 실행 중인지 확인합니다: http://127.0.0.1:8000
3. 코드 수정 후에는 서버를 재시작합니다.
4. 브라우저에서 `Ctrl + F5`로 강력 새로고침합니다.

포트가 꼬였을 때는 8000번 Django, 8001번 FastAPI 프로세스를 종료한 뒤 다시 실행하면 됩니다.
