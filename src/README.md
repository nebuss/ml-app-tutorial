## 코드 실행
### sample.py
```
python src/sample.py
```

### streamlit app
```
streamlit run src/streamlit_app/main.py
```

### filter data
주어진 데이터셋에서 필요한 데이터를 선택하고 필터링하여 저장합니다. 자세한 설명은 `python src/filter_data.py --help`를 참고하세요

예시
```
python src/filter_data.py `
    data/filtered_data200 `
    --src-dir data/KoreanFOOD_Detecting `
    --data-type train `
    --classes jjajangmyeon `
    --classes Jjambbong `
    --num-sample 200
```