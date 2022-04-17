# Visualization Manual
## 환경 설정

1. 사용하고자 하는 font를 다운 받는다.
2. PIL 모듈 ImageFont.py 861 line 에 dirs에 font_path를 추가한다.

## argment 설명

### 1. instant mode
True -> 앱을 실행하면 바로 실행 되지만, page를 변경할 때 마다 image를 reload한다.
False -> 앱을 실행시킬 때 모든 image를 load하여 일단 실행 시키면 다음 page가 금방 나타난다.

### 2. row, column
한 page에 나타나는 image의 layout

### 3. img_height
resize 되는 image의 높이, 비율 유지
