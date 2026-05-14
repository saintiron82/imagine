# Blind Random QuerySet V1

이미지, 캡션, 태그를 보지 않고 고정 조건 사전과 랜덤 시드로 만든 검색 질문지입니다.
scope가 있는 경우에도 DB 경로명/파일 수만 사용하며 이미지 내용은 보지 않습니다.

## blind-random-q0001 [easy]

다음 요소가 함께 보이는 배경: 가로등, 책장 (야외, 실내 분위기)

- scope: -
- must: 가로등, 책장
- soft: 야외, 실내

## blind-random-q0002 [hard]

다음 요소가 함께 보이고 하나 이상은 뚜렷한 배경: 강, 푸른 하늘, 책장 (학교 분위기) / 제외 조건: 텍스트 중심

- scope: -
- must: 강, 푸른 하늘, 책장
- soft: 학교
- exclude: 텍스트 중심

## blind-random-q0003 [medium]

다음 요소가 함께 보이고 하나 이상은 뚜렷한 배경: 난간, 돌기둥, 가로등 (주거지, 상점가 분위기)

- scope: -
- must: 돌기둥, 가로등, 난간
- soft: 주거지, 상점가

## blind-random-q0004 [easy]

다음 요소가 함께 보이는 배경: 책상, 숲 (도시 분위기)

- scope: -
- must: 책상, 숲
- soft: 도시

## blind-random-q0005 [easy]

다음 요소가 함께 보이는 배경: 푸른 하늘, 거리 조명 (주거지 분위기)

- scope: -
- must: 거리 조명, 푸른 하늘
- soft: 주거지

## blind-random-q0006 [hard]

다음 요소가 한 화면에 함께 보이는 배경: 별, 난간, 돌기둥, 숲 (비 오는, 마을 분위기) / 제외 조건: 소품 단독

- scope: -
- must: 별, 숲, 난간, 돌기둥
- soft: 비 오는, 마을
- exclude: 소품 단독

## blind-random-q0007 [hard]

다음 요소가 함께 보이고 하나 이상은 뚜렷한 배경: 푸른 하늘, 돌기둥, 골목 (실내, 현대 건축 분위기) / 제외 조건: 소품 단독

- scope: -
- must: 푸른 하늘, 골목, 돌기둥
- soft: 실내, 현대 건축
- exclude: 소품 단독

## blind-random-q0008 [hard]

다음 요소가 함께 보이고 하나 이상은 뚜렷한 배경: 푸른 하늘, 비, 난간 (학교 분위기) / 제외 조건: 인물, 전투 장면

- scope: -
- must: 푸른 하늘, 난간, 비
- soft: 학교
- exclude: 인물, 전투 장면

## blind-random-q0009 [hard]

다음 요소가 함께 보이고 하나 이상은 뚜렷한 배경: 바다, 전선, 폐허 (실내, 상점가 분위기) / 제외 조건: 추상 텍스처, 소품 단독

- scope: -
- must: 바다, 폐허, 전선
- soft: 실내, 상점가
- exclude: 추상 텍스처, 소품 단독

## blind-random-q0010 [hard]

다음 요소가 함께 보이고 하나 이상은 뚜렷한 배경: 전선, 도로, 별 (야외 분위기) / 제외 조건: 텍스트 중심

- scope: -
- must: 도로, 전선, 별
- soft: 야외
- exclude: 텍스트 중심

## blind-random-q0011 [easy]

다음 요소가 함께 보이는 배경: 다리, 안개 (학교, 마을 분위기) / 제외 조건: 전투 장면

- scope: -
- must: 다리, 안개
- soft: 학교, 마을
- exclude: 전투 장면

## blind-random-q0012 [hard]

다음 요소가 함께 보이고 하나 이상은 뚜렷한 배경: 노을, 선반, 책상 (야외, 밝은 낮 분위기) / 제외 조건: 근접 얼굴, 전투 장면

- scope: -
- must: 선반, 노을, 책상
- soft: 야외, 밝은 낮
- exclude: 근접 얼굴, 전투 장면

## blind-random-q0013 [easy]

다음 요소가 함께 보이는 배경: 커튼, 동굴 벽 (현대 건축 분위기) / 제외 조건: 추상 텍스처

- scope: -
- must: 커튼, 동굴 벽
- soft: 현대 건축
- exclude: 추상 텍스처

## blind-random-q0014 [hard]

다음 요소가 함께 보이고 하나 이상은 뚜렷한 배경: 커튼, 푸른 하늘, 숲 (야외 분위기) / 제외 조건: 근접 얼굴

- scope: -
- must: 커튼, 푸른 하늘, 숲
- soft: 야외
- exclude: 근접 얼굴

## blind-random-q0015 [hard]

다음 요소가 함께 보이고 하나 이상은 뚜렷한 배경: 난간, 노을, 가로등 / 제외 조건: 텍스트 중심

- scope: -
- must: 노을, 난간, 가로등
- exclude: 텍스트 중심

## blind-random-q0016 [hard]

다음 요소가 함께 보이고 하나 이상은 뚜렷한 배경: 책장, 가로등, 침대 / 제외 조건: 소품 단독

- scope: -
- must: 책장, 침대, 가로등
- exclude: 소품 단독

## blind-random-q0017 [easy]

다음 요소가 함께 보이는 배경: 창문, 책장 (현대 건축, 야외 분위기)

- scope: -
- must: 책장, 창문
- soft: 현대 건축, 야외

## blind-random-q0018 [medium]

다음 요소가 한 화면에 함께 보이는 배경: 벽돌 벽, 노을, 나무, 돌기둥 (상점가, 현대 건축 분위기)

- scope: -
- must: 나무, 돌기둥, 벽돌 벽, 노을
- soft: 상점가, 현대 건축

## blind-random-q0019 [easy]

다음 요소가 함께 보이는 배경: 숲, 폐허 / 제외 조건: 소품 단독

- scope: -
- must: 숲, 폐허
- exclude: 소품 단독

## blind-random-q0020 [easy]

다음 요소가 함께 보이는 배경: 계단, 달 (자연, 밝은 낮 분위기) / 제외 조건: 근접 얼굴

- scope: -
- must: 달, 계단
- soft: 자연, 밝은 낮
- exclude: 근접 얼굴

## blind-random-q0021 [easy]

다음 요소가 함께 보이는 배경: 돌기둥, 전선 (조용한 분위기) / 제외 조건: 인물, 추상 텍스처

- scope: -
- must: 전선, 돌기둥
- soft: 조용한
- exclude: 인물, 추상 텍스처

## blind-random-q0022 [hard]

다음 요소가 한 화면에 함께 보이는 배경: 비, 도로, 벽돌 벽, 밤하늘 (산업 시설 분위기) / 제외 조건: 추상 텍스처

- scope: -
- must: 벽돌 벽, 비, 밤하늘, 도로
- soft: 산업 시설
- exclude: 추상 텍스처

## blind-random-q0023 [easy]

다음 요소가 함께 보이는 배경: 밤하늘, 전선 (현대 건축 분위기) / 제외 조건: 전투 장면, 캐릭터

- scope: -
- must: 밤하늘, 전선
- soft: 현대 건축
- exclude: 전투 장면, 캐릭터

## blind-random-q0024 [easy]

다음 요소가 함께 보이는 배경: 비, 계단 (밝은 낮 분위기) / 제외 조건: 텍스트 중심

- scope: -
- must: 비, 계단
- soft: 밝은 낮
- exclude: 텍스트 중심

## blind-random-q0025 [easy]

다음 요소가 함께 보이는 배경: 푸른 하늘, 돌기둥 (마을 분위기)

- scope: -
- must: 푸른 하늘, 돌기둥
- soft: 마을

## blind-random-q0026 [medium]

다음 요소가 함께 보이고 하나 이상은 뚜렷한 배경: 별, 책장, 커튼 (도시 분위기)

- scope: -
- must: 커튼, 별, 책장
- soft: 도시

## blind-random-q0027 [easy]

다음 요소가 함께 보이는 배경: 비, 침대 (야외, 도시 분위기) / 제외 조건: 전투 장면

- scope: -
- must: 침대, 비
- soft: 야외, 도시
- exclude: 전투 장면

## blind-random-q0028 [medium]

다음 요소가 함께 보이고 하나 이상은 뚜렷한 배경: 커튼, 골목, 창문

- scope: -
- must: 커튼, 창문, 골목

## blind-random-q0029 [hard]

다음 요소가 한 화면에 함께 보이는 배경: 나무, 노을, 눈, 벽돌 벽 (현대 건축, 야외 분위기) / 제외 조건: 캐릭터, 추상 텍스처

- scope: -
- must: 나무, 눈, 노을, 벽돌 벽
- soft: 현대 건축, 야외
- exclude: 캐릭터, 추상 텍스처

## blind-random-q0030 [easy]

다음 요소가 함께 보이는 배경: 벽돌 벽, 나무 (상점가, 학교 분위기) / 제외 조건: 소품 단독

- scope: -
- must: 나무, 벽돌 벽
- soft: 상점가, 학교
- exclude: 소품 단독

## blind-random-q0031 [easy]

다음 요소가 함께 보이는 배경: 바다, 벽돌 벽 / 제외 조건: 인물

- scope: -
- must: 바다, 벽돌 벽
- exclude: 인물

## blind-random-q0032 [hard]

다음 요소가 한 화면에 함께 보이는 배경: 계단, 노을, 숲, 커튼 (상점가, 전통 건축 분위기) / 제외 조건: 소품 단독

- scope: -
- must: 커튼, 노을, 계단, 숲
- soft: 상점가, 전통 건축
- exclude: 소품 단독

## blind-random-q0033 [hard]

다음 요소가 함께 보이고 하나 이상은 뚜렷한 배경: 소파, 비, 커튼 (비 오는 분위기) / 제외 조건: 캐릭터

- scope: -
- must: 소파, 커튼, 비
- soft: 비 오는
- exclude: 캐릭터

## blind-random-q0034 [hard]

다음 요소가 함께 보이고 하나 이상은 뚜렷한 배경: 돌기둥, 폐허, 구름 (마을, 야외 분위기) / 제외 조건: 추상 텍스처

- scope: -
- must: 구름, 돌기둥, 폐허
- soft: 마을, 야외
- exclude: 추상 텍스처

## blind-random-q0035 [hard]

다음 요소가 한 화면에 함께 보이는 배경: 폐허, 골목, 돌기둥, 간판 (산업 시설, 밝은 낮 분위기) / 제외 조건: 전투 장면

- scope: -
- must: 폐허, 돌기둥, 간판, 골목
- soft: 산업 시설, 밝은 낮
- exclude: 전투 장면

## blind-random-q0036 [hard]

다음 요소가 함께 보이고 하나 이상은 뚜렷한 배경: 전선, 노을, 구름 (밝은 낮 분위기) / 제외 조건: 텍스트 중심

- scope: -
- must: 전선, 노을, 구름
- soft: 밝은 낮
- exclude: 텍스트 중심

## blind-random-q0037 [hard]

다음 요소가 함께 보이고 하나 이상은 뚜렷한 배경: 창문, 난간, 도로 (전통 건축 분위기) / 제외 조건: 근접 얼굴, 소품 단독

- scope: -
- must: 도로, 창문, 난간
- soft: 전통 건축
- exclude: 근접 얼굴, 소품 단독

## blind-random-q0038 [easy]

다음 요소가 함께 보이는 배경: 비, 선반 / 제외 조건: 텍스트 중심

- scope: -
- must: 비, 선반
- exclude: 텍스트 중심

## blind-random-q0039 [easy]

다음 요소가 함께 보이는 배경: 비, 노을 (전통 건축 분위기) / 제외 조건: 캐릭터

- scope: -
- must: 비, 노을
- soft: 전통 건축
- exclude: 캐릭터

## blind-random-q0040 [hard]

다음 요소가 함께 보이고 하나 이상은 뚜렷한 배경: 눈, 골목, 밤하늘 (자연, 상점가 분위기) / 제외 조건: 추상 텍스처

- scope: -
- must: 골목, 밤하늘, 눈
- soft: 자연, 상점가
- exclude: 추상 텍스처
