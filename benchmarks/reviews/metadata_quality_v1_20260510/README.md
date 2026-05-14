# Metadata Quality Review Sample V1

이미지 자체와 기존 AI 캡션/태그의 정합성을 사람이 평가하기 위한 샘플입니다.
이 결과는 이후 answerable 검색 질문셋을 만드는 기반으로 사용합니다.

- sample_count: 500
- candidate_count: 15921
- seed: 20260510
- max_per_source: 8
- include_missing: False
- include_partial: False
- analysis_status_counts: {'legacy_warning': 427, 'ok': 73}

## First Rows

### metadata-quality-v1-0001 / item 57524

- file: 57524_TM#04_077.psd
- source: #4
- status: ok 
- caption: Digital sketch of a building facade with windows and structural lines.
- caption_ko_display: 창문과 구조선이 보이는 건물 외관의 디지털 스케치입니다.
- tags: ["building", "windows", "sketch", "architecture", "blueprint", "lines", "concrete"]
- tags_ko_display: 건물 · 창문 · 스케치 · 건축 · 청사진 · 선 · 콘크리트

### metadata-quality-v1-0002 / item 56909

- file: 56909_KGS_01_163.psd
- source: bg
- status: ok 
- caption: A translucent anime-style man stands in a lush green landscape under a blue sky.
- caption_ko_display: 무성한 초록 풍경 안에 반투명한 애니메이션풍 남성이 서 있고, 푸른 하늘 아래입니다.
- tags: ["anime", "man", "landscape", "trees", "sky", "clouds", "translucent", "kimono"]
- tags_ko_display: 애니풍 · 남성 · 풍경 · 나무 · 하늘 · 구름 · 반투명한 · 기모노

### metadata-quality-v1-0003 / item 59999

- file: 59999_mky13_152.psd
- source: BG
- status: ok 
- caption: Three translucent anime characters in a blue-lit stadium at night.
- caption_ko_display: 밤의 푸른 조명이 비치는 경기장 안에 세 반투명한 애니메이션 캐릭터가 있습니다.
- tags: ["anime characters", "stadium", "night sky", "translucent figures", "blue lighting"]
- tags_ko_display: 애니메이션 캐릭터 · 경기장 · 밤하늘 · 반투명한 인물 · 푸른 조명

### metadata-quality-v1-0004 / item 40801

- file: 40801_grb01_092.psd
- source: webdav:/13730b09/예비/크랑베르무/장소/전투/1
- status: legacy_warning legacy_parse_or_caption_model
- caption: Purple motion blur effect
- caption_ko_display: 보라색 모션 블러 효과입니다.
- tags: ["motion blur", "purple", "dynamic", "high intensity"]
- tags_ko_display: 모션 블러 · 보라색 · 역동적인 · 고강도

### metadata-quality-v1-0005 / item 48312

- file: 48312_VS10_02_033_GENZU.psd
- source: 발송/외주 회사/뱅가드/26/#02/bg
- status: legacy_warning legacy_parse_or_caption_model
- caption: Four anime characters stand before a giant monster under a starry night sky.
- caption_ko_display: 네 애니메이션 캐릭터가 거대한 괴물 앞에 서 있고, 별이 있는 밤하늘 아래입니다.
- tags: ["anime", "monster", "night", "group", "fantasy", "sketch", "horror", "sci-fi"]
- tags_ko_display: 애니풍 · 괴물 · 밤 · 무리 · 판타지 · 스케치 · 공포 · 공상과학

### metadata-quality-v1-0006 / item 49148

- file: 49148_VS9_11_261_GENZU.psd
- source: 발송/외주 회사/뱅가드/25/#11/bg
- status: legacy_warning legacy_parse_or_caption_model
- caption: Anime-style portrait with lava and starry sky background
- caption_ko_display: 애니메이션풍 인물화이며, 용암 및 별이 있는 하늘 배경이/가 보입니다.
- tags: ["anime", "portrait", "lava", "stars", "blue", "orange", "digital art", "overlay"]
- tags_ko_display: 애니풍 · 인물화 · 용암 · 별 · 푸른색 · 주황색 · 디지털 아트 · 오버레이

### metadata-quality-v1-0007 / item 57710

- file: 57710_AT03_160A.psd
- source: #3
- status: ok 
- caption: Anime-style character sketch overlaid on a warm-toned interior scene
- caption_ko_display: 애니메이션풍 캐릭터 스케치가 따뜻한 색조의 실내 장면 위에 겹쳐져 있습니다.
- tags: ["anime character", "sketch overlay", "interior room", "wooden panels", "warm lighting"]
- tags_ko_display: 애니메이션 캐릭터 · 스케치 오버레이 · 실내 · 나무 패널 · 따뜻한

### metadata-quality-v1-0008 / item 56807

- file: 56807_MR07_043_BG.psd
- source: bg
- status: ok 
- caption: Abstract cityscape with geometric buildings and translucent overlays
- caption_ko_display: 추상적인 도시 풍경이며, 기하학적 건물 및 반투명한 오버레이이/가 보입니다.
- tags: ["cityscape", "buildings", "geometric", "translucent", "modern", "abstract", "urban", "digital"]
- tags_ko_display: 도시 풍경 · 건물 · 기하학적 · 반투명한 · 현대적인 · 추상적인 · 도시의 · 디지털

### metadata-quality-v1-0009 / item 39929

- file: 39929_grb09_144.psd
- source: webdav:/13730b09/예비/크랑베르무/장소/전투/5
- status: legacy_warning legacy_parse_or_caption_model
- caption: A translucent, winged mechanical creature perched on a rocky cliff against a misty blue background.
- caption_ko_display: 반투명한, 날개 달린 기계적인 생물이 바위 절벽 위에 앉아 있고, 안개 낀 푸른색 배경을/를 배경으로 합니다.
- tags: ["mechanical", "wings", "anime", "mysterious", "translucent", "cliff", "blue", "green", "rock"]
- tags_ko_display: 기계적인 · 날개 · 애니풍 · 신비로운 · 반투명한 · 절벽 · 푸른색 · 초록 · 바위

### metadata-quality-v1-0010 / item 41180

- file: 41180_grb06_256.psd
- source: webdav:/13730b09/예비/크랑베르무/장소/안나의집
- status: legacy_warning legacy_parse_or_caption_model
- caption: Character sketch of a young female with flowing hair and closed eyes, rendered in soft pastel tones with vertical color distortion effects.
- caption_ko_display: 흐르는 머리카락 및 감은 눈, 표현된 부드러운 파스텔 톤 함께 세로 색상 왜곡 효과이/가 보이는 젊은 여성의 캐릭터 스케치입니다.
- tags: ["grb06_256.psd", "concept art", "line art", "pastel palette", "side profile", "serene expression", "vertical distortion"]
- tags_ko_display: 콘셉트 아트 · 선화 · 파스텔 팔레트 · 옆모습 · 고요한 표정 · 세로 왜곡

### metadata-quality-v1-0011 / item 33653

- file: 33653_nfb11_178_BG1.psd
- source: webdav:/13730b09/발송4/범선/작품 쫑/후시/장소/마법창고
- status: legacy_warning legacy_parse_or_caption_model
- caption: A dimly lit gothic dungeon shelf with purple glowing accents and intricate carvings.
- caption_ko_display: 어둡게 조명된 고딕풍 던전 선반이며, 보라색으로 빛나는 강조 요소 및 정교한 조각이/가 보입니다.
- tags: ["dungeon", "gothic", "anime", "mysterious", "purple glow", "night", "shelf", "carvings"]
- tags_ko_display: 던전 · 고딕풍 · 애니풍 · 신비로운 · 보라색 광채 · 밤 · 선반 · 조각

### metadata-quality-v1-0012 / item 52808

- file: 52808_kya02_057.psd
- source: webdav:/13730b09/발송/범선/기절용사와 암살공주/장소/마을
- status: legacy_warning legacy_parse_or_caption_model
- caption: A wooden notice board with children's drawings in a sunny outdoor setting.
- caption_ko_display: 나무 게시판이며, 어린이의 그림 안에 햇빛 드는 야외 환경이/가 보입니다.
- tags: ["notice board", "children's drawings", "wooden structure", "trees", "stone wall", "roof", "outdoor scene"]
- tags_ko_display: 게시판 · 어린이 그림 · 나무 · 돌 · 지붕 · 야외 장면

### metadata-quality-v1-0013 / item 59879

- file: 59879_mky13_203.psd
- source: BG
- status: ok 
- caption: A hand sketch of a foot with anatomical lines on a blue gradient background.
- caption_ko_display: 해부학적 선 위에 푸른색 그라데이션 배경이/가 보이는 발의 손그림 스케치입니다.
- tags: ["foot", "sketch", "anatomy", "blue", "gradient", "line art"]
- tags_ko_display: 발 · 스케치 · 해부학 · 푸른색 · 그라데이션 · 선화

### metadata-quality-v1-0014 / item 49514

- file: 49514_VS9_07_032_GENZU.psd
- source: 발송/외주 회사/뱅가드/25/#07/bg
- status: legacy_warning legacy_parse_or_caption_model
- caption: Anime-style characters in a street scene with ice cream.
- caption_ko_display: 애니메이션풍 캐릭터 안에 거리 장면이며, 아이스크림이/가 보입니다.
- tags: ["anime", "street", "ice cream", "characters", "sketch", "urban", "casual", "daytime"]
- tags_ko_display: 애니풍 · 거리 · 아이스크림 · 캐릭터 · 스케치 · 도시의 · 일상적인 · 낮

### metadata-quality-v1-0015 / item 37958

- file: 37958_외부-004~4.jpg
- source: webdav:/13730b09/발송작품/작품/켄신
- status: legacy_warning legacy_parse_or_caption_model
- caption: A traditional Japanese thatched house nestled in a snowy autumn landscape, surrounded by trees with red and white foliage, evoking a serene, nostalgic rural atmosphere.
- caption_ko_display: 전통적인 일본식 초가집 자리한 눈 덮인 가을 풍경, 둘러싸인 나무이며, 붉은 및 흰 잎, 느낌을 주는 고요한, 향수를 불러일으키는 시골 분위기이/가 보입니다.
- tags: ["rural", "snow", "autumn", "traditional_japanese_house", "painterly", "peaceful", "thatched_roof", "forest_background", "seasonal_colors"]
- tags_ko_display: 시골 · 눈 · 가을 · 일본 전통 가옥 · 회화풍 · 평온한 · 초가지붕 · 숲 · 계절감 있는 색상

### metadata-quality-v1-0016 / item 30194

- file: dwg03_185_R.psd
- source: 발송/범선/작품 쫑/다윈즈게임/장소/도시 낮
- status: legacy_warning legacy_parse_or_caption_model
- caption: A quiet, empty street in an urban setting bathed in cool twilight light, with muted tones and a solitary signpost indicating a pedestrian crossing.
- caption_ko_display: 조용한, 빈 거리 안에 도시 환경 물든 차가운 황혼빛,이며, 차분한 색조 및 홀로 선 표지판 나타내는 횡단보도이/가 보입니다.
- tags: ["urban", "twilight", "melancholic", "empty_street", "modern_architecture", "cool_palette", "realistic"]
- tags_ko_display: 도시의 · 황혼 · 쓸쓸한 · 빈 거리 · 현대적인 · 차가운 색감 · 사실적인

### metadata-quality-v1-0017 / item 52689

- file: 52689_kya05_225B_245B_BG1.psd
- source: webdav:/13730b09/발송/범선/기절용사와 암살공주/장소/마을
- status: legacy_warning legacy_parse_or_caption_model
- caption: Two characters stand in a moonlit medieval courtyard with stone buildings and glowing windows.
- caption_ko_display: 달빛이 비치는 중세풍 안뜰 안에 두 캐릭터가 서 있고, 석조 건물 및 빛나는 창문이/가 보입니다.
- tags: ["moon", "night sky", "stone buildings", "characters", "windows", "courtyard", "plants"]
- tags_ko_display: 달 · 밤하늘 · 석조 건물 · 캐릭터 · 창문 · 안뜰 · 식물

### metadata-quality-v1-0018 / item 40454

- file: 40454_grb05_211.psd
- source: webdav:/13730b09/예비/크랑베르무/장소/전투/3
- status: legacy_warning legacy_parse_or_caption_model
- caption: A mecha dragon stands in a canyon under a starry sky with aurora-like lights.
- caption_ko_display: 협곡 안에 메카 드래곤이 서 있고, 별이 있는 하늘 아래에 오로라 같은 조명이/가 보입니다.
- tags: ["mecha", "dragon", "canyon", "starry sky", "aurora", "anime", "epic", "mysterious"]
- tags_ko_display: 메카 · 용 · 협곡 · 별이 있는 하늘 · 오로라 · 애니풍 · 장대한 · 신비로운

### metadata-quality-v1-0019 / item 51304

- file: 51304_kya08_098_e.psd
- source: webdav:/13730b09/발송/범선/기절용사와 암살공주/#08/bg
- status: legacy_warning legacy_parse_or_caption_model
- caption: Anime character sketch overlaid on a purple wall with decorative lines.
- caption_ko_display: 애니메이션 캐릭터 스케치가 보라색 벽 함께 장식적인 선 위에 겹쳐져 있습니다.
- tags: ["anime character", "sketch", "wall", "purple", "headwear", "lines", "overlay"]
- tags_ko_display: 애니메이션 캐릭터 · 스케치 · 벽 · 보라색 · 머리 장식 · 선 · 오버레이

### metadata-quality-v1-0020 / item 53350

- file: 53350_kya06_089.psd
- source: webdav:/13730b09/발송/범선/기절용사와 암살공주/#06/bg
- status: legacy_warning legacy_parse_or_caption_model
- caption: A red-tinted illustration of a torso framed by wooden panels.
- caption_ko_display: 나무 패널에 둘러싸인 붉은 색조의 일러스트 의 몸통입니다.
- tags: ["torso", "wood", "red", "sketch", "frame"]
- tags_ko_display: 몸통 · 나무 · 붉은 · 스케치 · 프레임
