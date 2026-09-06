// 플랜 정의 (서버 단위 연 라이선스 + 좌석·분석기 한도). 가격은 표시용 — 실제 결제는
// Stripe Price(STRIPE_PRICE_* env)로 처리. trial 은 결제 없이 14일 체험.
export const PLANS = [
  {
    id: 'trial', name: '체험', price: '무료', period: '14일',
    seats: 3, analyzers: 2, checkout: false,
    blurb: '설치하고 14일 동안 전부 사용해 보세요.',
    features: ['좌석 3', '분석기 2대', '전체 검색·분석 기능', '카드 등록 불필요'],
  },
  {
    id: 'studio', name: '스튜디오', price: '₩690,000', period: '연',
    seats: 10, analyzers: 5, checkout: true,
    blurb: '소규모 팀의 표준. 설치비 + 연 갱신.',
    features: ['좌석 10', '분석기 5대', '이메일 초대', '외부 접속(터널)', '우선 지원'],
    highlight: true,
  },
  {
    id: 'team', name: '팀', price: '₩1,490,000', period: '연',
    seats: 25, analyzers: 12, checkout: true,
    blurb: '큰 라이브러리·다수 분석기 운영.',
    features: ['좌석 25', '분석기 12대', '스튜디오 전부', '사용현황 리포트'],
  },
]

export const planById = (id) => PLANS.find(p => p.id === id) || null
