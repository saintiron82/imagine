/**
 * 검색 결과 클라이언트 페이징 불변식 (IMGV2 #263 회귀 잠금).
 * 전체 풀(all)에서 앞 limit개가 화면에 보이고,
 * limit가 풀 전체를 덮으면 "더 보기"가 사라진다.
 */
export function pageState(all, limit) {
  return { visible: all.slice(0, limit), noMore: limit >= all.length };
}
