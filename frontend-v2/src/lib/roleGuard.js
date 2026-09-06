/**
 * 운영자 전용 경로 판정 — AppShell 역할 가드가 사용 (IMGV2 #264 회귀 잠금).
 * 하위 경로(/admin/... 등)도 전부 가드된다.
 */
export const OPERATOR_ONLY_PATHS = ['/folders', '/analysis', '/admin'];

export function isOperatorOnlyPath(pathname) {
  return OPERATOR_ONLY_PATHS.some(p => pathname.startsWith(p));
}
