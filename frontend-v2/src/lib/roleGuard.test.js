import { describe, it, expect } from 'vitest';
import { isOperatorOnlyPath } from './roleGuard';

describe('isOperatorOnlyPath — 역할 라우트 가드 (#264)', () => {
  it.each(['/folders', '/analysis', '/admin', '/admin/tools', '/folders/x'])(
    '운영자 전용: %s', (p) => expect(isOperatorOnlyPath(p)).toBe(true));

  it.each(['/search', '/settings', '/start', '/'])(
    '일반 접근 허용: %s', (p) => expect(isOperatorOnlyPath(p)).toBe(false));
});
