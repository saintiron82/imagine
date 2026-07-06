import { describe, it, expect } from 'vitest';
import { pageState } from './paging';

const items = (n) => Array.from({ length: n }, (_, i) => ({ id: i }));

describe('pageState — 검색 페이징 불변식 (#263)', () => {
  it('풀이 페이지보다 크면 앞 20개만 보이고 더 보기가 남는다', () => {
    const { visible, noMore } = pageState(items(25), 20);
    expect(visible).toHaveLength(20);
    expect(noMore).toBe(false);
  });

  it('정확히 20개면 더 보기가 나타나지 않는다 (버그 #263의 트리거 조건)', () => {
    const { visible, noMore } = pageState(items(20), 20);
    expect(visible).toHaveLength(20);
    expect(noMore).toBe(true);
  });

  it('더 보기 진행: limit 40이면 40개 표시, 100개 풀이면 더 보기 유지', () => {
    const { visible, noMore } = pageState(items(100), 40);
    expect(visible).toHaveLength(40);
    expect(noMore).toBe(false);
  });

  it('마지막 페이지: limit가 풀을 넘어서면 전체 표시 + 더 보기 종료', () => {
    const { visible, noMore } = pageState(items(35), 40);
    expect(visible).toHaveLength(35);
    expect(noMore).toBe(true);
  });

  it('빈 결과', () => {
    const { visible, noMore } = pageState([], 20);
    expect(visible).toHaveLength(0);
    expect(noMore).toBe(true);
  });
});
