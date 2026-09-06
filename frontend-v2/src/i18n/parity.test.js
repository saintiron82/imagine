import { describe, it, expect } from 'vitest';
import en from './locales/en-US.json';
import ko from './locales/ko-KR.json';

// 보호 계약: 두 로케일의 키 집합은 완전히 일치해야 한다 (repairman.adapter.yaml).
describe('i18n 키 패리티 (en-US ↔ ko-KR)', () => {
  it('en에만 있는 키가 없다', () => {
    const koKeys = new Set(Object.keys(ko));
    expect(Object.keys(en).filter(k => !koKeys.has(k))).toEqual([]);
  });

  it('ko에만 있는 키가 없다', () => {
    const enKeys = new Set(Object.keys(en));
    expect(Object.keys(ko).filter(k => !enKeys.has(k))).toEqual([]);
  });
});
