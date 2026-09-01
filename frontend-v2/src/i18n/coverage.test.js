import { describe, it, expect } from 'vitest';
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

/**
 * i18n 사용률 잠금.
 *
 * parity.test.js 는 en-US ↔ ko-KR 의 키 집합이 같은지만 본다. 그것만으로는
 * "번역돼 있다"가 보장되지 않는다 — 실제로 이 앱은 키를 1040개 갖고도 셸과
 * 6개 화면이 전부 한글 하드코딩이라, 기본 로케일(en-US) 사용자에게 검색을
 * 제외한 모든 화면이 한국어로 보였다. 패리티는 통과하는데 앱은 번역돼
 * 있지 않은 상태였다.
 *
 * 이 테스트는 그 구멍을 막는다: UI 파일에서 사용자에게 보이는 위치의
 * 한글 리터럴을 잡는다. 주석은 대상이 아니다(코드 주석은 한국어로 쓴다).
 */

const SRC = join(fileURLToPath(new URL('../', import.meta.url)));

// UI 를 렌더하거나 표시 문자열을 만드는 곳. 나머지(테스트, 로케일 JSON 등)는 제외.
const UI_DIRS = ['shell', 'screens', 'flows', 'components', 'state', 'api'];

function walk(dir) {
  const out = [];
  for (const entry of readdirSync(dir)) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) out.push(...walk(full));
    else if (/\.jsx?$/.test(entry) && !/\.test\.jsx?$/.test(entry)) out.push(full);
  }
  return out;
}

/** 주석(라인·블록)과 import 경로를 제거해 "코드 본문"만 남긴다. */
function stripComments(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, '')      // /* ... */ 및 JSDoc
    .replace(/\/\/[^\n]*/g, '')            // // ...
    .replace(/\{\s*\/\*[\s\S]*?\*\/\s*\}/g, ''); // {/* JSX 주석 */}
}

const HANGUL = /[가-힣]/;

// 언어 선택기는 각 언어를 자기 이름으로 표시한다 — 번역 대상이 아니다.
const ALLOWED = [/'ko-KR':\s*'한국어'/];

describe('i18n 사용률 — 하드코딩된 한글 UI 문자열이 없어야 한다', () => {
  const files = UI_DIRS
    .flatMap(d => { try { return walk(join(SRC, d)); } catch { return []; } });

  it('검사 대상 파일이 실제로 존재한다 (경로 오타로 공허하게 통과하지 않도록)', () => {
    expect(files.length).toBeGreaterThan(10);
  });

  for (const file of files) {
    const rel = relative(SRC, file);
    it(`${rel} 에 하드코딩된 한글이 없다`, () => {
      const body = stripComments(readFileSync(file, 'utf8'));
      const offenders = body
        .split('\n')
        .map((line, i) => [i + 1, line])
        .filter(([, line]) => HANGUL.test(line))
        .filter(([, line]) => !ALLOWED.some(re => re.test(line)))
        .map(([n, line]) => `  ${rel}:${n}  ${line.trim().slice(0, 100)}`);

      expect(
        offenders,
        `하드코딩된 한글 UI 문자열 — t('...') 로 바꾸고 두 로케일에 키를 추가할 것:\n${offenders.join('\n')}`,
      ).toEqual([]);
    });
  }
});
