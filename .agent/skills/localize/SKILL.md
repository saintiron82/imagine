---
description: 프론트엔드 로컬라이제이션 관리 에이전트 (하드코딩 탐지, 키 생성, 번역 파일 동기화)
---
# Localization Agent

## Overview
프론트엔드 UI 문자열의 로컬라이제이션을 관리하는 에이전트입니다.
하드코딩된 문자열을 탐지하고, i18n 키로 치환하며, 번역 파일 동기화를 보장합니다.

## Capabilities

### 1. Hardcoded String Scan
JSX/TSX 파일에서 하드코딩된 UI 문자열을 탐지합니다.

**탐지 대상:**
- JSX 텍스트 노드: `<span>Hello</span>`
- 속성 문자열: `placeholder="Enter text..."`, `title="Click me"`
- 템플릿 리터럴 내 텍스트: `` {`${count} items`} ``
- 조건부 텍스트: `{isOpen ? 'Close' : 'Open'}`

**제외 대상:**
- CSS 클래스명 (`className="..."`)
- 데이터 속성 (`data-*`)
- 이벤트 핸들러 (`onClick={...}`)
- import 경로
- 숫자, boolean, 변수명
- 아이콘/이모지 단독 사용

### 2. Key Generation
네이밍 컨벤션에 따라 i18n 키를 자동 생성합니다.

**키 네이밍 컨벤션:**

| Prefix | Usage | Example |
|--------|-------|---------|
| `app.` | App-level / global | `app.title` |
| `tab.` | Tab names | `tab.search`, `tab.archive` |
| `action.` | Buttons / actions | `action.process`, `action.save` |
| `label.` | Form labels | `label.notes`, `label.rating` |
| `placeholder.` | Input hints | `placeholder.search` |
| `status.` | Status display | `status.loading`, `status.no_results` |
| `filter.` | Filter-related | `filter.format`, `filter.category` |
| `settings.` | Settings screen | `settings.title` |
| `meta.` | Metadata modal | `meta.format`, `meta.layers` |
| `category.` | Category options | `category.characters` |
| `msg.` | Alerts / messages | `msg.no_metadata` |
| `lang.` | Language toggle | `lang.original`, `lang.korean` |
| `search.` | Search-related | `search.title`, `search.subtitle` |

### 3. Translation File Sync
`en-US.json`과 `ko-KR.json` 간 키 동기화를 검증합니다.

**검증 항목:**
- 양쪽 파일의 키 개수 일치
- 누락된 키 탐지
- 미사용 키 탐지 (코드에서 참조되지 않는 키)
- 보간 파라미터 일치 (`{count}` 등)

## File Locations

| File | Purpose |
|------|---------|
| `frontend/src/i18n/LocaleContext.jsx` | React Context + useLocale hook |
| `frontend/src/i18n/index.js` | Entry point (re-export) |
| `frontend/src/i18n/locales/en-US.json` | English translations |
| `frontend/src/i18n/locales/ko-KR.json` | Korean translations |

## Usage Pattern

```jsx
import { useLocale } from '../i18n';

const MyComponent = () => {
  const { t } = useLocale();

  return (
    <div>
      <h1>{t('app.title')}</h1>
      <span>{t('status.selected', { count: 5 })}</span>
      <input placeholder={t('placeholder.search')} />
    </div>
  );
};
```

## Rules

1. **No hardcoded UI text** - All user-visible strings must use `t()` calls
2. **Both locales required** - Every key must exist in both en-US.json and ko-KR.json
3. **Flat key structure** - Use dot notation, no nesting (e.g., `app.title`, not `{ app: { title } }`)
4. **Interpolation syntax** - Use `{paramName}` for dynamic values
5. **Missing key fallback** - Returns the key itself + console.warn in dev mode
