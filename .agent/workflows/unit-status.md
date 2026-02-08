---
description: 현재 유닛 상태 확인
---
# 유닛 상태 확인

현재 진행 중인 유닛과 전체 진행률을 확인합니다.

## 실행 단계

// turbo
1. task.md 내용 확인
```powershell
type "$env:USERPROFILE\.gemini\antigravity\brain\e7a93919-c95a-4c48-8dd8-e2b23704dc10\task.md"
```

2. 진행 상태 해석:
   - `[ ]`: 미시작
   - `[/]`: 진행 중
   - `[x]`: 완료

3. 다음 작업 결정:
   - 진행 중 유닛이 있으면 → `/unit-start`로 계속
   - 없으면 → 다음 미시작 유닛으로 `/unit-start`
