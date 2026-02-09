"""
배치 처리 성능 벤치마크 스크립트
각 티어별로 5개, 10개, 20개 파일 처리 시간 측정
"""
import subprocess
import time
import json
from pathlib import Path

# 테스트 설정
TIERS = ["standard", "pro", "ultra"]
BATCH_SIZES = [1, 5, 10, 20]
TEST_DIR = Path(r"C:\Users\saint\ImageParser\test_assets")

# 결과 저장
results = {}

print("=" * 60)
print("배치 처리 성능 벤치마크")
print("=" * 60)

# 테스트 파일 확인
test_files = list(TEST_DIR.glob("**/*.psd")) + list(TEST_DIR.glob("**/*.png"))
print(f"\n사용 가능한 테스트 파일: {len(test_files)}개")

if len(test_files) < 20:
    print(f"⚠️  테스트 파일이 부족합니다 (최소 20개 필요, 현재 {len(test_files)}개)")
    # 파일 복사로 20개 만들기
    print("테스트용 파일 복사 중...")
    original_files = test_files.copy()
    idx = 0
    while len(test_files) < 20:
        src = original_files[idx % len(original_files)]
        dst = TEST_DIR / f"copy_{len(test_files)}_{src.name}"
        if not dst.exists():
            import shutil
            shutil.copy2(src, dst)
            test_files.append(dst)
        idx += 1
    print(f"✅ 테스트 파일 준비 완료: {len(test_files)}개")

# 각 티어별로 테스트
for tier in TIERS:
    results[tier] = {}
    print(f"\n{'='*60}")
    print(f"티어: {tier.upper()}")
    print(f"{'='*60}")

    for batch_size in BATCH_SIZES:
        print(f"\n[{tier.upper()}] 배치 크기: {batch_size}개...")

        # 파일 선택
        selected_files = test_files[:batch_size]
        files_json = json.dumps([str(f) for f in selected_files])

        # 처리 시작
        start_time = time.time()

        cmd = [
            "python",
            "backend/pipeline/ingest_engine.py",
            "--files", files_json,
            "--batch-size", str(batch_size)
        ]

        # tier 설정 (환경 변수 대신 config 우선)
        import os
        env = os.environ.copy()
        env['AI_TIER_OVERRIDE'] = tier

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5분 타임아웃
                env=env
            )

            elapsed = time.time() - start_time

            if result.returncode == 0:
                results[tier][batch_size] = {
                    'time': elapsed,
                    'status': 'success',
                    'per_file': elapsed / batch_size
                }
                print(f"  ✅ 완료: {elapsed:.1f}초 (파일당 {elapsed/batch_size:.1f}초)")
            else:
                results[tier][batch_size] = {
                    'status': 'failed',
                    'error': result.stderr[:200]
                }
                print(f"  ❌ 실패: {result.stderr[:100]}")

        except subprocess.TimeoutExpired:
            print(f"  ⏱️  타임아웃 (5분 초과)")
            results[tier][batch_size] = {'status': 'timeout'}

# 결과 저장
output_file = Path("batch_performance_results.json")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\n\n✅ 결과 저장됨: {output_file}")

# 결과 표 출력
print("\n" + "=" * 80)
print("배치 처리 성능 비교 결과")
print("=" * 80)

print("\n티어별 처리 시간 (초)")
print(f"{'Tier':<10} | {'1개':<8} | {'5개':<8} | {'10개':<8} | {'20개':<8}")
print("-" * 80)

for tier in TIERS:
    row = f"{tier.upper():<10} |"
    for batch_size in BATCH_SIZES:
        data = results[tier].get(batch_size, {})
        if data.get('status') == 'success':
            row += f" {data['time']:>6.1f}s |"
        else:
            row += f" {'FAIL':<6} |"
    print(row)

print("\n파일당 평균 시간 (초)")
print(f"{'Tier':<10} | {'1개':<8} | {'5개':<8} | {'10개':<8} | {'20개':<8}")
print("-" * 80)

for tier in TIERS:
    row = f"{tier.upper():<10} |"
    for batch_size in BATCH_SIZES:
        data = results[tier].get(batch_size, {})
        if data.get('status') == 'success':
            row += f" {data['per_file']:>6.1f}s |"
        else:
            row += f" {'-':<6} |"
    print(row)
