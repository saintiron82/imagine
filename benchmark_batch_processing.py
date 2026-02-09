"""
Batch Processing Performance Benchmark
Measures processing time for 1, 5, 10, 20 files across Standard, Pro, Ultra tiers
"""
import subprocess
import time
import json
import yaml
import shutil
import sys
import io
from pathlib import Path
from datetime import datetime

# Force UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 설정
TEST_DIR = Path(r"C:\Users\saint\ImageParser\test_assets")
CONFIG_FILE = Path(r"C:\Users\saint\ImageParser\config.yaml")
BACKUP_CONFIG = Path(r"C:\Users\saint\ImageParser\config.yaml.backup")
TIERS = ["standard", "pro", "ultra"]
BATCH_SIZES = [1, 5, 10, 20]

# Results saved
results = {
    "timestamp": datetime.now().isoformat(),
    "tiers": {}
}

def backup_config():
    """config.yaml 백업"""
    shutil.copy2(CONFIG_FILE, BACKUP_CONFIG)
    print(f"[OK] Config backed up: {BACKUP_CONFIG}")

def restore_config():
    """config.yaml 복원"""
    shutil.copy2(BACKUP_CONFIG, CONFIG_FILE)
    print(f"[OK] Config restored from backup")

def set_tier(tier_name: str):
    """config.yaml의 tier override 설정"""
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config['ai_mode']['override'] = tier_name

    with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print(f"[OK] Tier set to: {tier_name}")

def get_test_files(count: int):
    """테스트 파일 경로 리스트 반환"""
    all_files = list(TEST_DIR.glob("*.png")) + list(TEST_DIR.glob("*.psd"))
    return [str(f) for f in all_files[:count]]

def run_test(tier: str, batch_size: int):
    """single 테스트 실행"""
    print(f"\n{'='*60}")
    print(f"Tier: {tier.upper()} | Batch size: {batch_size} files")
    print(f"{'='*60}")

    # 파일 선택
    files = get_test_files(batch_size)
    if len(files) < batch_size:
        print(f"[WARNING] Not enough files: {len(files)}/{batch_size}")
        return None

    files_json = json.dumps(files)

    # 명령어 실행
    cmd = [
        "python",
        "backend/pipeline/ingest_engine.py",
        "--files", files_json,
        "--batch-size", str(batch_size)
    ]

    print(f"Processing {batch_size} files...")
    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10분 타임아웃
            cwd=r"C:\Users\saint\ImageParser"
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            per_file = elapsed / batch_size
            print(f"[OK] Completed: {elapsed:.1f}s (avg {per_file:.1f}s/file)")
            return {
                'status': 'success',
                'total_time': round(elapsed, 2),
                'per_file': round(per_file, 2),
                'files_count': batch_size
            }
        else:
            error_msg = result.stderr[:200] if result.stderr else "Unknown error"
            print(f"[FAIL] Error: {error_msg}")
            return {
                'status': 'failed',
                'error': error_msg
            }

    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] Exceeded 10 minutes")
        return {'status': 'timeout'}

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return {'status': 'error', 'error': str(e)}

def main():
    print("=" * 70)
    print("Batch Processing Performance Benchmark (All Tiers)")
    print("=" * 70)
    print(f"\nTest files location: {TEST_DIR}")
    print(f"Tiers: {', '.join(TIERS)}")
    print(f"Batch sizes: {', '.join(map(str, BATCH_SIZES))}")
    print(f"\nEstimated time: ~20분")
    print()

    # Config 백업
    backup_config()

    try:
        # 각 Tiers별 테스트
        for tier in TIERS:
            print(f"\n\n{'#'*70}")
            print(f"# Tiers: {tier.upper()}")
            print(f"{'#'*70}")

            # Tiers 설정
            set_tier(tier)
            time.sleep(1)  # 설정 적용 대기

            results['tiers'][tier] = {}

            # 각 Batch sizes별 테스트
            for batch_size in BATCH_SIZES:
                result = run_test(tier, batch_size)
                if result:
                    results['tiers'][tier][batch_size] = result

                # 다음 테스트 전 대기 (모델 언로드)
                time.sleep(2)

        # Results saved
        output_file = Path("benchmark_results.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n\n{'='*70}")
        print(f"[OK] Benchmark completed!")
        print(f"Results saved: {output_file}")
        print(f"{'='*70}")

        # 결과 표 출력
        print_results(results)

    finally:
        # Config 복원
        restore_config()

def print_results(results):
    """결과 표 출력"""
    print("\n\n" + "=" * 80)
    print("[CHART] batch 처리 성능 비교 결과")
    print("=" * 80)

    print("\n[TIME]  Total Time (seconds)")
    print("-" * 80)
    print(f"{'Tier':<12} | {'1':<10} | {'5':<10} | {'10':<10} | {'20':<10}")
    print("-" * 80)

    for tier in TIERS:
        tier_data = results['tiers'].get(tier, {})
        row = f"{tier.upper():<12} |"
        for batch_size in BATCH_SIZES:
            data = tier_data.get(batch_size, {})
            if data.get('status') == 'success':
                row += f" {data['total_time']:>8.1f}s |"
            else:
                row += f" {'FAIL':>8} |"
        print(row)

    print("\n[GRAPH] Average Time (sec/file)")
    print("-" * 80)
    print(f"{'Tier':<12} | {'1':<10} | {'5':<10} | {'10':<10} | {'20':<10}")
    print("-" * 80)

    for tier in TIERS:
        tier_data = results['tiers'].get(tier, {})
        row = f"{tier.upper():<12} |"
        for batch_size in BATCH_SIZES:
            data = tier_data.get(batch_size, {})
            if data.get('status') == 'success':
                row += f" {data['per_file']:>8.1f}s |"
            else:
                row += f" {'-':>8} |"
        print(row)

    print("\n[SPEEDUP] Batch Efficiency Analysis")
    print("-" * 80)
    for tier in TIERS:
        tier_data = results['tiers'].get(tier, {})
        single = tier_data.get(1, {}).get('per_file')
        batch_20 = tier_data.get(20, {}).get('per_file')

        if single and batch_20:
            speedup = single / batch_20
            savings = (1 - batch_20/single) * 100
            print(f"{tier.upper():<12}: single {single:.1f}s → batch {batch_20:.1f}s "
                  f"(speedup: {speedup:.1f}배, saved: {savings:.0f}%)")

if __name__ == "__main__":
    main()
