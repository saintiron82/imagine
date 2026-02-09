#!/usr/bin/env python3
"""
Database Reset Tool - ImageParser v3.1

기존 SQLite 데이터베이스를 백업하고 완전히 초기화합니다.

Usage:
    python tools/db_reset.py --backup --confirm
    python tools/db_reset.py --no-backup --confirm
"""

import argparse
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path


def get_project_root() -> Path:
    """프로젝트 루트 디렉토리 반환."""
    return Path(__file__).parent.parent


def get_db_path() -> Path:
    """데이터베이스 파일 경로 반환."""
    return get_project_root() / "imageparser.db"


def get_schema_path() -> Path:
    """스키마 파일 경로 반환."""
    return get_project_root() / "backend" / "db" / "sqlite_schema.sql"


def backup_database(db_path: Path) -> Path:
    """
    데이터베이스 파일 백업.

    Args:
        db_path: 원본 DB 파일 경로

    Returns:
        백업 파일 경로
    """
    if not db_path.exists():
        print(f"⚠️  No database file found at: {db_path}")
        return None

    # 백업 파일명 생성 (타임스탬프 포함)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"imageparser.db.backup.v3.0.{timestamp}"

    try:
        shutil.copy2(db_path, backup_path)
        file_size = backup_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ Backup created: {backup_path.name} ({file_size:.2f} MB)")
        return backup_path
    except Exception as e:
        print(f"❌ Backup failed: {e}")
        sys.exit(1)


def delete_database(db_path: Path):
    """
    데이터베이스 파일 삭제.

    Args:
        db_path: DB 파일 경로
    """
    if not db_path.exists():
        print(f"ℹ️  No database file to delete: {db_path}")
        return

    try:
        db_path.unlink()
        print(f"✅ Database deleted: {db_path.name}")
    except Exception as e:
        print(f"❌ Delete failed: {e}")
        sys.exit(1)


def recreate_schema(db_path: Path, schema_path: Path):
    """
    스키마 파일로부터 데이터베이스 재생성.

    Args:
        db_path: 새 DB 파일 경로
        schema_path: SQL 스키마 파일 경로
    """
    if not schema_path.exists():
        print(f"❌ Schema file not found: {schema_path}")
        sys.exit(1)

    try:
        # 스키마 파일 읽기
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()

        # 새 데이터베이스 생성 및 스키마 적용
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # 여러 SQL 문 실행
        cursor.executescript(schema_sql)

        conn.commit()
        conn.close()

        print(f"✅ Database schema recreated: {db_path.name}")

        # 테이블 확인
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        print(f"   Tables created: {', '.join(tables)}")

    except Exception as e:
        print(f"❌ Schema recreation failed: {e}")
        sys.exit(1)


def reset_database(backup: bool = True, confirm: bool = False):
    """
    데이터베이스 완전 초기화 메인 로직.

    Args:
        backup: 백업 생성 여부
        confirm: 사용자 확인 여부
    """
    db_path = get_db_path()
    schema_path = get_schema_path()

    print("=" * 60)
    print("🔄 ImageParser v3.1 Database Reset Tool")
    print("=" * 60)
    print(f"Database: {db_path}")
    print(f"Schema: {schema_path}")
    print(f"Backup: {'Yes' if backup else 'No'}")
    print("=" * 60)

    if not confirm:
        print("\n⚠️  WARNING: This will DELETE all existing data!")
        response = input("   Type 'YES' to confirm: ")
        if response != "YES":
            print("❌ Reset cancelled.")
            sys.exit(0)

    print("\n🚀 Starting database reset...\n")

    # Step 1: 백업 (옵션)
    if backup and db_path.exists():
        print("📦 Step 1: Creating backup...")
        backup_database(db_path)
    else:
        print("📦 Step 1: Skipping backup...")

    # Step 2: 기존 DB 삭제
    print("\n🗑️  Step 2: Deleting old database...")
    delete_database(db_path)

    # Step 3: 스키마 재생성
    print("\n🏗️  Step 3: Recreating schema...")
    recreate_schema(db_path, schema_path)

    print("\n" + "=" * 60)
    print("🎉 Database reset complete!")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   1. Run: python backend/pipeline/ingest_engine.py --discover \"C:\\path\\to\\assets\"")
    print("   2. Or: Use the frontend to process files")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Reset ImageParser database with optional backup"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        default=True,
        help="Create backup before reset (default: True)"
    )
    parser.add_argument(
        "--no-backup",
        dest="backup",
        action="store_false",
        help="Skip backup creation"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt (use with caution!)"
    )

    args = parser.parse_args()

    reset_database(backup=args.backup, confirm=args.confirm)


if __name__ == "__main__":
    main()
