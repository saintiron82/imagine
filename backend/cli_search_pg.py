"""
CLI search tool using PostgreSQL + pgvector.

This replaces backend/cli_search.py (ChromaDB version) with PostgreSQL.

Usage:
    python backend/cli_search_pg.py "cartoon city"
    python backend/cli_search_pg.py "fantasy character" --mode hybrid --format PSD
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.search.pg_search import PgVectorSearch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Search images using PostgreSQL + pgvector")
    parser.add_argument("query", help="Search query text")
    parser.add_argument("--mode", choices=["vector", "hybrid", "metadata"], default="vector",
                        help="Search mode (default: vector)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of results (default: 10)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Similarity threshold 0.0-1.0 (default: 0.0)")

    # Metadata filters
    parser.add_argument("--format", help="Filter by format (e.g., PSD, PNG)")
    parser.add_argument("--min-width", type=int, help="Minimum width")
    parser.add_argument("--max-width", type=int, help="Maximum width")
    parser.add_argument("--tags", help="Filter by tags (ILIKE search)")

    args = parser.parse_args()

    # Initialize search
    try:
        search = PgVectorSearch()
    except Exception as e:
        logger.error(f"Failed to initialize search: {e}")
        logger.error("\nPlease ensure:")
        logger.error("1. PostgreSQL is running")
        logger.error("2. Database is migrated (python tools/migrate_to_postgres.py)")
        return 1

    # Build filters
    filters = {}
    if args.format:
        filters['format'] = args.format
    if args.min_width:
        filters['min_width'] = args.min_width
    if args.max_width:
        filters['max_width'] = args.max_width
    if args.tags:
        filters['tags'] = args.tags

    # Perform search
    logger.info(f"Searching: '{args.query}' (mode: {args.mode}, top_k: {args.top_k})")

    if filters:
        logger.info(f"Filters: {filters}")

    try:
        results = search.search(
            query=args.query,
            mode=args.mode,
            filters=filters if filters else None,
            top_k=args.top_k,
            threshold=args.threshold
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return 1

    # Display results
    print(f"\n{'='*80}")
    print(f"Search Results: {len(results)} matches")
    print(f"{'='*80}\n")

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['file_name']}")
        print(f"   Path: {result['file_path']}")
        print(f"   Format: {result.get('format', 'N/A')} | "
              f"Size: {result.get('width', 0)}x{result.get('height', 0)}")

        if 'similarity' in result:
            print(f"   Similarity: {result['similarity']:.4f}")

        if result.get('ai_caption'):
            print(f"   Caption: {result['ai_caption']}")

        if result.get('ai_tags'):
            print(f"   Tags: {', '.join(result['ai_tags'])}")

        # Show layer info from metadata
        if result.get('metadata') and result['metadata'].get('layer_count'):
            layer_count = result['metadata']['layer_count']
            print(f"   Layers: {layer_count}")

        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
