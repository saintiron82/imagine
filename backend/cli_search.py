import sys
import json
import logging
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging to stderr so stdout is clean for JSON
logging.basicConfig(level=logging.ERROR)

from backend.vector.searcher import VectorSearcher

def main():
    if len(sys.argv) < 2:
        print(json.dumps([]))
        return

    query = sys.argv[1]
    
    try:
        searcher = VectorSearcher()
        results = searcher.search(query, top_k=50) # Return top 50
        print(json.dumps(results))
    except Exception as e:
        # Log error to stderr, empty list to stdout
        logging.error(f"Search failed: {e}")
        print(json.dumps([]))

if __name__ == "__main__":
    main()
