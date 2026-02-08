"""Test script for BaseParser ABC verification."""
from backend.parser.base_parser import BaseParser
from pathlib import Path

class IncompleteParser(BaseParser):
    """Parser that doesn't implement parse() - should fail"""
    supported_extensions = ['test']

try:
    p = IncompleteParser()
    p.parse(Path('test.txt'))
    print("ERROR: Should have raised TypeError!")
except TypeError as e:
    print(f"BaseParser OK - Abstract method enforced: {e}")
