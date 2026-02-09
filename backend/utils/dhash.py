"""
Perceptual hashing using dHash (difference hash).

dHash is a simple, fast perceptual hash that detects near-duplicate images.
Uses 9x8 grayscale gradient to generate a 64-bit hash.

Pure PIL implementation, no external dependencies.
"""

from PIL import Image
from typing import Union


def dhash64(image: Union[Image.Image, str], hash_size: int = 8) -> int:
    """
    Compute 64-bit dHash for an image.

    Args:
        image: PIL Image object or path to image file
        hash_size: Hash dimensions (default 8 → 64-bit hash)

    Returns:
        64-bit integer hash

    Example:
        >>> from PIL import Image
        >>> img = Image.open('photo.jpg')
        >>> hash1 = dhash64(img)
        >>> hash2 = dhash64('photo_resized.jpg')
        >>> hamming_distance = bin(hash1 ^ hash2).count('1')
        >>> is_duplicate = hamming_distance < 10  # Threshold: 10 bits
    """
    # Load image if path provided
    if isinstance(image, str):
        image = Image.open(image)

    # Convert to grayscale
    gray = image.convert('L')

    # Resize to (hash_size+1) x hash_size
    # +1 width for horizontal gradient comparison
    resized = gray.resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)

    # Compute horizontal gradient (compare adjacent pixels)
    pixels = list(resized.getdata())
    width = hash_size + 1

    # Build hash from gradient bits
    hash_value = 0
    for row in range(hash_size):
        for col in range(hash_size):
            offset = row * width + col
            left = pixels[offset]
            right = pixels[offset + 1]

            # Set bit if left > right
            if left > right:
                bit_index = row * hash_size + col
                hash_value |= (1 << bit_index)

    return hash_value


def hamming_distance(hash1: int, hash2: int) -> int:
    """
    Compute Hamming distance between two hashes.

    Args:
        hash1, hash2: Integer hashes from dhash64()

    Returns:
        Number of differing bits (0-64)

    Example:
        >>> dist = hamming_distance(hash1, hash2)
        >>> if dist < 10:
        >>>     print("Images are nearly identical")
    """
    return bin(hash1 ^ hash2).count('1')


def is_duplicate(hash1: int, hash2: int, threshold: int = 10) -> bool:
    """
    Check if two hashes represent duplicate/similar images.

    Args:
        hash1, hash2: Integer hashes
        threshold: Max Hamming distance for duplicates (default 10)
                  - 0-5: Nearly identical
                  - 6-10: Very similar
                  - 11-15: Similar
                  - 16+: Different

    Returns:
        True if images are duplicates
    """
    return hamming_distance(hash1, hash2) <= threshold
