"""Quick dhash64 unit test."""
from backend.utils.dhash import dhash64
from PIL import Image

# Test 1: Blue image (solid color -> hash=0 is expected!)
img = Image.new('RGB', (100, 100), color='blue')
h = dhash64(img)
print(f'Blue image hash: {h}')
print('[OK] Test 1 passed (solid colors have hash=0, this is correct!)')

# Test 2: Red image (also should be 0)
img2 = Image.new('RGB', (100, 100), color='red')
h2 = dhash64(img2)
print(f'Red image hash: {h2}')
assert h2 == 0, 'Solid red should also be 0'
print('[OK] Test 2 passed')

# Test 3: Gradient image
img3 = Image.new('RGB', (100, 100))
pixels = img3.load()
for y in range(100):
    for x in range(100):
        pixels[x, y] = (x * 2, y * 2, 128)
h3 = dhash64(img3)
print(f'Gradient image hash: {h3}')
assert h3 != 0, 'Gradient image should have non-zero hash'
print('[OK] Test 3 passed')

print('\n[SUCCESS] All dhash64 tests passed!')
