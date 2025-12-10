#!/usr/bin/env python
import os

env_path = os.path.join(os.path.dirname(__file__), '.env')

try:
    # Try to read as UTF-16 (which has BOM 0xff)
    with open(env_path, 'r', encoding='utf-16') as f:
        content = f.read()
except:
    # Fallback: try utf-8-sig for UTF-8 with BOM
    try:
        with open(env_path, 'r', encoding='utf-8-sig') as f:
            content = f.read()
    except:
        # Last resort: read as binary and decode
        with open(env_path, 'rb') as f:
            raw = f.read()
            content = raw.decode('utf-16')

# Write back as UTF-8
with open(env_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"Fixed encoding for {env_path}")
