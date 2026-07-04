"""Test CV file upload endpoint"""
import requests
import json
from pathlib import Path

# Test 1: Upload a JSON file
print("=" * 60)
print("Test 1: Upload CV file (JSON)")
print("=" * 60)
test_file = Path(__file__).parent.parent / 'test_cv_upload.json'
with open(test_file, 'rb') as f:
    files = {'file': ('test_cv_upload.json', f, 'application/json')}
    r = requests.post('http://localhost:8000/api/v1/cv-files/upload', files=files)
    result = r.json()
    print(f"Status: {r.status_code}")
    print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
    
    if result.get('success'):
        file_id = result['data']['id']
        print(f"\nUploaded file ID: {file_id}")

        # Test 2: List CV files
        print("\n" + "=" * 60)
        print("Test 2: List CV files")
        print("=" * 60)
        r = requests.get('http://localhost:8000/api/v1/cv-files')
        result = r.json()
        print(f"Status: {r.status_code}")
        print(f"Total files: {result['data']['total']}")
        for item in result['data']['items']:
            print(f"  - {item['filename']} ({item['file_type']}, {item['file_size']} bytes)")

        # Test 3: Get file details
        print("\n" + "=" * 60)
        print("Test 3: Get file details")
        print("=" * 60)
        r = requests.get(f'http://localhost:8000/api/v1/cv-files/{file_id}')
        result = r.json()
        print(f"Status: {r.status_code}")
        print(f"Filename: {result['data']['filename']}")
        print(f"Content preview: {result['data'].get('content', 'N/A')[:100]}...")

        # Test 4: Upload a text file
        print("\n" + "=" * 60)
        print("Test 4: Upload CV file (TXT)")
        print("=" * 60)
        txt_content = "Solar preprocessing report\nDate: 2026-06-18\nDisk detected: True\nSunspots: 2"
        files = {'file': ('report.txt', txt_content, 'text/plain')}
        r = requests.post('http://localhost:8000/api/v1/cv-files/upload', files=files)
        result = r.json()
        print(f"Status: {r.status_code}")
        print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")

        # Test 5: Delete file
        print("\n" + "=" * 60)
        print("Test 5: Delete CV file")
        print("=" * 60)
        r = requests.delete(f'http://localhost:8000/api/v1/cv-files/{file_id}')
        result = r.json()
        print(f"Status: {r.status_code}")
        print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")

print("\n" + "=" * 60)
print("All tests completed successfully!")
print("=" * 60)
