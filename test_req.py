import requests
import base64

with open('images/image.jpg', 'rb') as img_file:
    base64_string = base64.b64encode(img_file.read()).decode('utf-8')

payload = {'frame': base64_string}

response = requests.post('http://127.0.0.1:5000/predict', json=payload)

print(response.json())