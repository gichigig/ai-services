import requests
url = "https://ngangabildad--video-analyzer-service-fastapi-app.modal.run/ask-question"
res = requests.post(url, json={"student_id": "test", "question": "What is 2+2?"})
print(f"Status: {res.status_code}")
print(f"Text: {res.text}")
