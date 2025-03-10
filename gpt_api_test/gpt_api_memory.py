import openai
import time
import os
from dotenv import load_dotenv

load_dotenv("openaiAPI.env")
api_key = os.getenv("api_key")

client = openai.OpenAI(api_key=api_key)

# 1. GPT-4V를 사용하여 이미지 분석
image_file = client.files.create(
    file=open("/home/ws/Desktop/LMTG/language-models-trajectory-generators/gpt_api_test/image.png", "rb"),
    purpose="vision"
)

file_id = image_file.id

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are an AI that extracts useful information for manipulator robot from images."},
        {"role": "user", "content": [
            {"type": "text", "text": "이 이미지에서 중요한 내용을 요약해줘. 특히 프링글스 통을 아래쪽으로 이동하기 위해 필요한 정보들을 알려줘."},
            {"type": "image_url", "image_url": f"{file_id}"}
        ]}
    ],
    max_tokens=500
)

# 2. 이미지 분석 결과 저장
image_analysis_result = response.choices[0].message["content"]
print("이미지 분석 결과:", image_analysis_result)

# 3. Assistants API에 텍스트로 저장
assistant = client.beta.assistants.create(
    name="Image Memory Assistant",
    instructions="You remember the analyzed content of uploaded images.",
    model="gpt-4-turbo",
    tools=[{"type": "retrieval"}]  # 검색 기능 활성화
)

# 4. 새로운 대화 스레드 생성
thread = client.beta.threads.create()

# 5. 분석한 내용을 Assistants API에 입력
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content=f"이전에 분석한 이미지 내용: {image_analysis_result}"
)

# 6. 나중에 같은 스레드에서 질문하면 해당 내용이 유지됨
message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="내가 올렸던 이미지에서 추출한 정보 다시 말해줘."
)

# 7. Assistants API 실행
run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

while run.status != "completed":
    time.sleep(1)
    run = client.beta.threads.runs.retrieve(run.id)

messages = client.beta.threads.messages.list(thread_id=thread.id)
print(messages.data[0].content)  # 이전 이미지 분석 내용 출력
