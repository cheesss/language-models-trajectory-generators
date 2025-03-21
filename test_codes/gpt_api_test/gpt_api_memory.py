import openai
import time
import os
from dotenv import load_dotenv

load_dotenv("openaiAPI.env")
api_key = os.getenv("api_key")

client = openai.OpenAI(api_key=api_key)

google_drive_image_url = "https://drive.google.com/uc?id=10tiP_DW0BhKpbdJWQi6xvARaDMpH0FPJ"
# 드라이브 공개 URL로 사진 가져오기


# response = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are an AI that extracts useful information for manipulator robot from images."},
#         # 역할지정
#         {"role": "user", "content": [
#             {"type": "text", "text": "이 이미지에서 중요한 내용을 요약해줘. 특히 프링글스 통을 아래쪽으로 이동하기 위해 필요한 정보들을 알려줘."},
#             {"type": "image_url", "image_url": {"url": google_drive_image_url}}
#         ]}
#         # 다운로드해온 이미지 분석
#     ],
#     max_tokens=500
# )

# image_analysis_result = response.choices[0].message.content
# # 분석결과 저장
# print("이미지 분석 결과:", image_analysis_result)
print("-----------------------------------------------")


thread = client.beta.threads.create()
assistant = client.beta.assistants.create(
    name="Image Memory Assistant",
    instructions="You remember the analyzed content of uploaded images. And you're a six-degree manipulator. To implement the instructions with image analysis, analyze the caveats, such as collisions, and create an action plan to carry out the instructions. 충돌을 피하기 위해 고려해야하는 물건들을 모두 말해줘. 앞으로 한글로만 답해줘",
    model="gpt-4o-mini",
    tools=[{"type": "code_interpreter"}]
)


uploaded_file = client.files.create(
    file=open("image_1.png", "rb"), 
    purpose="assistants"
)



message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="프링글스 통을 리얼센스 박스 왼쪽으로 옮기려면 어떻게 해야해?",
    attachments=[{
        "file_id": uploaded_file.id, 
        "tools": [{"type":"code_interpreter"}]
    }] 
)



run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

while run.status != "completed":
    time.sleep(1)
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

messages = client.beta.threads.messages.list(thread_id=thread.id)
print("-------------------------------------------------------------------------------------")
print("이미지 분석 결과:")
print(messages.data[0].content)  # Assistant의 이미지 분석 결과 출력
print("-------------------------------------------------------------------------------------")


uploaded_file2 = client.files.create(
    file=open("image_2.png", "rb"), 
    purpose="assistants"
)


message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="전에 지시사항이 무엇이었는지 말하고, 지시사항이 성공적으로 수행되었는지 너가 판단해서 알려줘. 또 이전의 지시사항이 이행되었는지에 대해 어떤 기준으로 판단한건지도 말해줘.",
    attachments=[{
        "file_id": uploaded_file2.id, 
        "tools": [{"type":"code_interpreter"}]
    }] 
)



run = client.beta.threads.runs.create(
    thread_id=thread.id,
    assistant_id=assistant.id
)

while run.status != "completed":
    time.sleep(1)
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)

messages = client.beta.threads.messages.list(thread_id=thread.id)
print("-------------------------------------------------------------------------------------")
print("이미지 분석 결과:")
print(messages.data[0].content)  # Assistant의 이미지 분석 결과 출력
print("-------------------------------------------------------------------------------------")