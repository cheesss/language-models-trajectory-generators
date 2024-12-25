from openai import OpenAI
import os
from dotenv import load_dotenv


load_dotenv("openaiAPI.env")
api_key = os.getenv("api_key")
client = OpenAI(api_key=api_key)

# OpenAI API 키 설정
  # 실제 API 키로 교체

# 대화 히스토리 초기화
conversation = [{"role": "system", "content": "You are a helpful assistant."}]

def chat_with_openai(user_input):
    # 사용자의 입력 추가
    conversation.append({"role": "user", "content": user_input})

    # API 호출
    try:
        response = client.chat.completions.create(model="gpt-3.5-turbo",  # 또는 gpt-3.5-turbo
        messages=conversation)
        # 응답 추출
        assistant_reply = response.choices[0].message.content
        print(f"Assistant: {assistant_reply}")

        # 대화 히스토리에 추가
        conversation.append({"role": "assistant", "content": assistant_reply})
    except Exception as e:
        print(f"Error: {e}")

# 메인 실행 루프
if __name__ == "__main__":
    print("Chat with the assistant! Type 'exit' or 'quit' to end the chat.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chat.")
            break
        chat_with_openai(user_input)
