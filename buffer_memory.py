from langchain.chains.conversation.memory import ConversationSummaryMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
load_dotenv()
memory=ConversationBufferMemory()
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash")
conversation=ConversationChain(llm=llm,memory=memory)
while True:
    user_input=input("Saikishen:")
    if user_input.lower() in ['exit']:
        print("bye")
        print(conversation.memory.buffer)
        break
    response=conversation.predict(input= user_input)
    print("\n Gemeni:",response)
