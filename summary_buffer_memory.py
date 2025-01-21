from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
llm= ChatGoogleGenerativeAI(model="gemini-1.5-flash")
memory = ConversationSummaryBufferMemory(
llm=llm,
max_token_limit=200,
)
conversation = ConversationChain(
llm=llm,
memory=memory,
verbose= True 
)
while True:
    user_input = input("Saii ")
    if user_input.lower() in ['exit']:
        print("Goodbye!")
        print("\n Summary:")
        print(conversation.memory.buffer)
        break
    
    response= conversation.predict(input=user_input)
    print("\nAI:", response)
