from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.tools import tool
from langchain.tools.render import render_text_description

def count_vowels(string: str) -> int:
    """Counts the number of vowels in the string."""
    print(f"Counting vowels in string: {string}")
    vowels = "aeiouAEIOU"
    return sum(1 for char in string if char in vowels)

def caesar_cipher(string: str, shift: int = 4) -> str:
    """Shifts characters in the string by the given shift amount for Caesar cipher encryption."""
    print(f"Encrypting string using Caesar cipher: {string}")
    result = []
    for char in string:
        if char.isalpha():
            shift_base = ord('A') if char.isupper() else ord('a')
            result.append(chr((ord(char) - shift_base + shift) % 26 + shift_base))
        else:
            result.append(char)
    return ''.join(result)

@tool
def get_vowel_count(string: str) -> int:
    """Returns the number of vowels in the string."""
    return count_vowels(string)

@tool
def get_caesar_cipher(string: str) -> str:
    """Returns the input string encrypted with Caesar cipher (shift of 4)."""
    return caesar_cipher(string)

if __name__ == "__main__":
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(
        temperature=0,
        model="gemini-1.0-pro",
        max_tokens=1024,
    )

    # List of tools
    tools = [get_vowel_count, get_caesar_cipher]

    # Prompt template
    template = """
        Answer the following questions as best you can.
        You have access to the following tools:
        {tools}
        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action (just the value, no function call syntax)
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: """

    # Render tools for prompt
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools])
    )

    # Define the chain
    chain = {"input": lambda x: x["input"]} | prompt | llm

    while True:
        user_input = input("Enter your question (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting...")
            break
        res = chain.invoke({"input": user_input})
        print("Response:", res)
