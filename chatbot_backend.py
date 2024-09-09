from langchain.chains import ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_aws import ChatBedrock
from langchain.schema import HumanMessage, AIMessage  # Import necessary classes

# Function to initialize the Bedrock client with the specified model and parameters
def demo_chatbot():
    demo_llm = ChatBedrock(
        credentials_profile_name='default',
        model_id='meta.llama3-8b-instruct-v1:0',
        model_kwargs={
            "temperature": 0.1,
            "top_p": 0.9,
            # Removed max_tokens as it's not supported by the model
        }
    )
    return demo_llm

# Function to initialize ConversationSummaryBufferMemory
def demo_memory():
    llm_data = demo_chatbot()  # Getting the chatbot instance
    memory = ConversationSummaryBufferMemory(
        llm=llm_data,
        max_token_limit=300
    )
    return memory

# Function to create a conversation chain and handle input text with memory
def demo_conversation(input_text, memory):
    # Ensure the input is a string
    if not isinstance(input_text, str):
        raise ValueError("Input text must be a string")

    llm_chain_data = demo_chatbot()  # Getting the chatbot instance
    llm_conversation = ConversationChain(
        llm=llm_chain_data,
        memory=memory,
        verbose=True
    )

    # Ensure input is passed as a string
    chat_reply = llm_conversation.predict(input=input_text)

    # Ensure the response is in string format
    if not isinstance(chat_reply, str):
        chat_reply = str(chat_reply)

    return chat_reply  # Returning the response directly

# Example usage in your Streamlit app
if __name__ == "__main__":
    import streamlit as st

    # Initialize memory for the conversation
    if "memory" not in st.session_state:
        st.session_state.memory = demo_memory()

    st.title("Chatbot with AWS Bedrock")

    # Input from the user
    input_text = st.text_input("Enter your message:")

    if st.button("Send"):
        if input_text:
            try:
                chat_response = demo_conversation(input_text=input_text, memory=st.session_state.memory)
                st.write("Chatbot:", chat_response)
            except ValueError as e:
                st.error(f"Input Error: {e}")
            except Exception as e:
                st.error(f"Error: {e}")
