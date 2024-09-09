import streamlit as st
from PIL import Image
import plant  # Import your plant module
import chatbot_backend as demo  # Import your chatbot backend module

def app():
    st.title("CropTalk 😎")  # Customize this title as needed

    # Initialize memory for the conversation in session state
    if 'memory' not in st.session_state:
        st.session_state.memory = demo.demo_memory()  # Initialize memory

    # Initialize chat history in session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []  # Initialize chat history

    # Initialize disease prediction in session state
    if 'disease_prediction' not in st.session_state:
        st.session_state.disease_prediction = None

    # Upload an image with a unique key
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"], key="file_uploader_1")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Predict disease
        if st.button('Classify'):
            with st.spinner('Classifying the image...'):
                try:
                    # Pass the FastAPI URL directly to the predict_disease function
                    fastapi_url = "http://ec2-65-2-137-251.ap-south-1.compute.amazonaws.com:8000/predict"
                    if hasattr(plant, 'predict_disease'):
                        # Store the prediction in session state
                        st.session_state.disease_prediction = plant.predict_disease(uploaded_file, fastapi_url)
                        st.success(f'Disease Prediction: {st.session_state.disease_prediction}')
                    else:
                        st.error('Function predict_disease not found in plant module.')
                except Exception as e:
                    st.error(f"Error: {e}")

    # Show "Know More" button
    if st.button('Know More'):
        if st.session_state.disease_prediction:  # Check if a disease prediction is available
            prompt = f"Give information about {st.session_state.disease_prediction} and its preventive measures"
            with st.spinner('Getting information...'):
                response = demo.demo_conversation(input_text=prompt, memory=st.session_state.memory)
                st.write(f"Chatbot: {response}")
        else:
            st.error("No disease prediction available to provide more information.")

    # Re-render the chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])

    # Get user input from the chat input box
    input_text = st.chat_input("From Seeds to Harvest! I am here to cultivate success")  # Placeholder text for the chat input box

    if input_text:  # If the user provides input
        with st.chat_message("user"):
            st.markdown(input_text)

        # Add user input to chat history
        st.session_state.chat_history.append({"role": "user", "text": input_text})

        # Get chatbot response
        try:
            chat_response = demo.demo_conversation(input_text=input_text, memory=st.session_state.memory)
            
            # Ensure the chatbot response is in string format
            if not isinstance(chat_response, str):
                chat_response = str(chat_response)

            # Display chatbot response
            with st.chat_message("assistant"):
                st.markdown(chat_response)

            # Add chatbot response to chat history
            st.session_state.chat_history.append({"role": "assistant", "text": chat_response})
        except Exception as e:
            st.error(f"Error: {e}")  # Display error message if something goes wrong

if __name__ == "__main__":
    app()
