## Basic Streamlit chatbot that uses a local Hugging Face LLM (Mistral)

import streamlit as st

from transformers import AutoTokenizer
from transformers import pipeline
from transformers import TextIteratorStreamer
from transformers import BitsAndBytesConfig

import torch

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

generate_kwargs = {
    "max_new_tokens": 1024,
    "do_sample": True,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.95
}

# Initialize chat history with system message.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "user",
            "content": "You are a friendly chatbot who is curious about the user. You will ask the user a question to get to know them better. You will keep your responses short."
        }
    ]

if "pipe" not in st.session_state:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)
    st.session_state.streamer = streamer

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
    
    pipe = pipeline(
        "conversational",
        model=model_id,
        model_kwargs={
            "torch_dtype": "auto",
            # "quantization_config": bnb_config,
            # "attn_implementation": "flash_attention_2"
        },
        device_map="auto",
        tokenizer=tokenizer,
        streamer=streamer
    )
    
    st.session_state.pipe = pipe
    
    # Send the system message to the model
    pipe(
        conversations=st.session_state.messages,
        bos_token_id=pipe.model.config.bos_token_id,
        eos_token_id=pipe.model.config.eos_token_id,
        pad_token_id=pipe.model.config.eos_token_id,
        kwargs=generate_kwargs
    )
    
    # Wait until the system message response is done
    generated_text = ""
    for text in streamer:
        generated_text += text
        
st.title("MeChat")

# Display chat messages from history on app rerun. Skip first system message and its response.
for message in st.session_state.messages[1:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What's up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    pipe = st.session_state.pipe
    
    model_kwargs = dict(
        conversations=st.session_state.messages,
        bos_token_id=pipe.model.config.bos_token_id,
        eos_token_id=pipe.model.config.eos_token_id,
        pad_token_id=pipe.model.config.eos_token_id,
        kwargs=generate_kwargs
    )
    
    # Run the inferencing in a thread so that the output can be streamed
    from threading import Thread
    thread = Thread(target=pipe, kwargs=model_kwargs)
    thread.start()
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response = st.write_stream(st.session_state.streamer)
