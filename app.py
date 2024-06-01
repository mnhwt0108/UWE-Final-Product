import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, VisionEncoderDecoderModel
from PIL import Image

# Define the generate_caption function
def generate_caption(image, model, tokenizer, device, image_height, image_width):
    # Resize and convert image to numpy array for tensor conversion
    image_resized = image.resize((image_width, image_height))
    image_array = np.array(image_resized, dtype=np.uint8)
    image_tensor = torch.tensor(np.moveaxis(image_array, -1, 0)).unsqueeze(0).to(device)
    
    # Encode image using the model's encoder
    encoder_output = model.encoder(pixel_values=image_tensor)
    
    # Initialize the decoder input with the special token for start of sequence
    decoder_input_ids = torch.tensor(tokenizer.encode("[CLS]")).unsqueeze(0).to(device)
    
    # Initialize the list to hold generated token IDs
    generated_ids = []
    
    # Set maximum length for generated caption
    max_length = 32
    
    # Generate tokens one by one using the decoder
    for _ in range(max_length):
        # Generate next token
        outputs = model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=encoder_output.last_hidden_state)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = next_token_logits.argmax(1).unsqueeze(-1)
        
        # Append the token to the list of generated tokens
        generated_ids.append(next_token_id.item())
        
        # Break if the end of sequence token is generated
        if next_token_id.item() == tokenizer.sep_token_id:
            break
        
        # Prepare input for the next iteration
        decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)
    
    # Decode the generated token IDs into a caption string
    generated_caption = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_caption

# Load the model, tokenizer, and feature extractor
@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained('TEST')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# Streamlit app
st.title("Image Captioning with Transformer Models")

# Load the model, tokenizer, and device
model, tokenizer, device = load_model()

# File uploader to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # To read the image file buffer as a PIL Image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Display the original image
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    
    # Generate caption
    caption = generate_caption(image, model, tokenizer, device, image_height=224, image_width=224)
    
    
    # Display the generated caption
    st.write(f"Generated Caption: {caption}")