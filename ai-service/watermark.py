"""
SportShield AI - Digital Watermarking Module
Implements invisible LSB (Least Significant Bit) watermarking for media integrity.
"""

import numpy as np
import cv2
from PIL import Image
import io

def embed_lsb_watermark(image_np, secret_text):
    """
    Embed secret text into the LSB of the blue channel of an image.
    """
    # Convert text to binary
    binary_secret = ''.join(format(ord(i), '08b') for i in secret_text)
    # Add a sentinel to mark the end of the message
    binary_secret += '1111111111111110' 
    
    data_index = 0
    data_len = len(binary_secret)
    
    # Work on a copy
    watermarked = image_np.copy()
    
    height, width, _ = watermarked.shape
    
    for y in range(height):
        for x in range(width):
            if data_index < data_len:
                # Get the pixel's blue channel value
                blue = watermarked[y, x, 0]
                # Replace LSB with bit from secret
                bit = int(binary_secret[data_index])
                watermarked[y, x, 0] = (blue & ~1) | bit
                data_index += 1
            else:
                break
        if data_index >= data_len:
            break
            
    return watermarked

def extract_lsb_watermark(image_np):
    """
    Extract secret text from the LSB of the blue channel.
    """
    binary_data = ""
    height, width, _ = image_np.shape
    
    for y in range(height):
        for x in range(width):
            blue = image_np[y, x, 0]
            binary_data += str(blue & 1)
            
            # Check for sentinel every 8 bits
            if len(binary_data) % 8 == 0 and len(binary_data) >= 16:
                if binary_data[-16:] == '1111111111111110':
                    # Found sentinel
                    binary_data = binary_data[:-16]
                    # Convert binary to text
                    all_bytes = [binary_data[i:i+8] for i in range(0, len(binary_data), 8)]
                    decoded_text = ""
                    for b in all_bytes:
                        decoded_text += chr(int(b, 2))
                    return decoded_text
                    
    return None
