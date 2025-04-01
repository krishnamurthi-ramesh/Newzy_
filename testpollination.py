import requests
from urllib.parse import quote
import time
import os
import random

def test_pollination():
    print("Testing Pollinations.ai image generation...")
    
    # First, get available models
    models_url = "https://image.pollinations.ai/models"
    try:
        response = requests.get(models_url, timeout=10)
        if response.status_code == 200:
            models = response.json()
            print(f"Available models: {models}")
            # Prefer more reliable models if available
            preferred_models = ["flux", "turbo", "stability", "stable-diffusion"]
            selected_model = None
            for model in preferred_models:
                if model in models:
                    selected_model = model
                    break
            if not selected_model and models:
                selected_model = models[0]
            else:
                selected_model = "flux"  # Default to flux as it seems most reliable
        else:
            print(f"Failed to get models: Status code {response.status_code}")
            selected_model = "flux"
            
        # Test with a simple prompt
        test_prompt = "Beautiful sunset over mountains with clouds"
        encoded_prompt = quote(test_prompt)
        
        # Try different URL formats based on current Pollinations.ai API
        urls_to_test = [
            f"https://image.pollinations.ai/prompt/{encoded_prompt}",
            f"https://image.pollinations.ai/{selected_model}/prompt/{encoded_prompt}",
            f"https://image.pollinations.ai/{selected_model}/{encoded_prompt}",
            f"https://image.pollinations.ai/prompt/{encoded_prompt}?height=400&width=800&nologo=true"
        ]
        
        for i, url in enumerate(urls_to_test):
            print(f"\nTesting URL #{i+1}: {url}")
            
            try:
                # Special handling for this URL to add wait time
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
                
                # First make a GET request to trigger image generation
                start_time = time.time()
                response = requests.get(url, headers=headers, timeout=30, stream=True)
                
                # Read full content to ensure we get the complete image
                content = response.content
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    print(f"Success! Status code: {response.status_code}")
                    print(f"Response time: {elapsed:.2f} seconds")
                    print(f"Content type: {response.headers.get('Content-Type')}")
                    print(f"Content size: {len(content)} bytes")
                    
                    # Check if this is actually an image
                    if 'image' in response.headers.get('Content-Type', ''):
                        # Save the image
                        filename = f"test_image_{i+1}.jpg"
                        with open(filename, 'wb') as f:
                            f.write(content)
                        
                        # Check if image has content
                        file_size = os.path.getsize(filename)
                        print(f"Saved image: {filename}, Size: {file_size} bytes")
                        
                        if file_size > 1000:
                            print(f"✓ VALID IMAGE FOUND with URL format #{i+1}")
                            return url
                        else:
                            print("✗ Image file is too small - likely not a valid image")
                    else:
                        print("✗ Response is not an image")
                else:
                    print(f"✗ Failed: Status code {response.status_code}")
            except Exception as e:
                print(f"✗ Error with URL #{i+1}: {str(e)}")
                
        # If all standard methods failed, try a direct approach with higher timeout
        print("\nTrying final direct method with long timeout...")
        final_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        
        try:
            print(f"Final URL: {final_url}")
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "image/jpeg, image/png, */*"
            }
            
            # We need to wait longer for this request
            response = requests.get(final_url, headers=headers, timeout=60)
            
            if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                filename = "test_final_image.jpg"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                    
                file_size = os.path.getsize(filename)
                print(f"Final method image size: {file_size} bytes")
                
                if file_size > 1000:
                    print("✓ SUCCESS with final method!")
                    return final_url
        except Exception as e:
            print(f"✗ Final method failed: {str(e)}")
    
        return None
        
    except Exception as e:
        print(f"Error testing Pollinations.ai: {str(e)}")
        return None

if __name__ == "__main__":
    working_url = test_pollination()
    if working_url:
        print(f"\nSuccessfully generated test image. The working URL is:\n{working_url}")
    else:
        print("\nFailed to generate image using Pollinations.ai") 