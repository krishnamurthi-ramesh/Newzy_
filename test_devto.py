import os
import sys
from urllib.parse import quote
import requests
import json
from datetime import datetime

# Test article data
test_article = {
    'title': 'Test Article for Image Generation',
    'summary': 'This is a test summary to verify that Pollinations.ai image generation works properly with the DevtoPublisher class.',
    'text': 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam auctor, nisl eget ultricies tincidunt, nunc nisl aliquam nisl, eget aliquam nisl nisl eget nisl.',
    'url': 'https://example.com/test-article',
    'publish_date': datetime.now(),
    'image_url': 'https://example.com/test-image.jpg',
    'source': 'Test Source'
}

class DevtoPublisherTest:
    def __init__(self):
        self.api_key = os.environ.get('DEVTO_API_KEY', 'your_devto_api_key')
        self.api_url = "https://dev.to/api/articles"
        self.headers = {
            'api-key': self.api_key,
            'Content-Type': 'application/json'
        }

    def generate_image(self, article):
        """Generate image using Pollinations.ai"""
        try:
            # Get available models first
            models_url = "https://image.pollinations.ai/models"
            response = requests.get(models_url, timeout=10)
            if response.status_code == 200:
                models = response.json()
                print(f"Available models: {models}")
                # Use the first available model (usually 'flux')
                selected_model = models[0] if models else "flux"
            else:
                print(f"Failed to get models: {response.status_code}")
                selected_model = "flux"  # Default to flux

            # Create prompt from article title and summary
            prompt = article.get('title', '')
            summary = article.get('summary', '')
            if summary and len(summary) > 10:
                prompt += f" - {summary[:100]}"
            
            # Clean and encode the prompt
            prompt = prompt.strip()
            if not prompt:
                prompt = "News article image"
            encoded_prompt = quote(prompt, safe='')
            
            print(f"Using prompt: {prompt}")
            
            # Generate image URL with working format
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?height=400&width=800&nologo=true"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "image/jpeg, image/png, */*"
            }
            
            print(f"Making request to: {image_url}")
            
            # First make a GET request to generate the image
            response = requests.get(image_url, headers=headers, timeout=30, stream=True)
            
            print(f"Response status: {response.status_code}")
            print(f"Content-Type: {response.headers.get('Content-Type')}")
            
            if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                content_length = len(response.content)
                print(f"Content length: {content_length} bytes")
                
                # Save the image to verify it worked
                filename = "test_devto_image.jpg"
                with open(filename, 'wb') as f:
                    f.write(response.content)
                print(f"Saved test image to: {filename}")
                
                if content_length > 1000:  # Ensure we got a real image
                    print(f"✓ Successfully generated image!")
                    # Check if file exists and has content
                    file_size = os.path.getsize(filename)
                    print(f"File size on disk: {file_size} bytes")
                    return image_url
                else:
                    print(f"✗ Image generation produced too small file: {content_length} bytes")
                    return None
            else:
                print(f"✗ Image generation failed with status code: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"✗ Error generating image: {str(e)}")
            return None

def test():
    print("Testing DevtoPublisher image generation...")
    publisher = DevtoPublisherTest()
    image_url = publisher.generate_image(test_article)
    
    if image_url:
        print("\n✓ SUCCESS: Image generated successfully!")
        print(f"Image URL: {image_url}")
        
        # Update test article with image URL
        test_article['ai_image_url'] = image_url
        
        # Display how it would appear in an article
        print("\nExample markdown for article:")
        print("```markdown")
        print(f"# {test_article['title']}")
        print(f"\n{test_article['summary']}\n")
        print("## AI-Generated Image")
        print(f"![AI-Generated Image]({image_url})")
        print("```")
        
        return True
    else:
        print("\n✗ FAILURE: Could not generate image")
        return False

if __name__ == "__main__":
    test() 