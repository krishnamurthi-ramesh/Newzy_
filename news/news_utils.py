import os
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from newspaper import Article as NewsArticle
from .models import Article
from deep_translator import GoogleTranslator
from deep_translator.exceptions import RequestError
import time
from duckduckgo_search import DDGS
import json
from transformers import pipeline
import traceback

# Configure language options - Limited to 10 most commonly used languages
LANGUAGE_MAP = {
    "en": "English", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "pt": "Portuguese",
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "ml": "Malayalam", "bn": "Bengali"
}

def safe_translate(text, target_language, chunk_size=4900, max_retries=3):
    """
    Translate text in chunks to avoid deep_translator length limits.
    Retries translation up to max_retries times. On failure, returns the original chunk.
    """
    if not text or target_language == "en":
        return text
        
    translator = GoogleTranslator(source='auto', target=target_language)
    translated_text = ""
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        for attempt in range(max_retries):
            try:
                translated_text += translator.translate(chunk)
                break  # Break out of retry loop on success
            except RequestError as e:
                if attempt == max_retries - 1:
                    translated_text += chunk  # Fallback: append original text
                else:
                    time.sleep(1)  # Wait before retrying
    return translated_text

class NewsSearcher:
    def __init__(self):
        self.config = {
            'region': 'wt-wt',  # Worldwide
            'safesearch': 'off',
            'timelimit': 'w',  # Last week
            'max_results': 10
        }

    def search_news(self, query, location=''):
        try:
            with DDGS() as ddgs:
                search_query = f"{query}"
                if location:
                    search_query += f" location:{location}"
                
                results = []
                try:
                    for r in ddgs.news(search_query, **self.config):
                        # Handle potential missing keys with defaults
                        article = {
                            'title': r.get('title', 'No Title'),
                            'url': r.get('url', r.get('link', '')),  # Try both url and link
                            'source': r.get('source', 'Unknown Source'),
                            'publish_date': r.get('date', r.get('published', '')),  # Try both date and published
                            'snippet': r.get('excerpt', r.get('body', 'No excerpt available'))  # Try both excerpt and body
                        }
                        
                        # Only add if we have a URL
                        if article['url']:
                            results.append(article)
                            
                        # Break after getting enough results
                        if len(results) >= self.config['max_results']:
                            break
                except Exception as e:
                    print(f"Error during DuckDuckGo search: {str(e)}")
                    traceback.print_exc()
                
                return results
        except Exception as e:
            print(f"Error during news search: {str(e)}")
            traceback.print_exc()
            return []

class NewsProcessor:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def process_article(self, article_data):
        try:
            # Add modern user-agent and headers to avoid 403 errors
            article = NewsArticle(article_data['url'])
            article.config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'
            article.config.headers = {
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://www.google.com'
            }
            
            # Add delay to avoid rate limiting
            time.sleep(2)
            article.download()
            article.parse()
            
            text = article.text or article_data.get('snippet', 'No content available')
            
            # Generate summary if text is available
            summary = text
            if text and text != 'No content available':
                # Determine appropriate max_length based on input length
                input_length = len(text.split())
                max_length = min(130, max(30, input_length // 2))
                
                # Split text into chunks of 1024 tokens for BART model
                chunks = [text[i:i + 1024] for i in range(0, len(text), 1024)]
                summaries = []
                for chunk in chunks:
                    if len(chunk.split()) > 50:  # Only summarize chunks with sufficient content
                        try:
                            summary_output = self.summarizer(chunk, max_length=max_length, min_length=min(30, max_length-10), do_sample=False)
                            summaries.append(summary_output[0]['summary_text'])
                        except Exception as e:
                            print(f"Warning: Summarization failed for chunk: {str(e)}")
                            summaries.append(chunk[:200] + "...")
                
                summary = ' '.join(summaries) if summaries else text[:500] + "..."
            
            # Clean up title
            title = article.title or article_data.get('title', 'No Title')
            title = ''.join(char for char in title if char.isprintable())
            
            return {
                'title': title,
                'text': text,
                'summary': summary,
                'image_url': article.top_image or '',
                'publish_date': article.publish_date.isoformat() if article.publish_date else article_data.get('publish_date', ''),
                'authors': article.authors or [],
                'url': article_data['url'],
                'source': article_data.get('source', 'Unknown Source')
            }
        except Exception as e:
            print(f"Error processing article: {str(e)}")
            return {
                'title': article_data.get('title', 'No Title'),
                'text': article_data.get('snippet', 'No content available'),
                'summary': article_data.get('snippet', 'No content available'),
                'image_url': '',
                'publish_date': article_data.get('publish_date', ''),
                'authors': [],
                'url': article_data['url'],
                'source': article_data.get('source', 'Unknown Source')
            }

class HashnodePublisher:
    def __init__(self, api_key=None):
        self.api_url = "https://api.hashnode.com/"
        api_token = api_key or os.getenv('HASHNODE_API_TOKEN')
        if not api_token:
            raise ValueError("No Hashnode API token provided")
            
        self.headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }
        
        self.publication_id = os.getenv('HASHNODE_PUBLICATION_ID')
        if not self.publication_id:
            raise ValueError("No Hashnode publication ID provided")
        
    def publish_article(self, title, content, tags=None, cover_image_url=None):
        """
        Publish an article to Hashnode using their GraphQL API
        
        Args:
            title (str): The title of the article
            content (str): The markdown content of the article
            tags (list): List of tags for the article
            cover_image_url (str): URL of the cover image for the article
            
        Returns:
            str: URL of the published article or None if publishing failed
        """
        if not title or not content:
            raise ValueError("Title and content are required")
            
        if tags is None:
            tags = ["news"]
            
        # GraphQL mutation for publishing an article
        mutation = """
        mutation CreatePost($input: CreatePostInput!) {
          createPost(input: $input) {
            post {
              id
              title
              slug
              url
              brief
              coverImage
            }
          }
        }
        """
        
        # Prepare variables for the GraphQL mutation
        variables = {
            "input": {
                "title": title,
                "contentMarkdown": content,
                "tags": tags,
                "coverImageURL": cover_image_url,
                "publication": {
                    "id": self.publication_id
                }
            }
        }
        
        # Print debug information
        print("\nHashnode API Request:")
        print(f"URL: {self.api_url}")
        print(f"Headers: {json.dumps(self.headers, indent=2)}")
        print(f"Variables: {json.dumps(variables, indent=2)}")
        
        # Make the API request
        try:
            response = requests.post(
                self.api_url,
                json={"query": mutation, "variables": variables},
                headers=self.headers
            )
            
            # Print response for debugging
            print("\nHashnode API Response:")
            print(f"Status Code: {response.status_code}")
            print(f"Headers: {json.dumps(dict(response.headers), indent=2)}")
            print(f"Body: {response.text}")
            
            response.raise_for_status()
            
            result = response.json()
            
            # Check for errors
            if "errors" in result:
                error_msg = json.dumps(result['errors'], indent=2)
                print(f"\nHashnode API Error:\n{error_msg}")
                raise Exception(f"Hashnode API Error: {error_msg}")
                
            # Extract the URL of the published article
            post_data = result.get("data", {}).get("createPost", {}).get("post", {})
            article_url = post_data.get("url")
            
            if article_url:
                print(f"\nSuccessfully published to Hashnode. Article URL: {article_url}")
            else:
                raise Exception("Failed to get article URL from API response")
            
            return article_url
            
        except requests.exceptions.RequestException as e:
            print(f"\nNetwork error: {str(e)}")
            raise
        except Exception as e:
            print(f"\nError publishing to Hashnode: {str(e)}")
            raise

class DevToAPI:
    def __init__(self, api_key=None):
        self.base_url = "https://dev.to/api"
        self.api_key = api_key or os.getenv('DEVTO_API_KEY')
        if not self.api_key:
            raise ValueError("No Dev.to API key provided")
        
        self.headers = {
            'api-key': self.api_key,
            'Content-Type': 'application/json',
            'Accept': 'application/vnd.forem.api-v1+json'
        }

    def create_post(self, title, content, tags=None, series=None):
        """
        Create a post on Dev.to
        
        Args:
            title (str): The title of the article
            content (str): The markdown content of the article
            tags (list): List of tags for the article
            series (str): Optional series name
            
        Returns:
            str: URL of the published article or None if publishing failed
        """
        if not title or not content:
            raise ValueError("Title and content are required")

        if tags is None:
            tags = ["news"]

        article_data = {
            "article": {
                "title": title,
                "body_markdown": content,
                "published": True,
                "tags": tags
            }
        }

        if series:
            article_data["article"]["series"] = series

        try:
            response = requests.post(
                f"{self.base_url}/articles",
                json=article_data,
                headers=self.headers
            )
            response.raise_for_status()
            
            # Get the published URL from response
            result = response.json()
            if "url" in result:
                return result["url"]
            else:
                print(f"Warning: No URL in Dev.to response: {result}")
                return None

        except requests.exceptions.RequestException as e:
            error_message = str(e)
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    error_message = f"{error_message} - {json.dumps(error_details)}"
                except:
                    error_message = f"{error_message} - {e.response.text}"
            print(f"Error publishing to Dev.to: {error_message}")
            return None

    def create_roundup_post(self, articles, topic, location=''):
        """
        Create a news roundup post from processed articles
        """
        # Create title
        date_str = datetime.now().strftime("%B %d, %Y")
        title = f"News Roundup: {topic.title()}"
        if location:
            title += f" in {location.title()}"
        title += f" - {date_str}"
        
        # Create content
        content = f"# {title}\n\n"
        content += "Here's your AI-curated news roundup for today:\n\n"
        
        # Add each article
        for i, article in enumerate(articles, 1):
            content += f"## {i}. {article['title']}\n\n"
            content += f"*Source: {article['source']}*\n\n"
            content += f"{article['summary']}\n\n"
            if article['image_url']:
                content += f"![Article Image]({article['image_url']})\n\n"
            content += f"[Read full article]({article['url']})\n\n"
            content += "---\n\n"
        
        # Add footer
        content += "\n\n*This roundup was automatically generated by SmartScrapAI*"
        
        return self.create_post(title, content, tags=["news", "ai", topic.lower().replace(" ", "")])