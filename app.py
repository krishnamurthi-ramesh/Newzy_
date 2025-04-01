from flask import Flask, render_template, request, jsonify
import os
import nltk
import json
from newspaper import Article, Config
from urllib.parse import quote, urlparse
from typing import List, Dict
from duckduckgo_search import DDGS
from datetime import datetime
from deep_translator import GoogleTranslator
from deep_translator.exceptions import RequestError
import time
import re
import unicodedata
from transformers import pipeline
import concurrent.futures
import asyncio
import aiohttp
import functools
import hashlib
from functools import lru_cache
import torch
import requests

# Ensure NLTK data path is properly set
nltk_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nltk_data')
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Pre-download NLTK resources before using them
for resource in ['punkt', 'stopwords', 'averaged_perceptron_tagger']:
    nltk.download(resource, download_dir=nltk_data_path, quiet=True)

# Import NLTK modules after ensuring resources are downloaded
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation

# Model configuration
MODELS = {
    'small': {
        'name': 'facebook/bart-small-cnn',  # Smaller, faster model
        'max_length': 130,
        'min_length': 30,
        'batch_size': 8
    },
    'medium': {
        'name': 'sshleifer/distilbart-cnn-12-6',  # Distilled model, good balance
        'max_length': 130,
        'min_length': 30,
        'batch_size': 6
    }
}

# Choose model based on available resources
def setup_model():
    """Setup the most appropriate model based on available resources"""
    try:
        # Check available memory
        if torch.cuda.is_available():
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU Memory available: {memory_gb:.2f} GB")
            if memory_gb >= 4:
                device = "cuda"
            else:
                device = "cpu"
        else:
            device = "cpu"
            import psutil
            memory_gb = psutil.virtual_memory().available / 1e9
            print(f"CPU Memory available: {memory_gb:.2f} GB")

        # Choose model based on available memory
        if memory_gb >= 4:
            model_config = MODELS['medium']
            print("Using medium-sized model: distilbart-cnn")
        else:
            model_config = MODELS['small']
            print("Using small model: bart-small-cnn")

        # Initialize the model
        print(f"Device set to use {device}")
        summarizer = pipeline(
            "summarization",
            model=model_config['name'],
            device=device,
            framework="pt",
            model_kwargs={"low_cpu_mem_usage": True}
        )
        return summarizer, model_config
    except Exception as e:
        print(f"Error setting up model: {e}")
        # Fallback to smallest model on CPU
        model_config = MODELS['small']
        summarizer = pipeline(
            "summarization",
            model=model_config['name'],
            device="cpu",
            framework="pt",
            model_kwargs={"low_cpu_mem_usage": True}
        )
        return summarizer, model_config

# Initialize the transformer model once globally
transformer_summarizer, MODEL_CONFIG = setup_model()

# Global cache for summaries with TTL
summary_cache = {}
SUMMARY_CACHE_TTL = 3600  # 1 hour in seconds

def get_cached_summary(text: str) -> tuple:
    """Get summary from cache if it exists and is not expired"""
    cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
    if cache_key in summary_cache:
        timestamp, summary = summary_cache[cache_key]
        if time.time() - timestamp < SUMMARY_CACHE_TTL:
            return summary
        else:
            del summary_cache[cache_key]
    return None

def set_cached_summary(text: str, summary: str):
    """Store summary in cache with timestamp"""
    cache_key = hashlib.md5(text.encode('utf-8')).hexdigest()
    summary_cache[cache_key] = (time.time(), summary)

def transformer_summarize(text: str, max_chunk_size: int = 1000) -> str:
    """
    Summarize text with improved caching and chunking
    """
    if not text:
        return ""
    
    # Check cache first
    cached_summary = get_cached_summary(text)
    if cached_summary:
        return cached_summary
    
    # Optimize text for processing
    text = text.replace('\n', ' ').replace('\r', ' ').strip()
    if len(text) <= MODEL_CONFIG['min_length']:
        return text
    
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    # Create optimized chunks
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    summary_text = ""
    try:
        # Process chunks in optimized batches
        batch_size = MODEL_CONFIG['batch_size']
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            summaries = transformer_summarizer(
                batch,
                max_length=MODEL_CONFIG['max_length'],
                min_length=MODEL_CONFIG['min_length'],
                do_sample=False,
                batch_size=len(batch),
                truncation=True
            )
            summary_text += " ".join(s['summary_text'] for s in summaries) + " "
    except Exception as e:
        print(f"Error during transformer summarization: {str(e)}")
        # Fallback to extractive summarization
        summary_text = " ".join(sentences[:3]) + "..."
    
    summary_text = summary_text.strip()
    set_cached_summary(text, summary_text)
    return summary_text

# Initialize Flask app
app = Flask(__name__, 
            template_folder=os.path.join(os.getcwd(), 'templates/news'), 
            static_folder=os.path.join(os.getcwd(), 'static'),
            static_url_path='/static')

# Set a secret key for the Flask application
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'news-genie-secure-key-2025')

# Configure language options
language_map = {
    "en": "English", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "pt": "Portuguese",
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu",
    "ml": "Malayalam", "bn": "Bengali"
}

# Cache for storing article data
article_cache = {}

# ------------------- Helper Functions -------------------
def safe_translate(text, target_language, chunk_size=4900, max_retries=3):
    """
    Translate text in chunks with caching and retry logic
    """
    cache_key = f"{text[:100]}_{target_language}"
    if cache_key in article_cache:
        return article_cache[cache_key]

    translator = GoogleTranslator(source='auto', target=target_language)
    translated_text = ""
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        for attempt in range(max_retries):
            try:
                translated_text += translator.translate(chunk)
                break
            except RequestError as e:
                if attempt == max_retries - 1:
                    translated_text += chunk
                else:
                    time.sleep(0.5)
    
    article_cache[cache_key] = translated_text
    return translated_text

def sanitize_cache_key(key):
    """
    Sanitize cache key to be memcached compatible:
    - Must not include control characters or whitespace
    - Must not include spaces
    - Maximum key length is 250 characters
    - Special characters that are unsafe for memcached are removed or replaced
    """
    # Replace problematic characters
    sanitized = str(key).replace(' ', '_')
    sanitized = ''.join(c for c in sanitized if c.isprintable())
    sanitized = sanitized.replace(':', '_')
    sanitized = sanitized.replace('/', '_')
    sanitized = sanitized.replace('\\', '_')
    sanitized = sanitized.replace('?', '_')
    sanitized = sanitized.replace('*', '_')
    sanitized = sanitized.replace('"', '_')
    sanitized = sanitized.replace('<', '_')
    sanitized = sanitized.replace('>', '_')
    sanitized = sanitized.replace('|', '_')
    
    # If key is too long, hash it
    if len(sanitized) > 250:
        sanitized = hashlib.md5(key.encode('utf-8')).hexdigest()
    
    return sanitized

@lru_cache(maxsize=100)
def get_cached_search_results(query, language):
    """Cache wrapper for search results with sanitized keys"""
    cache_key = sanitize_cache_key(f"search_{query}_{language}")
    return cache_key

# ------------------- NewsSearcher -------------------
class NewsSearcher:
    def __init__(self):
        self.config = Config()
        self.config.browser_user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/91.0.4472.124 Safari/537.36"
        )
        self.config.request_timeout = 10
        self.config.memoize_articles = False
        self.config.http_success_only = False
        self.config.keep_article_html = True
        self.config.http_headers = {
            'User-Agent': self.config.browser_user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
        self.search_settings = {
            'region': 'in-en',
            'safesearch': 'off',
            'timelimit': 'm',
            'max_results': 5
        }
        self.timeout = aiohttp.ClientTimeout(total=10)
        self.session = None
        self.connector = aiohttp.TCPConnector(limit=10, force_close=True)
        self.cache = {}  # In-memory cache

    async def get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(connector=self.connector, timeout=self.timeout)
        return self.session

    async def fetch_article_async(self, url: str) -> dict:
        """Optimized article fetching with better error handling"""
        if url in article_cache:
            return article_cache[url]

        try:
            session = await self.get_session()
            config = Config()
            config.browser_user_agent = self.config.browser_user_agent
            config.request_timeout = self.config.request_timeout
            config.memoize_articles = self.config.memoize_articles
            config.http_success_only = self.config.http_success_only
            config.keep_article_html = self.config.keep_article_html
            config.http_headers = self.config.http_headers
            
            article = Article(url, config=config)
            
            async with session.get(url, headers=self.config.http_headers) as response:
                if response.status == 403:
                    # Handle forbidden access
                    domain = urlparse(url).netloc
                    return {
                        'title': f"Article from {domain}",
                        'text': "This article requires authentication or is not publicly accessible.",
                        'summary': "Article preview not available. Please visit the original source.",
                        'url': url,
                        'publish_date': None,
                        'image_url': None,
                        'source': domain
                    }
                elif response.status != 200:
                    raise Exception(f"HTTP {response.status}")
                
                html_content = await response.text()
                
            # Parse in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            article.download_state = 2
            article.html = html_content
            await loop.run_in_executor(None, article.parse)
            
            result = {
                'title': article.title or f"Article from {urlparse(url).netloc}",
                'text': article.text.replace('\n', ' ').replace('\r', '') if article.text else "Article content not available.",
                'url': url,
                'publish_date': article.publish_date or article.meta_data.get('article:published_time'),
                'image_url': article.top_image,
                'source': article.source_url or urlparse(url).netloc
            }
            
            # Generate summary if text is available
            if result['text'] and result['text'] != "Article content not available.":
                result['summary'] = transformer_summarize(result['text'])
            else:
                result['summary'] = "Article preview not available. Please visit the original source."
            
            article_cache[url] = result
            return result
            
        except Exception as e:
            print(f"Error processing article: {str(e)} on URL {url}")
            domain = urlparse(url).netloc
            result = {
                'title': f"Article from {domain}",
                'text': "This article could not be retrieved. You can visit the original source for the complete information.",
                'summary': "Article preview not available. Please visit the original source.",
                'url': url,
                'publish_date': None,
                'image_url': None,
                'source': domain
            }
            article_cache[url] = result
            return result

    async def search_news(self, query: str, location: str = None) -> List[Dict]:
        # Generate cache key
        cache_key = get_cached_search_results(f"{query}_{location}", "en")
        
        # Check in-memory cache first
        if cache_key in self.cache:
            print(f"Cache hit for {cache_key}")
            return self.cache[cache_key]

        try:
            keywords = f"{query} {location} news -site:msn.com -site:usnews.com" if location else f"{query} news -site:msn.com -site:usnews.com"
            keywords = keywords.strip().replace("  ", " ")
            
            # Get search results with timeout
            with DDGS() as ddgs:
                results = list(ddgs.news(
                    keywords=keywords,
                    region=self.search_settings['region'],
                    safesearch=self.search_settings['safesearch'],
                    timelimit=self.search_settings['timelimit'],
                    max_results=self.search_settings['max_results']
                ))

            # Process articles concurrently
            tasks = []
            articles = []
            
            for result in results:
                if result['url'] in article_cache:
                    articles.append(article_cache[result['url']])
                else:
                    tasks.append(self.fetch_article_async(result['url']))

            if tasks:
                completed_articles = await asyncio.gather(*tasks, return_exceptions=True)
                articles.extend([a for a in completed_articles if isinstance(a, dict)])

            # Cache the results
            self.cache[cache_key] = articles
            return articles

        except Exception as e:
            print(f"Error in DuckDuckGo news search: {str(e)}")
            return []

    async def cleanup(self):
        if self.session:
            await self.session.close()

# ------------------- NewsProcessor -------------------
class NewsProcessor:
    def __init__(self):
        self.stopwords = set(stopwords.words('english') + list(punctuation))
        self.max_batch_size = MODEL_CONFIG['batch_size']
        self.summarizer_cache = {}

    async def process_articles(self, articles: List[Dict]) -> List[Dict]:
        """Optimized batch processing with improved caching"""
        processed_articles = []
        summarization_batch = []
        
        for article in articles:
            # Check if article is already processed and cached
            cache_key = article.get('url', '')
            if cache_key in article_cache and 'summary' in article_cache[cache_key]:
                processed_articles.append(article_cache[cache_key])
                continue
            
            # Add to summarization batch if needed
            if article.get('text'):
                cached_summary = get_cached_summary(article['text'])
                if cached_summary:
                    article['summary'] = cached_summary
                    article_cache[cache_key] = article
                    processed_articles.append(article)
                else:
                    summarization_batch.append(article)
            else:
                # Handle articles without text
                article['summary'] = "Article text not available."
                processed_articles.append(article)
        
        # Process summarization batch if any
        if summarization_batch:
            batch_results = await self.process_batch(summarization_batch)
            processed_articles.extend(batch_results)
        
        return processed_articles

    async def process_batch(self, articles: List[Dict]) -> List[Dict]:
        """Process batch of articles with optimized summarization"""
        loop = asyncio.get_event_loop()
        processed_batch = []
        
        # Group articles into smaller batches for efficient processing
        for i in range(0, len(articles), self.max_batch_size):
            batch = articles[i:i + self.max_batch_size]
            texts = [article['text'] for article in batch]
            
            try:
                # Process batch of summaries
                summaries = await loop.run_in_executor(
                    None,
                    lambda: transformer_summarizer(
                        texts,
                        max_length=MODEL_CONFIG['max_length'],
                        min_length=MODEL_CONFIG['min_length'],
                        do_sample=False,
                        batch_size=len(texts),
                        truncation=True
                    )
                )
                
                # Update articles with summaries
                for article, summary in zip(batch, summaries):
                    article['summary'] = summary['summary_text']
                    cache_key = article.get('url', '')
                    if cache_key:
                        article_cache[cache_key] = article
                    processed_batch.append(article)
            except Exception as e:
                print(f"Error in batch summarization: {str(e)}")
                # Fallback to simple extractive summarization
                for article in batch:
                    sentences = sent_tokenize(article['text'])
                    article['summary'] = " ".join(sentences[:3]) + "..."
                    processed_batch.append(article)
        
        return processed_batch

# ------------------- DevtoPublisher -------------------
class DevtoPublisher:
    def __init__(self):
        self.api_key = os.environ.get('DEVTO_API_KEY', 'your_devto_api_key')
        self.api_url = "https://dev.to/api/articles"
        self.headers = {
            'api-key': self.api_key,
            'Content-Type': 'application/json'
        }

    def generate_image(self, article: dict) -> str:
        """Generate image using Pollinations.ai"""
        try:
            # Get available models first
            models_url = "https://image.pollinations.ai/models"
            response = requests.get(models_url, timeout=10)
            if response.status_code == 200:
                models = response.json()
                # Use the first available model (usually 'flux')
                selected_model = models[0] if models else "flux"
            else:
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
            
            # Generate image URL with working format
            image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?height=400&width=800&nologo=true"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "image/jpeg, image/png, */*"
            }
            
            # First make a GET request to generate the image
            response = requests.get(image_url, headers=headers, timeout=30, stream=True)
            
            if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
                content_length = len(response.content)
                if content_length > 1000:  # Ensure we got a real image
                    print(f"Successfully generated image for '{prompt[:30]}...' ({content_length} bytes)")
                    return image_url
                else:
                    print(f"Image generation produced too small file: {content_length} bytes")
                    return None
            else:
                print(f"Image generation failed with status code: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"Error generating image: {str(e)}")
            return None

    def publish_article(self, articles, topic: str, location: str = None, language: str = "en") -> dict:
        """Publish article to Dev.to"""
        try:
            # Generate images for articles
            for article in articles:
                ai_image = self.generate_image(article)
                if ai_image:
                    article['ai_image_url'] = ai_image

            # Create article content
            content = self.format_content(articles, topic, location, language)
            
            # Create article data
            article_data = {
                "article": {
                    "title": f"News Roundup: {topic.title()}" + (f" in {location.title()}" if location else ""),
                    "body_markdown": content,
                    "published": True,
                    "tags": ["news", "ai", "technology"],
                    "series": "AI News Roundup"
                }
            }

            # Publish to Dev.to
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=article_data
            )

            if response.status_code == 201:
                result = response.json()
                return {
                    'success': True,
                    'url': result.get('url'),
                    'title': result.get('title')
                }
            else:
                print(f"Dev.to API Error: {response.status_code}\n{response.text}")
                return None

        except Exception as e:
            print(f"Error publishing to Dev.to: {str(e)}")
            return None

    def format_content(self, articles, topic: str, location: str = None, language: str = "en") -> str:
        """Format content for Dev.to article"""
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Create introduction
        content = f"# News Roundup: {topic.title()}"
        if location:
            content += f" in {location.title()}"
        content += f"\n\n*Published on {current_date}*\n\n"
        
        # Add introduction section
        content += "## Introduction\n"
        content += f"Welcome to our AI-curated news roundup about **{topic}**"
        if location:
            content += f" in **{location}**"
        content += ". This post aggregates multiple sources and includes AI-generated illustrations.\n\n"
        
        # Add combined summary
        combined_text = " ".join(article.get('summary', '') for article in articles if article.get('summary'))
        if combined_text:
            content += "## Overview\n"
            content += combined_text + "\n\n"
        
        # Add detailed articles
        content += "## Featured Articles\n\n"
        for idx, article in enumerate(articles, 1):
            title = article.get('title', '').strip() or f"Article #{idx}"
            content += f"### {idx}. {title}\n\n"
            
            # Add source information
            source_name = article.get('source', 'Unknown Source')
            source_url = article.get('url', '')
            content += f"**Source**: {source_name}\n\n"
            if source_url:
                content += f"**Read Full Article**: [Link]({source_url})\n\n"
            
            # Add article summary
            if article.get('summary'):
                content += f"**Summary**:\n\n{article['summary']}\n\n"
            
            # Add images
            if article.get('image_url'):
                content += "**Original Image**:\n\n"
                content += f"![Original Article Image]({article['image_url']})\n\n"
            
            if article.get('ai_image_url'):
                content += "**AI-Generated Illustration**:\n\n"
                content += f"![AI Generated Illustration]({article['ai_image_url']})\n\n"
                content += "*AI-generated image related to this article.*\n\n"
            
            content += "---\n\n"
        
        # Add footer
        content += "\n\n---\n"
        content += "*This news roundup was automatically curated and published using AI. "
        content += f"Last updated: {current_date}*"
        
        # Translate if needed
        if language != "en":
            content = safe_translate(content, language)
        
        return content

# --------------------- Routes ---------------------
@app.route('/')
def index():
    return render_template('index.html', languages=language_map)

@app.route('/search', methods=['POST'])
async def search_news():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        location = data.get('location', '').strip()
        language = data.get('language', 'en').strip()
    
        if not query:
            return jsonify({'error': 'Please provide a search query'}), 400

        news_searcher = NewsSearcher()
        news_processor = NewsProcessor()

        try:
            # Search and process articles concurrently
            articles = await news_searcher.search_news(query, location)
            
            if not articles:
                return jsonify({
                    'articles': [],
                    'query': query,
                    'location': location,
                    'language': language,
                    'language_name': language_map.get(language, 'Unknown')
                })

            # Process articles in optimized batches
            processed_articles = await news_processor.process_articles(articles)

            # Parallel translation if needed
            if language != 'en':
                translation_tasks = []
                for article in processed_articles:
                    cache_key = f"{article['url']}_{language}"
                    if cache_key in article_cache:
                        article.update(article_cache[cache_key])
                    else:
                        tasks = []
                        if article.get('title'):
                            tasks.append(('title', article['title']))
                        if article.get('text'):
                            tasks.append(('text', article['text']))
                        if article.get('summary'):
                            tasks.append(('summary', article['summary']))
                        if tasks:
                            translation_tasks.append((article, tasks))

                if translation_tasks:
                    async with aiohttp.ClientSession() as session:
                        for article, tasks in translation_tasks:
                            translations = await asyncio.gather(*[
                                loop.run_in_executor(None, safe_translate, text, language)
                                for field, text in tasks
                            ])
                            for (field, _), translation in zip(tasks, translations):
                                article[field] = translation
                            article_cache[f"{article['url']}_{language}"] = article
        
            return jsonify({
                'articles': processed_articles,
                'query': query,
                'location': location,
                'language': language,
                'language_name': language_map.get(language, 'Unknown')
            })

        finally:
            await news_searcher.cleanup()

    except Exception as e:
        print(f"Error in search_news: {str(e)}")
        return jsonify({'error': 'An error occurred while processing your request'}), 500

@app.route('/publish', methods=['POST'])
def publish_article():
    data = request.json
    articles = data.get('articles', [])
    topic = data.get('topic', '')
    location = data.get('location', '')
    language = data.get('language', 'en')
    
    if not articles or not topic:
        return jsonify({
            'success': False,
            'error': 'Missing required parameters.'
        }), 400
    
    try:
        publisher = DevtoPublisher()
        result = publisher.publish_article(articles, topic, location, language)
        
        if result and result.get('success'):
            return jsonify({
                'success': True,
                'article_url': result.get('url'),
                'article_title': result.get('title')
            })
        else:
            error_msg = 'Failed to publish article. Please check your Dev.to API key and try again.'
            print(f"Publishing error: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
    
    except Exception as e:
        error_msg = str(e)
        print(f"Error publishing article: {error_msg}")
        return jsonify({
            'success': False,
            'error': f'Error publishing article: {error_msg}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
