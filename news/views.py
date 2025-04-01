from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.core.cache import cache
import json
import traceback
import os
from django.conf import settings
from django.urls import reverse
from django.utils import timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from urllib.parse import quote
import requests
from .news_utils import (
    NewsSearcher, 
    NewsProcessor, 
    DevToAPI,
    safe_translate,
    LANGUAGE_MAP
)
from .models import Article, SearchQuery, PublishedRoundup
import time

def process_single_article(article, processor, language):
    """Process a single article with translation if needed"""
    try:
        processed = processor.process_article(article)
        if processed and language != "en":
            for key in ['title', 'text']:
                if processed.get(key):
                    processed[key] = safe_translate(processed[key], language)
        return processed
    except Exception as e:
        print(f"Error processing article {article.get('url', 'unknown')}: {str(e)}")
        return None

def index(request):
    """Home page with search form"""
    query = request.GET.get('query', '').strip()
    location = request.GET.get('location', '').strip()
    language = request.GET.get('language', 'en')
    
    # If we have search parameters, handle the search
    if query:
        return search_news(request)
    
    # Otherwise show the index page
    languages = LANGUAGE_MAP
    recent_searches = SearchQuery.objects.order_by('-created_at')[:6]
    
    context = {
        'languages': languages,
        'recent_searches': recent_searches,
    }
    
    return render(request, 'news/index.html', context)

def search_news(request):
    """Handle news search and display results"""
    if request.method == 'GET':
        query = request.GET.get('query', '').strip()
        location = request.GET.get('location', '').strip()
        language = request.GET.get('language', 'en')
        
        if query:
            try:
                # Check cache first
                cache_key = f"search_{query}_{location}_{language}"
                cached_results = cache.get(cache_key)
                
                if cached_results:
                    context = {
                        'articles': cached_results,
                        'query': query,
                        'location': location,
                        'language': language,
                        'from_cache': True
                    }
                    return render(request, 'news/results.html', context)
                
                # Search for news
                searcher = NewsSearcher()
                processor = NewsProcessor()
                
                # Get raw articles
                raw_articles = searcher.search_news(query, location)
                
                if not raw_articles:
                    messages.warning(request, 'No articles found for your search query.')
                    return redirect('index')
                
                # Process articles in parallel with timeout
                articles = []
                with ThreadPoolExecutor(max_workers=5) as executor:
                    # Submit all tasks
                    future_to_article = {
                        executor.submit(process_single_article, article, processor, language): article 
                        for article in raw_articles[:10]  # Limit to first 10 articles
                    }
                    
                    # Get results as they complete
                    for future in as_completed(future_to_article):
                        processed = future.result()
                        if processed:
                            articles.append(processed)
                
                if not articles:
                    messages.warning(request, 'No articles could be processed.')
                    return redirect('index')
                
                # Save search query
                SearchQuery.objects.create(
                    query=query,
                    location=location,
                    language=language,
                    results_count=len(articles)
                )
                
                # Cache the results
                cache.set(cache_key, articles, 3600)  # Cache for 1 hour
                
                context = {
                    'articles': articles,
                    'query': query,
                    'location': location,
                    'language': language,
                    'from_cache': False
                }
                return render(request, 'news/results.html', context)
                
            except Exception as e:
                print(f"Error in search_news: {str(e)}")
                traceback.print_exc()
                messages.error(request, 'An error occurred while searching for news.')
                return redirect('index')
    
    return redirect('index')

def results(request):
    """Display search results"""
    # This is a fallback in case someone navigates directly to /results/
    return redirect('index')

@csrf_exempt
@require_http_methods(["POST"])
def publish_roundup(request):
    """Handle publishing a news roundup"""
    try:
        data = json.loads(request.body)
        articles = data.get('articles', [])
        topic = data.get('topic', '')
        location = data.get('location', '')
        
        if not articles:
            return JsonResponse({'error': 'No articles provided'}, status=400)
            
        # Initialize Dev.to API client
        devto_api = DevToAPI()
        
        try:
            # Create and publish the roundup
            url = devto_api.create_roundup_post(articles, topic, location)
            
            if url:
                # Save the published roundup
                PublishedRoundup.objects.create(
                    topic=topic,
                    location=location,
                    article_count=len(articles),
                    devto_url=url  # Store the Dev.to URL
                )
                
                return JsonResponse({'success': True, 'url': url})
            else:
                return JsonResponse({'error': 'Failed to publish to Dev.to'}, status=500)
                
        except Exception as e:
            print(f"Error publishing roundup: {str(e)}")
            traceback.print_exc()
            return JsonResponse({'error': str(e)}, status=500)
            
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON data'}, status=400)
    except Exception as e:
        print(f"Error in publish_roundup: {str(e)}")
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)

def published(request, pk):
    """View a published roundup"""
    roundup = get_object_or_404(PublishedRoundup, pk=pk)
    
    return render(request, 'news/published.html', {
        'roundup': roundup,
        'languages': LANGUAGE_MAP
    })

def history(request):
    """Show search history and published roundups"""
    searches = SearchQuery.objects.order_by('-created_at')[:20]
    roundups = PublishedRoundup.objects.order_by('-created_at')[:10]
    
    return render(request, 'news/history.html', {
        'searches': searches,
        'roundups': roundups
    })

@csrf_exempt
def api_search(request):
    """API endpoint for searching news"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        query = data.get('query', '').strip()
        location = data.get('location', '').strip()
        language = data.get('language', 'en')
        
        if not query:
            return JsonResponse({'error': 'Query is required'}, status=400)
        
        searcher = NewsSearcher()
        processor = NewsProcessor()
        
        articles = searcher.search_news(query, location)
        
        if not articles:
            return JsonResponse({'error': 'No articles found'}, status=404)
        
        processed_articles = []
        for article in articles[:10]:
            processed = processor.process_article(article)
            if processed:
                processed_articles.append(processed)
        
        return JsonResponse({'articles': processed_articles})
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
def api_publish(request):
    """API endpoint for publishing a roundup"""
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        title = data.get('title', '').strip()
        articles = data.get('articles', [])
        
        if not title:
            return JsonResponse({'error': 'Title is required'}, status=400)
        
        if not articles:
            return JsonResponse({'error': 'At least one article is required'}, status=400)
        
        # Implement publishing logic
        
        return JsonResponse({'success': True})
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
