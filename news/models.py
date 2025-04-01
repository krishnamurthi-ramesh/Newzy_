import os
from django.db import models
from django.utils import timezone
import requests
import json
from PIL import Image
import io
from django.core.files.base import ContentFile
from urllib.parse import quote

# Create your models here.

# AI Image Generation Settings
AI_IMAGE_API = "https://image.pollinations.ai/prompt/"
AI_IMAGE_KEY = os.getenv('POLLINATIONS_API_KEY', 'your_pollinations_api_key')  # Will be replaced with actual API key

class Article(models.Model):
    title = models.CharField(max_length=255)
    url = models.URLField()
    source = models.CharField(max_length=100)
    publish_date = models.DateTimeField(null=True, blank=True)
    text = models.TextField(blank=True)
    summary = models.TextField(blank=True)
    image_url = models.URLField(blank=True, null=True)
    ai_image_url = models.URLField(blank=True, null=True)
    ai_image_prompt = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(default=timezone.now)

    def generate_ai_image(self):
        try:
            # Create a descriptive prompt for the AI image generator
            prompt = f"A visually appealing illustration of {self.title}. Show key elements from the article content in a modern, professional style."
            
            # URL encode the prompt
            encoded_prompt = quote(prompt, safe='')
            
            # Construct the image URL
            self.ai_image_url = f"{AI_IMAGE_API}{encoded_prompt}"
            self.ai_image_prompt = prompt
            self.save()
            return True
            
        except Exception as e:
            print(f"Error generating AI image: {str(e)}")
            return False

    def __str__(self):
        return self.title

class SearchQuery(models.Model):
    query = models.CharField(max_length=100)
    location = models.CharField(max_length=100, blank=True)
    language = models.CharField(max_length=10, default='en')
    results_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.query} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Search queries'

class PublishedRoundup(models.Model):
    """Model for storing published news roundups"""
    topic = models.CharField(max_length=200)
    location = models.CharField(max_length=200, blank=True)
    article_count = models.IntegerField(default=0)
    devto_url = models.URLField(max_length=500, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Published Roundup'
        verbose_name_plural = 'Published Roundups'

    def __str__(self):
        return f"{self.topic} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
