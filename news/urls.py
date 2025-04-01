from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('search/', views.search_news, name='search_news'),
    path('results/', views.results, name='results'),
    path('publish/', views.publish_roundup, name='publish'),
    path('published/<int:pk>/', views.published, name='published'),
    path('history/', views.history, name='history'),
    
    # API endpoints
    path('api/search/', views.api_search, name='api_search'),
    path('api/publish/', views.api_publish, name='api_publish'),
]
