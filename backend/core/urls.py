"""
URL configuration for ArXiv Classifier API.
"""
from django.urls import path, include

urlpatterns = [
    path('api/v1/', include('api.urls')),
]
