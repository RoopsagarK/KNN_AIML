from django.urls import path
from .views import knn

urlpatterns = [
    path('', knn, name='knn'),
]
