from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url
from .views import index, detectar, treinar, create_dataset

urlpatterns = [
    path('admin/', admin.site.urls),
    url(r'^$', index),
    url(r'^create_dataset$', create_dataset),
    url(r'^detectar$', detectar),
    url(r'^treinar$', treinar),
]
