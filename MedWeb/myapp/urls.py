from django.contrib import admin
from django.urls import path, include
from django.conf.urls import url
from myapp.views import *
from rest_framework import routers, serializers, viewsets

urlpatterns = [
    path('', ReactView.as_view(), name='anything'),

]