from django.db import models

# Create your models here.
class Message(models.Model):
    content = models.CharField(max_length=255)
    
class React(models.Model):
    input = models.CharField(max_length=30)
    output = models.CharField(max_length=30)