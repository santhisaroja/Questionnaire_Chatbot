from django.db import models
from django.contrib.auth.models import User

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete= models.CASCADE)
    profile_photo = models.FileField(default='default.jpg', upload_to='profile_photos')
    status_info = models.CharField(default="Enter status", max_length=1000) 


    def _str_(self):
        return f'{self.user.username} Profile'

class Chat(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    user_input = models.TextField( verbose_name="User Input", null=True)
    ai_response = models.TextField( verbose_name="User Input", null=True)
    timestamp = models.DateTimeField(auto_now_add=True)


class Meta:
        verbose_name = 'Chat'
        verbose_name_plurasignupl = 'Chats'


class Document(models.Model):
    pdf_file = models.FileField(upload_to='pdfs/')