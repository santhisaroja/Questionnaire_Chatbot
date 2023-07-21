# forms.py

from django import forms
from django.contrib.auth.models import User
from .models import Document

class SignUpForm(forms.ModelForm):
    username = forms.CharField(widget=forms.TextInput(attrs={'class':'form-control form-control-lg'}), max_length=30, required=True)
    email = forms.CharField(
        widget=forms.EmailInput(attrs={'class':'form-control form-control-lg'}), max_length=100, required=True)
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'class':'form-control form-control-lg'}), required=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password',]
        widgets = {
            'password': forms.PasswordInput(),
        }




class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ('pdf_file',)
        
class UserForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    email = forms.EmailField(max_length=254, help_text='Required field')
    class Meta:
        model = User
        fields = ['username','email','password']

