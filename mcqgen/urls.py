"""mcqgen URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from mcqgeneration import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.signin, name='signin'),
    path('signup/', views.signup, name='signup'),
    path('base/',views.base,name='base'),
    path('logout/', views.logout_view, name='logout'),
    path('clear/', views.clear, name='clear'),
    path('index/', views.AiChat, name='index'),
    path('upload_pdf/',views.upload_document,name='upload_pdf'),
    path('mcqs_gen/',views.read_pdf,name='mcqs_gen'),
    path('mcq_result/',views.mcq_result,name='mcq_result'),
]
