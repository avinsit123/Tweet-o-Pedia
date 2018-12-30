from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.
from django.shortcuts import render

def Startpage(request):
    return  render(request,'starter.html')
