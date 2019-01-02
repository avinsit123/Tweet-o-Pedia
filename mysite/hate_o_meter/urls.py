from django.shortcuts import render
from django.conf.urls import url
from . import views
# Create your views here.
urlpatterns=[
             url('mj/',views.Rendersearch12,name='hate'),
             url('lolo/',views.Rendersearchresult12,name='haadyearch')]
