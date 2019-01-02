from django.shortcuts import render
from django.conf.urls import url
from . import views
# Create your views here.
urlpatterns=[
             url(r'^$',views.Rendersearch1,name='qbs'),
             url('lolo/',views.Rendersearchresult1,name='qbsresult')]
