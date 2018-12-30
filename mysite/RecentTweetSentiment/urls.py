from django.shortcuts import render

from django.conf.urls import url


from . import views

# Create your views here.
urlpatterns=[
             url(r'^$',views.Rendersearch,name='Asd'),
             url('yoyo/',views.Rendersearchresult,name='search_result')]

# Create your views here.
