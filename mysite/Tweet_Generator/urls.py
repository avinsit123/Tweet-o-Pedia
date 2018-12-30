from django.conf.urls import url


from . import views

urlpatterns=[
             url(r'^$',views.Tweet_form,name='Tweetform'),
             url('yumyum',views.ProcessForm,name='vahmc')
             ]

