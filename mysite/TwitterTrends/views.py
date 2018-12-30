from django.shortcuts import render

from django.shortcuts import render
import tweepy
from tweepy.auth import OAuthHandler
from django.http import HttpResponse
import json

consumer_key="liABpG2XM62tIo02fjen7FDVJ"
consumer_secret="nUI6fH6PJuuUvQFlzylCieFxOZ8VZdf5yGrRGK294whWEenayA"
access_token="1046393582469500928-a4Il71HUPkyF0MtoS3V6dJL1igx84k"
access_secret="VcGhV2AcXHQISVXwYuOhmIVkvxaMuPnWlhnGS4LmwEkDN"


        
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

class Trend:
    def __int__(self,Tweet_text,Tweet_url,Tweet_user):
        self.Hashtag = Tweet_text
        self.url = Tweet_url
        self.volume = Tweet_user

# Create your views here.

def Trends(request):
    trend_list = []
    
    for niggas in api.trends_place(2442047):
         for trends in niggas['trends']:
             trend = Trend()
             trend.Hashtag,trend.url,trend.volume = trends['name'] ,trends['url'],trends['tweet_volume'] 
             print(trend.Hashtag,trend.url,trend.volume)
             trend_list.append(trend)
             
    return render(request,'Trendy.html',{ "trends" : trend_list })
    
    
if __name__=='__main__':
     for trends in api.trends_place(1):
         for niggas in trends['trends']:
             print((niggas['name']))