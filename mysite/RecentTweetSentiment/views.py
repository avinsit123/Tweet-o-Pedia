
from django.shortcuts import render
import tweepy
from tweepy.auth import OAuthHandler
from django.http import HttpResponse
import json

consumer_key="liABpG2XM62tIo02fjen7FDVJ"
consumer_secret="nUI6fH6PJuuUvQFlzylCieFxOZ8VZdf5yGrRGK294whWEenayA"
access_token="1046393582469500928-a4Il71HUPkyF0MtoS3V6dJL1igx84k"
access_secret="VcGhV2AcXHQISVXwYuOhmIVkvxaMuPnWlhnGS4LmwEkDN"




class Tweet:
    def __int__(self,Tweet_text,Tweet_url,Tweet_user):
        self.Tweet_text = Tweet_text
        self.Tweet_url = Tweet_url
        self.Tweet_user = Tweet_user
        
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

def process_or_store(tweet):
    print(json.dumps(tweet))
    
def Query_tweets(api,query):
    tweetad = []
    for tweet in  tweepy.Cursor(api.search,
                           q=query,
                           rpp=1000,
                           result_type="recent",
                           tweet_mode = "extended",
                           include_entities=True,
                           lang="en").items(10):
        
       a = Tweet()
       a.Tweet_text,a.Tweet_url,a.Tweet_user= tweet.full_text,tweet.user.profile_background_image_url,tweet.user.name
       tweetad.append(a)
       
    return tweetad

# Create your views here.
def Rendersearch(request):
    return render(request , 'TweetSearch.html')

def Rendersearchresult(request):
    #return render(request , 'TweetSearch.html')
    #return HttpResponse(Query_tweets(api,request.POST["Query"]))
    tweetcolls = Query_tweets(api,request.POST["Query"])
    return render(request,'TweetSearch2.html',{ "tweets": tweetcolls})







