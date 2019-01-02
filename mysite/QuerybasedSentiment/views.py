
from django.shortcuts import render
import tweepy
from tweepy.auth import OAuthHandler
from django.http import HttpResponse
import json
import torch
from torchtext import data
from torchtext import datasets
import random
import torch.nn as nn
import torch.nn.functional as F
device = "cpu"
from wordcloud import WordCloud, STOPWORDS 
import cv2 
import matplotlib.pyplot as plt

#############################DL PART#############################################################################################################################################################
######################################################################################################################################################################################################

import spacy
nlp = spacy.load('en')

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs,embedding_dim)) for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes)*n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x.permute(1, 0)
        embedded = self.embedding(x)
        embedded = embedded.unsqueeze(1)
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)



def predict_sentiment(sentence, hoho,model,min_len=5 ):
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    #print(tokenized)
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [hoho[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(model(tensor))
    return prediction.item()

def goloader(tweet):
   INPUT_DIM = 25002
   EMBEDDING_DIM = 100
   N_FILTERS = 100
   FILTER_SIZES = [3,4,5]
   OUTPUT_DIM = 1
   DROPOUT = 0.5 

   model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT)
   with open('/Users/r17935avinash/Downloads/hateorlove.pth', 'rb') as f:
       checkpoint = torch.load(f,map_location='cpu')

   model.load_state_dict(checkpoint['state_dict'])
   dictionary = checkpoint['stoi']
   return predict_sentiment(tweet,dictionary,model)


#############################TWITTER PART#############################################################################################################################################################
######################################################################################################################################################################################################

import preprocessor as p
import re 
import string

#Class to print range of sentiment
class tweettype:
    def __init__(self,count,torange):
        self.count = 0
        self.range = torange
        
#creating a list of ranges
def createlist():
    range_list = [tweettype(0,"haha") for _ in range(5)]
    range_list[0].range = "0 - 20 "
    range_list[1].range = "20 - 40 "
    range_list[2].range = "40 - 60 "
    range_list[3].range = "60 - 80 "
    range_list[4].range = "80 - 100 "
    return range_list
    
    
#return range of tweet
def retrange(sentiment):
    if sentiment >= 0 and sentiment <=20 :
        return 0
    elif sentiment > 20 and sentiment <=40 :
        return 1
    elif sentiment > 40 and sentiment <=60 :
        return 2
    elif sentiment > 60 and sentiment <=80 :
        return 3
    else:
        return 4
    return -1


#remove all punctuation
def remove_punctuation(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, "")
    return text.strip(' ')

#removes URLs
def removes_url(text):
    text = re.sub(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))', '', text)
    return text.strip(' ')

#removes # and @ in the beginning of each word. ex: #Good -> Good
def remove_hashtag(text):
    new_text = ""
    for words in text.split():
        if words.startswith('#') or words.startswith('@'): #remove @ amd #
            new_text += words[1:]
            new_text += ' '
        else:
            new_text += words
            new_text += ' '
    return new_text.strip(' ')

#removes # and @ even between words. ex: #life#is#good -> life is good
def remove_hash_symbol(text):
    to_be_removed = ['#', '@']
    for prohibited_symbol in to_be_removed:
        text = text.replace(prohibited_symbol, ' ')
    text = ' '.join(text.split())
    return text.strip(' ')


consumer_key="liABpG2XM62tIo02fjen7FDVJ"
consumer_secret="nUI6fH6PJuuUvQFlzylCieFxOZ8VZdf5yGrRGK294whWEenayA"
access_token="1046393582469500928-a4Il71HUPkyF0MtoS3V6dJL1igx84k"
access_secret="VcGhV2AcXHQISVXwYuOhmIVkvxaMuPnWlhnGS4LmwEkDN"

       
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

api = tweepy.API(auth)

def process_or_store(tweet):
    print(json.dumps(tweet))
    
def Query_tweets(api,query):
    
    range_list = createlist()
    tweet_text = []
    for tweet in  tweepy.Cursor(api.search,
                                q=query,
                                tweet_mode = 'extended',
                                rpp=1000,
                                result_type="recent",
                                include_entities=True,
                                lang="en").items(50):
        #print(tweet.full_text)
        tweet_text.append(tweet.full_text)
    print(len(tweet_text))
    tot_sentiment = 0
    cloud = " "
    for tweet in tweet_text:
        tweet = remove_punctuation(tweet)
        tweet = removes_url(tweet)
        tweet = remove_hashtag(tweet)
        tweet = remove_hash_symbol(tweet)
        tot_sentiment += goloader(tweet)
        cloud = cloud + " " + tweet
        range_list[retrange(goloader(tweet) * 100)].count = range_list[retrange(goloader(tweet) * 100)].count + 1
        
        #print(retrange(goloader(tweet) * 100))
        #p.clean(tweet)
        print(tweet)
    for i in range(len(range_list)):
        range_list[i].count = round(range_list[i].count * 2,2)
        
    stopwords = set(STOPWORDS) 
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(cloud)
    print(type(wordcloud))

    wordcloud.to_file("QuerybasedSentiment/static/wordcloud.jpg")
    hyperlink = query+".jpg"

    return round(tot_sentiment * 2,2),range_list,hyperlink

# Create your views here.
def Rendersearch1(request):
    return render(request , 'TweetSearch01.html')

def Rendersearchresult1(request):
    #return render(request , 'TweetSearch.html')
    #return HttpResponse(Query_tweets(api,request.POST["Query"]))
    #tweetcolls = Query_tweets(api,request.POST["Query"])
    avg_sentiment , range_table , hyperlink = Query_tweets(api,request.POST["Query"])
    return render(request,'TweetSearch02.html',{ "Query": request.POST["Query"] , "avg_sentiment":avg_sentiment , "table" : range_table ,"hyperlink" : hyperlink})

    #return  HttpResponse(Query_tweets(api,request.POST["Query"]))

if __name__ == "__main__":
    Query_tweets(api,"Mueller Investigation")
    




