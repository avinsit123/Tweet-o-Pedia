
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
import os
import numpy as np
import pandas as pd
from torch.autograd import Variable
from nltk.tokenize import TweetTokenizer
device = "cpu"
device = "cpu"
from wordcloud import WordCloud, STOPWORDS
import cv2


#############################DL PART#############################################################################################################################################################
######################################################################################################################################################################################################


import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class LSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier
    """

    def __init__(self, hidden_dim, batch_size, embedding_size, num_layers=1, dropout=0, verbose=False):
        super(LSTMClassifier, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.verbose = verbose
        self.inner_hidden_dim = 200
        
        self.num_directions = 2 # Bidirectional

        self.lstm = nn.LSTM(embedding_size, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim*2, self.inner_hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(self.inner_hidden_dim, 1)
        self.sig = nn.Sigmoid()

    def init_hidden(self):
        # Tuple of two tensors: (h_0, c_0)
        return (torch.zeros(self.num_layers*self.num_directions, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers*self.num_directions, self.batch_size, self.hidden_dim))

    def forward(self, input, unpadded_lengths):

        #if self.verbose:
           # print('Input:',input.size())
        
        packed = pack_padded_sequence(input, unpadded_lengths, batch_first=True)
        #if self.verbose:
           # print("After packing:",packed.data.size())
        
        packed_output, (hidden, cell) = self.lstm(packed, self.hidden)
        
        #if self.verbose:
           # print('Packed output after LSTM:',packed_output.data.size())
        
        # Get last hidden state for each sequence
        # For each batch, concat the last hidden state from the forward LSTM
        # with the last hidden state from the reverse LSTM.
        # (aka get the hidden state from the last _timestep_)
        
        def unzip(t):
            """
            Returns a tensor where dimension 0 is the layer. So with 3 layers, shape[0] = 3.
            Along each layer, the forward and backward hidden states are concatenated.
            """
            return torch.cat([t[0:t.size(0):2], t[1:t.size(0):2]], 2)
        
        # Get the forward and backward hidden states for the top layer
        out = unzip(hidden)[-1]
        
        out = self.dropout1(out)
        
        #if self.verbose:
            #print('Top hidden layer from LSTM:', out.size())
            
        out = self.relu1(self.fc1(out))
        out = self.dropout1(out)
        out = self.sig(self.fc2(out))
        out = out.squeeze(1)
        
        #if self.verbose:
            #print('Output:', out.size())
            
        return out

def predict(sentence,embeddings_dict,model, max_seq_length):
  UNK = '<unknown>'
  PAD = 'PAD_TOKEN'
  
  tk = TweetTokenizer()
  sentence_lengths = torch.LongTensor([min(len(sentence),max_seq_length)])
  sent_embedded=[]
  sentence = sentence[:sentence_lengths]
  gloigunda = 0
  for i,word in enumerate(tk.tokenize(sentence)):
    if word in embeddings_dict:
      word_embedding = embeddings_dict[word]
    else:
      word_embedding = embeddings_dict[UNK]
    sent_embedded.append(torch.FloatTensor(word_embedding))
    gloigunda  = i
  for padding in range(max_seq_length- gloigunda):
    sent_embedded.append(torch.FloatTensor(embeddings_dict[PAD]))
  batch_source_embedded = Variable(torch.stack(sent_embedded))
  batch_source_embedded = batch_source_embedded.view(1,batch_source_embedded.shape[0],batch_source_embedded.shape[1])
  max_seq_length = torch.FloatTensor([max_seq_length])
  origi = batch_source_embedded
  orilen =  max_seq_length
  for i in range(49):
    batch_source_embedded = torch.cat((origi,batch_source_embedded),0)
    max_seq_length = torch.cat((orilen,max_seq_length),0) 
  #print( max_seq_length.size())
  #print(batch_source_embedded.size(),"sdg")
  return model(batch_source_embedded,max_seq_length)
  
  

  
  

def goloader(tweet):
   BATCH_SIZE = 50
   EMBEDDING_DIM = 200
   model = LSTMClassifier(hidden_dim=100, batch_size=BATCH_SIZE, embedding_size=EMBEDDING_DIM,
                       num_layers=4, dropout=0, verbose=True)
   
   with open('/Users/r17935avinash/Downloads/hateme.pth', 'rb') as f:
       checkpoint = torch.load(f,map_location='cpu')

   model.load_state_dict(checkpoint['state_dict'])
   MAX_LENGTH = checkpoint['MAX_LENGTH']
   embeddings_dict = checkpoint['embeddings_dict']
   model.hidden = model.init_hidden()
   return np.average(predict(tweet,embeddings_dict,model,36).detach().numpy(),axis=0)


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
                                lang="en").items(10):
        #print(tweet.full_text)
        tweet_text.append("!!!!" + tweet.full_text)
    print(len(tweet_text))
    cloud = " "
    tot_sentiment = 0
    for tweet in tweet_text:
        tweet = remove_punctuation(tweet)
        tweet = removes_url(tweet)
        tweet = remove_hashtag(tweet)
        tweet = remove_hash_symbol(tweet)
        tot_sentiment += goloader(tweet)
        range_list[retrange(goloader(tweet) * 100)].count = range_list[retrange(goloader(tweet) * 100)].count + 1
        cloud = cloud + " " + tweet
        print(tweet +"\n")
        #print(retrange(goloader(tweet) * 100))
        #p.clean(tweet)
        #print(tweet)
    for i in range(len(range_list)):
        range_list[i].count = round(range_list[i].count * 2,2)
        
    stopwords = set(STOPWORDS) 
    wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(cloud)
    print(type(wordcloud))
    wordcloud.to_file("hate_o_meter/static/wordcloud.jpg")
    hyperlink = "wordcloud.jpg"
    return round(tot_sentiment * 2,2),range_list,hyperlink

# Create your views here.
def Rendersearch12(request):
    return render(request , 'HateSearch01.html')

def Rendersearchresult12(request):
    avg_sentiment , range_table , hyperlink= Query_tweets(api,request.POST["Query"])
    return render(request,'HateSearch02.html',{ "Query": request.POST["Query"] , "avg_sentiment":avg_sentiment , "table" : range_table ,"hyperlink" : hyperlink})

    #return  HttpResponse(Query_tweets(api,request.POST["Query"]))






