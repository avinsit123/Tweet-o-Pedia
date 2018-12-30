from django.shortcuts import render
from django.http import HttpResponse
from .forms import TweetForm
from .LSTMModel  import sample,LoadModel

# Create your views here.

def Tweet_form(request):

    if request.method=='POST':
        print("Asd")
    form = TweetForm()
    return render(request,'Tweetform.html',{'form' : form})
    

def ProcessForm(request):
    if request.method == 'POST':
        text = request.POST["start_phrase"]
        no_of_words = int(request.POST["now"])
        model = LoadModel()
        texttweet = sample(model, no_of_words, text, top_k=20)
        return render(request,'tweetform2.html',{'newman' : texttweet})
    return  HttpResponse("Hello")
