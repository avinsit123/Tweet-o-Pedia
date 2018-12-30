from django.shortcuts import render

# Create your views here.


def objopage(request):
    return render(request,"objective.html")