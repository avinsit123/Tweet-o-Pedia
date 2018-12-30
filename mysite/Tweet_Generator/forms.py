from django import forms

# Register your models here.
class TweetForm(forms.Form):
    Starting_phrase = forms.CharField()
    Total_Characters = forms.CharField( widget=forms.TextInput(attrs={'type':'number'}))

