# Tweet-O-Pedia (The Wikipedia for Twitter )

## Our Objective
Social Media has become an integrable part of our lives.It is a storehouse of public sentiment and perception.However,a lot of this data lies unused.Tweetopedia,The Wikipedia of Twitter includes some basic functionalities which would let you thoroughly analyse the data on twitter.Some of these are:
  <ul>
    <li>Generate Tweets Like President Trump.It is an effective way to Trump's opinions on important Topics.Just Enter the Topic as the starting phrase and watch magic unfold.</li>
    <li>Search for Tweets on Twitter based on your own Search Terms.</li>
    <li>Find average public sentiment on twitter regarding any topic of Interest.This has wide ranging benefits from allowing you to measure public sentiment towards any event,policy to finding how the Internet likes the product you have launched.</li>
    <li>Have a look at some of the trending hashtags worldwide.</li>
    <li>Measure the hate-o-meter to guage hate towards certain groups or topics.Also report to the Authorities any highly offensive tweet that you might find out.</li></ul>
Hope you like it and enjoy it.

#### *In order to optimize the Machine Learning Models that I have used please visit* <a href="https://github.com/avinsit123/Tweetopedia_Collab_Notebooks"> *this notebook* </a> *and follow the instructions given in the readme*

### I have also included a few goodies for fans of data analysis.Please visit this* <a href ="https://github.com/avinsit123/Tweet-o-Pedia/tree/master/mysite"> README </a> *to know more*

## Running the project on your local system

### Installing the requirements
The project is built using Django framework so you must have Python installed on your local system and also preferably Anaconda Distribution.Follow the steps given below to set up the project on your local system

```terminal
$git clone 
$cd Tweet-o-Pedia 
$pip install -r requirements.txt 
```
 It is a long file and might take some time to install all the dependencies on your local pc.
 
 ### Setting up the Server
 
 After the above steps are completed follow the steps given below
 ```terminal
 $mkdir ~/virtualenvironment
 $source activate
 ```
 Your production environment will be replaced by (base)
 
 ```terminal
 $cd mysite
 $python manage.py runserver 
 ```
 
 Your output on the Terminal will be something like this.
 
 ```
 Performing system checks...

System check identified no issues (0 silenced).

You have 15 unapplied migration(s). Your project may not work properly until you apply the migrations for app(s): admin, auth, contenttypes, sessions.
Run 'python manage.py migrate' to apply them.

December 29, 2018 - 13:50:33
Django version 2.1.4, using settings 'mysite.settings'
Starting development server at http://127.0.0.1:8000/
Quit the server with CONTROL-C.
 ```
 
 Cut and paste http://127.0.0.1:8000/ on any browser of your choice or click on the link in the terminal.A WebPage of the following figure will be loaded:
 
 
 ![Optional Text](../master/Photos/screenomatic.png)
 
 <hr>
  
  
  ## To check whether all windows are working
  
  After Clicking on Tweet Generator the following Window would open.
  ![Optional Text](../master/Photos/pc6.png)
  
  After Clicking on Twitter trends the following Window would open
   ![Optional Text](../master/Photos/pc4.png)
   
  After clicking on Sentiment Search,the following would open
   ![Optional Text](../master/Photos/pc5.png)
   
  Enter your query and click Search a window of the following type would occur.
   ![Optional Text](../master/Photos/pic3.png)
   
  After clicking on Hate Finder ,the following would open
   ![Optional Text](../master/Photos/pc7.png)
   
  The percentage of hate would be displayed as :
    ![Optional Text](../master/Photos/pc9.png)
    
  Because I have used a strict criteria of Evaluation the hate percentage is lower . You can modify the notebooks to change the criteria.
  
  
   
  
