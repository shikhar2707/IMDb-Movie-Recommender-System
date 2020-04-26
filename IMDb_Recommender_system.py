#!/usr/bin/env python
# coding: utf-8

# # Recommender System
# ## Dataset link - [Click here](www.kaggle.com)

# In[1]:


import pandas as pd
import numpy as np


# ## Reading the datafile

# In[2]:


data = pd.read_csv("IMDB Movie Recommender System/imdb.csv" , error_bad_lines=False)


# In[3]:


data


# ## Dropping the columns which are not that useful for system/model.

# In[4]:


data = data.drop(["fn","tid","url"], axis = 1)


# In[5]:


data


# In[79]:


#The max values of each column
data.max()


# In[7]:


#The minimum values of each column. Ignore the title one though.
data.min()


# In[8]:


#Highest rated movie on imdb
data.query('imdbRating==9.9')['title']


# In[9]:


#Longest Movie imdb recognizes
data.query('duration==68400')['title']


# ## Removing all the null value containing movies since it would create nuisance in the model.

# In[10]:


data=data.dropna(subset=['imdbRating', 'title','type','ratingCount'])


# In[11]:


data


# ## Year in float data type? Not a thing. And so does Ratings count.

# In[12]:


data.dtypes


# In[13]:


data["year"] = data["year"].astype("int64")


# In[14]:


data["ratingCount"] = data["ratingCount"].astype("int64")


# In[15]:


data.dtypes


# In[16]:


data


# In[17]:


data.describe()


# In[18]:


#Type of content imdb provides
data['type'].value_counts()


# ## Nobody wants a suggestion on a game. Do they? Especially on IMDb.

# In[19]:


data = data[data.type != "game"]


# In[20]:


data


# ### Alright. Lets take a much needed break from Data Cleaning and Manipulatiom. We'll get back to it soon.
# ### Lets plot a few Bargraphs to see how the data is actually spread.

# In[23]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


#Lets plot the Year Wise Movie Count
plt.figure(figsize=(36,27))
data.groupby('year')['year'].count().plot(kind='bar')
plt.title('Year-Wise Movies',fontsize=20)
plt.yticks(fontsize=24)
plt.xticks(fontsize=12)
plt.xlabel('Year',fontsize=12)
plt.ylabel('Movie Count',fontsize=12)
plt.show()
print(data.groupby('year')['year'].count())


# ### So IMDb has more number of reviews available for movies for last decade(2011-20) than any other time period. Hmm!!

# In[25]:


#Lets check the Movie count w.r.t Imdb Ratings
plt.figure(figsize=(36,27))
data.groupby('imdbRating')['imdbRating'].count().plot(kind='bar')
plt.title('Movie Variance',fontsize=20)
plt.yticks(fontsize=24)
plt.xticks(fontsize=12)
plt.xlabel('IMDb Ratings',fontsize=12)
plt.ylabel('Movie Count',fontsize=12)
plt.show()
print(data.groupby('imdbRating')['imdbRating'].count())


# ### Most of the ratings around that 7 region which kind of justifies the fact that 7 is actually an average number as far as rating is concerned.

# # Data Preparation for Recommender System Pipeline

# In[26]:


# Dropping the columns not using in my system.
data = data.drop(["nrOfGenre" , "nrOfPhotos", "nrOfNewsArticles", "nrOfUserReviews"], axis = 1)


# In[27]:


#Now lets encode the type of the content.
data.type[data.type == 'video.movie'] = 2
data.type[data.type == 'video.tv'] = 1
data.type[data.type == 'video.episode'] = 0


# In[28]:


data = data.drop(["duration"], axis =1)


# ## Creating the Acclamation Parameter

# In[29]:


#Now to quantify the acclaimation index. Lets make a acclaimation index which is basically 0.5*NrNominations + NrWins
data["Acclaimation index"] = data["nrOfWins"] + 0.5*data["nrOfNominations"]


# In[30]:


data["Acclaimation index"].values


# In[31]:


column_list = data.columns.tolist()


# In[32]:


column_list


# In[33]:


# Now dropping the wins and nominations column
data = data.drop(["nrOfWins","nrOfNominations"], axis = 1)


# In[34]:


#Now normalizing the ratings count for ease.
min_ratings = data["ratingCount"].min()
max_ratings = data["ratingCount"].max()
data["ratingCount"] = (data["ratingCount"] - min_ratings)/(max_ratings - min_ratings) 


# In[35]:


data = data.drop(["wordsInTitle"] , axis = 1)


# In[36]:


data = data.drop(["year"] , axis = 1)


# ## Now lets create a list index-wise which will contain the genres of the movies in an array form.

# In[53]:


#Looking the columns we require for genre identification.
list(data.columns)


# In[54]:


#We need columns from index 4 to index 31.
len(list(data.columns))


# In[56]:


data


# In[55]:


data.iloc[1,3]


# In[81]:


Genre_Dict = []
for i in range(13040):
    genre_list = []
    for j in range(4,31):
        genre_list.append(data.iloc[1,j])
    Genre_Dict.append(np.array(genre_list))
Genre_Dict
    
        


# ### Now the list named "Genre_Dict" will contain the arrays corresponding to the movie index.
# ### This is one of our first metric we will consider alongwith Acclamation Index of a movie.

# ### So as far as calculations goes.
# ### We have following quantities for distance measurements.
# ### 1. Acclaimation Distance
# ### 2. Genre Array distance.
# ### 3. Ratings Count Distance.
# ### 4. IMDb Rating
# ### 5. Type of content.

# In[83]:


#Creating a dictionary of movie information vectors.
#A movie information vector will contain movie name,Acclaimation value, Genre array,type, Normalized ratings count and finally the IMDb rating.
movie_info_dict = {}
for i in range(13040):   
     movie_info_dict[i] = (data.iloc[i,0],data.iloc[i,-1],Genre_Dict[i],data.iloc[i,3],data.iloc[i,2],data.iloc[i,1])
    


# In[84]:


#Checking our Movie Info Vector for 1st movie.
movie_info_dict[0]


# ## Setting up the distance metric

# In[92]:


from scipy import spatial
def ComputeDistance(a,b):
    AccA = a[1]
    AccB = b[1]
    Acclaimation_Distance = abs(AccA - AccB)
    TypeA = a[3]
    TypeB = b[3]
    Type_Distance = abs(TypeA - TypeB)
    PopularityA = a[4]
    PopularityB = b[4]
    Popularity_Distance = abs(PopularityA - PopularityB)
    GenreA = a[2]
    GenreB = b[2]
    Genre_Distance = spatial.distance.cosine(GenreA, GenreB)
    return Acclaimation_Distance + Type_Distance + Popularity_Distance + Genre_Distance


# In[93]:


#For checking our prepared distance computing function.
ComputeDistance(movie_info_dict[1],movie_info_dict[3])


# ### Since our distance computing function is prepared. We must know the lesser the distance, the more resemblance between the content. Hence it should be recommended.

# In[102]:


#This is our distance calculation and the functon that will lend us our nearest neighbours.

import operator
def getNeighbors(i,K):
    distances = []
    for movie in movie_info_dict:
        if (movie != i):
            dist = ComputeDistance(movie_info_dict[i], movie_info_dict[movie])
            distances.append((movie, dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])
    return neighbors


# ## Our Recommendation model is finally ready to predict or recommend.

# # Recommedations

# In[103]:


# Lets check the recommendations for movie with index no. 2707
movie_info_dict[2707]


# ### Recommendations for the movie named "Wayne's World 2(1993)"

# In[107]:


#Lets predict five recommendations for the mentioned movie.
K = 5
Average_Rating = 0
neighbors = getNeighbors(2707, K) # Toy Story (1995)
for neighbor in neighbors:
    Average_Rating += movie_info_dict[neighbor][5]
    print (movie_info_dict[neighbor][0] + " " + str(movie_info_dict[neighbor][5]))
Average_Rating /= K


# #### Lets see whats the average rating of the movies recommended.

# In[108]:


Average_Rating


# #### Not bad considering our movie is rated 6.1

# In[ ]:




