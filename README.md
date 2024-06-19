# MySQL Database on Movies
 








### Natural Language Processing 

Natural language processing(NLP) was used to classify high and low rated movies.  By analyzing the language in movie reviews, the movies were classified as low or high rated movies.  Using sentiment analysis, word cloud and word frequency distribution, recommendations were made.
![wordclouds-reviews](https://github.com/SeeBee8/MySQL-Database-Movies/assets/141530991/71998349-1aa2-4fc6-a869-77033cba3e0c)
![freqdist-reviews](https://github.com/SeeBee8/MySQL-Database-Movies/assets/141530991/f804c34c-afbd-41b0-8144-cb03b4007b2f)

------------------------- 
#### Recommendations:

In order to have movies with high-rating reviews, the following is recomended:
1.  Use a well known and established director.  People seem to pay a great deal of attention to the director
2.  Have a femme fatale.  These are super popular in movies with good reviews
3.  Make an action movie.  Action movies are popular among the highly rated movies

In order to avoid low ratings, the following is recommended:
1.  Avoid underdeveloped characters or plots.  These are often pointed out as disapointing
2.  People who give low ratings use the word "unimaginative".  Rather than recreating the old, try making somethin new and never done before.
3.  Poor reviews also come from bad music.  If the music doesn't fit the movie or is just plain bad, people dislike the movie.


**Sources:**
https://seaborn.pydata.org/generated/seaborn.countplot.html
https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html
https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html
https://spotintelligence.com/2022/12/10/stop-words-removal/#:
https://amueller.github.io/word_cloud/generated/wordcloud.WordCloud.html
https://www.w3schools.com/python/ref_func_zip.asp
https://www.geeksforgeeks.org/change-font-size-in-matplotlib/
https://www.educative.io/answers/how-to-remove-emoji-from-the-text-in-python

---------------------------------------------------------------------------

## Movies Reviews App
This app was created through Streamlit to show: 

-  How reviews will rate
-  Look at model parameters
-  Check out frequency distribution of words used in review
-  See the Word Clouds for the analysis
-  Look at how training and testing data perfom
![Screenshot 2024-02-22 151332](https://github.com/SeeBee8/MySQL-Database-Movies/assets/141530991/e7525bde-ef8c-4e14-95e3-bdd029603c47)


-------![Screenshot 2024-02-22 151400](https://github.com/SeeBee8/MySQL-Database-Movies/assets/141530991/599cd1f2-4230-4f20-9be5-e4353d1822da)
----------------------------------------------------------------

Information courtesy of
IMDb
(https://www.imdb.com).
Used with permission.
![TMDBshort](https://github.com/SeeBee8/MySQL-Database-Movies/assets/141530991/4db18f17-f6e6-4774-9bba-74f96d8faefc)
<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 190.24 81.52"><defs><style>.cls-1{fill:url(#linear-gradient);}</style><linearGradient id="linear-gradient" y1="40.76" x2="190.24" y2="40.76" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#90cea1"/><stop offset="0.56" stop-color="#3cbec9"/><stop offset="1" stop-color="#00b3e5"/></linearGradient></defs><title>Asset 2</title><g id="Layer_2" data-name="Layer 2"><g id="Layer_1-2" data-name="Layer 1"><path class="cls-1" d="M105.67,36.06h66.9A17.67,17.67,0,0,0,190.24,18.4h0A17.67,17.67,0,0,0,172.57.73h-66.9A17.67,17.67,0,0,0,88,18.4h0A17.67,17.67,0,0,0,105.67,36.06Zm-88,45h76.9A17.67,17.67,0,0,0,112.24,63.4h0A17.67,17.67,0,0,0,94.57,45.73H17.67A17.67,17.67,0,0,0,0,63.4H0A17.67,17.67,0,0,0,17.67,81.06ZM10.41,35.42h7.8V6.92h10.1V0H.31v6.9h10.1Zm28.1,0h7.8V8.25h.1l9,27.15h6l9.3-27.15h.1V35.4h7.8V0H66.76l-8.2,23.1h-.1L50.31,0H38.51ZM152.43,55.67a15.07,15.07,0,0,0-4.52-5.52,18.57,18.57,0,0,0-6.68-3.08,33.54,33.54,0,0,0-8.07-1h-11.7v35.4h12.75a24.58,24.58,0,0,0,7.55-1.15A19.34,19.34,0,0,0,148.11,77a16.27,16.27,0,0,0,4.37-5.5,16.91,16.91,0,0,0,1.63-7.58A18.5,18.5,0,0,0,152.43,55.67ZM145,68.6A8.8,8.8,0,0,1,142.36,72a10.7,10.7,0,0,1-4,1.82,21.57,21.57,0,0,1-5,.55h-4.05v-21h4.6a17,17,0,0,1,4.67.63,11.66,11.66,0,0,1,3.88,1.87A9.14,9.14,0,0,1,145,59a9.87,9.87,0,0,1,1,4.52A11.89,11.89,0,0,1,145,68.6Zm44.63-.13a8,8,0,0,0-1.58-2.62A8.38,8.38,0,0,0,185.63,64a10.31,10.31,0,0,0-3.17-1v-.1a9.22,9.22,0,0,0,4.42-2.82,7.43,7.43,0,0,0,1.68-5,8.42,8.42,0,0,0-1.15-4.65,8.09,8.09,0,0,0-3-2.72,12.56,12.56,0,0,0-4.18-1.3,32.84,32.84,0,0,0-4.62-.33h-13.2v35.4h14.5a22.41,22.41,0,0,0,4.72-.5,13.53,13.53,0,0,0,4.28-1.65,9.42,9.42,0,0,0,3.1-3,8.52,8.52,0,0,0,1.2-4.68A9.39,9.39,0,0,0,189.66,68.47ZM170.21,52.72h5.3a10,10,0,0,1,1.85.18,6.18,6.18,0,0,1,1.7.57,3.39,3.39,0,0,1,1.22,1.13,3.22,3.22,0,0,1,.48,1.82,3.63,3.63,0,0,1-.43,1.8,3.4,3.4,0,0,1-1.12,1.2,4.92,4.92,0,0,1-1.58.65,7.51,7.51,0,0,1-1.77.2h-5.65Zm11.72,20a3.9,3.9,0,0,1-1.22,1.3,4.64,4.64,0,0,1-1.68.7,8.18,8.18,0,0,1-1.82.2h-7v-8h5.9a15.35,15.35,0,0,1,2,.15,8.47,8.47,0,0,1,2.05.55,4,4,0,0,1,1.57,1.18,3.11,3.11,0,0,1,.63,2A3.71,3.71,0,0,1,181.93,72.72Z"/></g></g></svg>
