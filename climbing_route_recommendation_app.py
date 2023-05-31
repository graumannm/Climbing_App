import pandas as pd
import numpy as np
import streamlit as st

# load data 
routes = pd.read_csv("./climbing_dataset/routes_rated.csv")

# simplify df a bit
routes.drop(['Unnamed: 0','name_id'],axis=1,inplace=True)

# determine numerical grade from French input grade
# take french grade as input, convert to list for streamlit and then to integer for filtering
grades    = pd.read_csv("./climbing_dataset/grades_conversion_table.csv")
list_text = grades.grade_fra.tolist()
fra_grade = st.selectbox('What is your preferred climbing grade', list_text) 

# now retrieve int value from conversion table
my_grade = grades.loc[grades['grade_fra'] == fra_grade].index[0]

# select best route for short climbers in Germany
my_country = st.selectbox('Which country are you from?',
                       ['and', 'arg', 'aus', 'aut', 'bel', 'bgr', 'bih', 'bra', 'can',
                        'che', 'chl', 'chn', 'col', 'cze', 'deu', 'dnk', 'ecu', 'esp',
                        'fin', 'fra', 'gbr', 'grc', 'hrv', 'hun', 'ind', 'isl', 'isr',
                         'ita', 'jor', 'lao', 'lux', 'mar', 'mex', 'mkd', 'mlt', 'msr',
                        'nld', 'nor', 'nzl', 'per', 'phl', 'pol', 'pri', 'prt', 'reu',
                        'rom', 'rus', 'srb', 'svk', 'svn', 'swe', 'tha', 'tur', 'twn',
                        'ukr', 'usa', 'ven', 'vnm', 'zaf']) 
tallness   = st.selectbox('Are you 1.80 m tall or more?',['Yes','No'])
st.write('Please select your preference: \
    0 - Soft routes 1 - Routes for some reason preferred by women \
2 - Famouse routes \
3 - Very hard routes \
4 - Very repeated routes \
5 - Chipped routes, with soft rate \
6 - Traditiona, not chipped routes \
7 - Easy to On-sight routes, not very repeated \
8 - Very famouse routes but not so repeated and not so traditional')
cluster    = st.slider('Please select your preference',min_value=0, max_value=8)

# get crags with lowest tall_recommended sum in Germany
if tallness=='Yes':
    sorted_routes = routes.sort_values(by=['tall_recommend_sum'], ascending=False)
elif tallness=='No':
    sorted_routes = routes.sort_values(by=['tall_recommend_sum'])

# show all German routes in descending tall difficult order
country_idx = sorted_routes['country']==my_country
sorted_routes = sorted_routes[country_idx] 

# remove country since we don't need it anymore
sorted_routes.drop('country',axis=1,inplace=True)

# filter by cluster and remove cluster
sorted_routes = sorted_routes[sorted_routes['cluster']== cluster]
sorted_routes.drop('cluster',axis=1,inplace=True)

# filter all routes +/- 5 above indicated grade
sorted_routes = sorted_routes[
    (my_grade-5 < sorted_routes['grade_mean'])  & (sorted_routes['grade_mean'] < my_grade+5)] 


# display best rated for our selection
sorted_routes = sorted_routes.sort_values(by=['rating_tot'],ascending=False)

# rename columns
sorted_routes = sorted_routes.rename(columns={"crag": "Crag", "sector": "Sector", "name": "Name",
                                            "tall_recommend_sum": "Preferred by Tall",
                                            'grade_mean': "Mean Grade",
                                             "rating_tot": "Average Rating"})

# convert final grading to french grading

# round grade mean to int
sorted_routes = sorted_routes.round({'grade_mean': 0})
grades = pd.read_csv("./climbing_dataset/grades_conversion_table.csv")

my_grade = list(np.zeros(len(sorted_routes['grade_mean'])))
for i, r in enumerate(sorted_routes['grade_mean']):
    my_grade[i] = str(grades.loc[int(r), 'grade_fra'])

sorted_routes['french_grades'] = my_grade
sorted_routes.drop('grade_mean',axis=1,inplace=True)

if st.button('Show recommended routes'):
    st.table(sorted_routes)