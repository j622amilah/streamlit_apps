# cd Documents/PROGRAMMING/Github_analysis_PROJECTS/
# streamlit run my_script.py
# strg + w

import numpy as np
import streamlit as st

# from streamlit_option_menu import option_menu
# from streamlit_shap import st_shap
# from streamlit_echarts import st_echarts#set settings for streamlit page

def sortarr(arr):
    sarr = []
    temp = arr
    while(len(temp) > 1):
        mv = min(temp)
        cnt = [1 for r in temp if r == mv]
        vals = [mv for r in range(sum(cnt))]
        sarr.append(vals)
        temp = [q for q in temp if q != mv]
    if any(temp) == True:
        sarr.append(temp)
        
    # une liste avec les valeurs unique
    unq = [sarr[i][0] for i in range(len(sarr))]
    
    # une liste avec les valeurs qui repeter
    nonunq = []
    ind = []
    for i in unq:
        for indd, j in enumerate(arr):
            if i == j:
                nonunq.append(j)
                ind.append(indd)
    
    return ind, unq, nonunq

# with st.sidebar:
#     selected = option_menu(None, ["Log In", "Title here"], 
#     icons=['house',  "list-task"], 
#     menu_icon="cast", default_index=0, orientation="vertical")
    
    
def head():

    st.title('Welcome to this test application!!')
    # OR
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: -35px;'>
        Test Streamlit: Sort an array
        </h1>
    """, unsafe_allow_html=True
    )
    
    st.caption("""
        <p style='text-align: center'>
        by <a href='https://github.com/j622amilah/streamlit_apps'>Github_repo</a>
        </p>
    """, unsafe_allow_html=True
    )
    
    st.write(
        "This is a silly application to learn how to use streamlit.",
        "Working with applications on the Internet seems difficult.",
        "Click the button \U0001F642."
    )
    return


def body(arr):
    out = sortarr(arr)
    st.info('sorted array}', icon='\U0001F916')
    st.write(out)
    



head()

if st.button('Start Body code'):
    # df = read_data('filename.csv')
    # choice = df.sample(1)
    A = [2, 4, 6, 4, 3]
    A = np.array(A)
    st.write('A : ', A)
    
    body(A)





