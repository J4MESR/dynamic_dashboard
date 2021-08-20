# -*- coding: utf-8 -*-
# ## IMPORTING LIBRARIES ###

# standard Python libraries
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import time

# data import libraries
import os
home = os.path.expanduser('~')
import psycopg2
from sqlalchemy import create_engine
from dotenv import load_dotenv
load_dotenv()

# visualisation libraries
import streamlit as st
# import matplotlib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.stats import zscore
# sns.set_theme(style = 'ticks', context = 'talk')

# lp libraries
from lp_program.lp_opt import *
import pulp as p

import pandas as pd
import numpy as np
import pulp as p

# ## DEFINING CACHED FUNCTIONS ###

@st.cache
def load_file(filename):
    """
    Cached load of CSV file matching filename into DataFrame
    """
    if '.xlsx' in filename:
        return pd.read_excel(filename)
    else:
        file = pd.read_csv(filename)
        file = file.set_index('RM').T
        return file


### INITIAL DASHBOARD LAYOUT AND ADDING SESSION STATE ###

st.title('Sika R&D Stixall Dynamic Formulation Dashboard')

if 'button variable' not in st.session_state:
    st.session_state['button variable'] = 0
if 'equals_state' not in st.session_state:
    st.session_state['equals state'] = 0

    
### UPLOADING DATA ###

st.header('')
with st.beta_expander('Upload raw material cost data'):
    filename = st.text_input('Enter a file path:', home) 
    if filename != "":
        try: 
            data_initial = load_file(filename)
            st.write(data_initial)
        except:
            st.text('dataset not found')
            st.stop()

    data = data_initial.copy()
    non_nan_cols = [col for col in data if data[col].isnull().sum() == 0] # list of columns removed nans
    nan_cols = [col for col in data if data[col].isnull().sum() != 0]
    #data = data[non_nan_cols]
    avaliable_rms = True

    

### SELECT VARIABLES ###

with st.beta_expander('Unavailable Raw Materials'):

    unavailable_rms_options = ['VSHI4BTF', 'VPOLSHI17C', 'VSTPE10', 'VSPUR1015LM', 'VAMMO', 'VDAMO', 'XDINP', 'VDINCH']
    unavailable_rms = st.multiselect("Select unavailable raw materials:", options = unavailable_rms_options, 
                                     key = 'Unavailable raw materials', default = None)

if 'VSHI4BTF' in unavailable_rms and 'VPOLSHI17C' in unavailable_rms and 'VSTPE10' in unavailable_rms and 'VSPUR1015LM' in unavailable_rms:
    st.warning('At least one polymer must be available for optimisation to proceed.')
    st.stop()

if 'VAMMO' in unavailable_rms and 'VDAMO' in unavailable_rms:
    st.warning('At least one aminosilane must be available for optimisation to proceed.')
    st.stop()

if 'XDINP' in unavailable_rms and 'VDINCH' in unavailable_rms:
    st.warning('At least one plasticiser must be available for optimisation to proceed.')
    st.stop()    
    
if 'VPOLSHI17C' in unavailable_rms and 'VSTPE10' in unavailable_rms and 'VSPUR1015LM' in unavailable_rms:
    st.warning('VSHI4BTF has only been experimentally proven at 66% of the total polymer content, please remove at least one polymer.')
    st.stop()

### CONSTRAINTS DICTIONARY ###

data = data.astype(float)

constraints_list1 = [{356.7: ['=', ['VSHI4BTF', 'VPOLSHI17C', 'VSTPE10', 'VSPUR1015LM']]}, {237.8: ['<=', ['VSHI4BTF']]}]
constraints_list2 = [{356.7: ['=', ['VSHI4BTF', 'VPOLSHI17C', 'VSTPE10', 'VSPUR1015LM']]}, {237.8: ['<=', ['VSHI4BTF']]}]
constraints_list3 = [{356.7: ['=', ['VSHI4BTF', 'VPOLSHI17C', 'VSTPE10', 'VSPUR1015LM']]}, {237.8: ['<=', ['VSHI4BTF']]}]
constraints_list4 = [{356.7: ['=', ['VSHI4BTF', 'VPOLSHI17C', 'VSTPE10', 'VSPUR1015LM']]}, {237.8: ['<=', ['VSHI4BTF']]}]  

constraints_lists = [constraints_list1, constraints_list2, constraints_list3, constraints_list4]
if 'VSHI4BTF' in unavailable_rms:
    constraints_lists = [[i[0]] for i in constraints_lists]

for n, i in enumerate(constraints_lists):
    dicty = i[0]
    keyy = list(dicty.keys())[0]
    signy, constrained_elements = dicty[keyy]
    constrained_elements = [i for i in constrained_elements if i not in unavailable_rms]
    constraints_lists[n][0][keyy] = [signy, constrained_elements]

percent_dict = {'VCRAYVALLACSLT': 19.5, 'VOMNYA5ML': 260, 'VHAKUENKACCRS': 780, 'VXL10': 65.0, 'VSHI4BTF':237.8, 'VPOLSHI17C':356.7, 
                 'VSTPE10': 356.7, 'VSPUR1015LM': 356.7, 'XTITAN2': 13.0, 'VCAT850': 3.3, 'VDIDP': 29.3, 'VAMMO': 17.9, 'VDAMO': 14.9, 
                 'XDINP': 241.8, 'VDINCH': 241.8}

new_dict = {}
for key, value in percent_dict.items():
    if key is not unavailable_rms:
        new_dict[key] = value


with st.beta_expander('Raw material conditions'):
    
    standard_options = ['VSHI4BTF', 'VPOLSHI17C', 'VSTPE10', 'VSPUR1015LM']
    rm_to_constrain_options = [rm for rm in standard_options if rm not in unavailable_rms]
    
    rm_to_constrain = st.multiselect("Select raw materials to constrain:", options = rm_to_constrain_options, 
                                     key = 'Constrained raw materials', default = None)
    
    
    for rm in rm_to_constrain:

        val = st.slider(label = 'Percentage ' + rm + ' of the total polymer, %', min_value = 0, max_value = 100, step = 1, key = 'constraining ' + rm + ' value')

        label_list = [rm +' = ', rm + ' ≥ ', rm + ' ≤ ']
        labelz = [i + str(val) + ' %' for i in label_list]
        
        sign = st.radio(label = '', options =labelz, index = st.session_state['equals state'], key = 'constraining ' + rm)
        st.session_state['equals state'] = labelz.index(sign)
        sign = sign.split(' ')[1]

        for lizt in constraints_lists:
            lizt.append({val*0.01*new_dict[rm]:(sign, [rm])})
            

button = st.button(label = 'Generate optimised solution')
if button:
    st.session_state['button variable'] = True
if st.session_state['button variable']:

### standard formulation cost calc ###
    standard_init = pd.read_csv('standard stixall formulation.csv')
    standard_init = standard_init.rename(columns = {'Unnamed: 0':'Formulation'})
    standard_init = standard_init.set_index('Formulation').fillna(0)
    standard_init ['Batch Size'] = standard_init.sum(axis = 1)
    standard_df = optimiser(standard_init, data).formulation_opt()
    standard_df = standard_df.drop(['Cost difference compared to standard per kg (£)', 'Total Cost (£)'], axis = 1)
    standard_df = standard_df.fillna(0)
    standard_cost = standard_df['Cost per kg (£)'].loc['Standard']

### Point Values ###        
    stp_data = pd.read_csv('Point formulations.csv')
    stp_data = stp_data.rename(columns = {'Unnamed: 0':'Formulation'})
    stp_data = stp_data.set_index('Formulation').fillna(0)
    stp_data ['Batch Size'] = stp_data.sum(axis = 1) 
    
    for rm in unavailable_rms:
        if rm in stp_data.columns:
            stp_data = stp_data[stp_data[rm] == 0] #removes rows with unavailable_rm values in them.
    
    for rm in unavailable_rms:
        stp_data = stp_data.drop(rm, axis = 1) #drops columns with rms in unavailable_rm from slider
        
    if 'VSTPE10' in rm_to_constrain or 'VSTPE10' in unavailable_rms:
        try:
            point_formulations = optimiser(stp_data, data).formulation_opt()
        except:
            point_formulations = stp_data
    else:
        point_formulations = optimiser(stp_data, data).formulation_opt() #REMEMBER function drops any rms that do not have a cost associated with them, i.e. if STPE10 does not have a rm cost it will not be included after the function is run.
        point_formulations = point_formulations.drop(['Total Cost (£)'], axis = 1)
        point_formulations = point_formulations.fillna(0)


    ### DECISION VARIABLES ###
    decision_variables_full = ['VSHI4BTF', 'VPOLSHI17C', 'VSTPE10', 'VSPUR1015LM']
    decision_variables = [rm for rm in decision_variables_full if rm not in unavailable_rms]

    ### LP1 Polymer for constant VAMMO & VDINP ###
    constants1 = {'VCRAYVALLACSLT': 19.5, 'VOMNYA5ML': 260, 'VHAKUENKACCRS': 780, 'VXL10': 65.0, 
                 'XTITAN2': 13.0, 'VCAT850': 3.3, 'VDIDP': 29.3, 'VAMMO': 17.9, 'XDINP': 241.8}
    ### LP2 Polymer for constant VAMMO & VDINCH ###        
    constants2 = {'VCRAYVALLACSLT': 19.5, 'VOMNYA5ML': 260, 'VHAKUENKACCRS': 780, 'VXL10': 65.0, 
                 'XTITAN2': 13.0, 'VCAT850': 3.3, 'VDIDP': 29.3, 'VAMMO': 17.9, 'VDINCH': 241.8}
    ### LP3 Polymer for constant VDAMO & VDINP ###        
    constants3 = {'VCRAYVALLACSLT': 19.5, 'VOMNYA5ML': 260, 'VHAKUENKACCRS': 780, 'VXL10': 65.0,
                  'XTITAN2': 13.0, 'VCAT850': 3.3, 'VDIDP': 29.3, 'VDAMO': 14.9, 'XDINP': 241.8}
    ### LP4 Polymer for constant VDAMO & VDINCH ###
    constants4 = {'VCRAYVALLACSLT': 19.5, 'VOMNYA5ML': 260, 'VHAKUENKACCRS': 780, 'VXL10': 65.0, 
                 'XTITAN2': 13.0, 'VCAT850': 3.3, 'VDIDP': 29.3, 'VDAMO': 14.9, 'VDINCH': 241.8}    
    
    constants_lists = [constants1, constants2, constants3, constants4]
    frame_list = []
    
    for n, c in enumerate(constants_lists):
        if unavailable_rms:
            if any(rm in c for rm in unavailable_rms) == False:
                lpframe = stp_lp_optimiser('LP' + str(n+1), c, decision_variables, data.fillna(1000000000), constraints_lists[n])
                frame_list.append(lpframe)
                
        else:
            lpframe = stp_lp_optimiser('LP' + str(n+1), c, decision_variables, data.fillna(1000000000), constraints_lists[n])
            frame_list.append(lpframe)

    final_df = pd.concat(frame_list, axis=0)
    final_df = pd.concat([final_df, point_formulations])
    cost_df = final_df[['Batch Size (kg)', 'Cost per kg (£)']]
    final_df = final_df.drop(['Batch Size (kg)', 'Cost per kg (£)'], axis = 1)
    final_df = pd.concat([final_df, cost_df], axis = 1) #adds batch size and cost per kg to end of df 
    final_df = final_df.fillna(0)


    final_df['Cost difference compared to standard per kg (£)'] = final_df['Cost per kg (£)'] - standard_cost
    final_df['Cost difference compared to standard per kg (£)'] = final_df['Cost difference compared to standard per kg (£)'].round(3)
    final_df = final_df.sort_values('Cost per kg (£)')
    final_df = final_df.drop(['Batch Size (kg)'], axis = 1)

    for rm in nan_cols:
        if rm in final_df.columns:
            final_df = final_df[final_df[rm] == 0]
    
    for i in constraints_lists:
        constraints_to_consider = i[1:]
        for constraints_dict in constraints_to_consider:
            val = list(constraints_dict.keys())[0]
            sign, rm = constraints_dict[val]
            rm = rm[0]
            
            if rm in final_df.columns:
                if sign == '=':
                    final_df = final_df[np.isclose(final_df[rm], val)]
                elif sign == '≥':
                    final_df = final_df[final_df[rm] >= val]
                elif sign == '≤':
                    final_df = final_df[final_df[rm] <= val]    
    try:
        final_df = final_df.drop('Total Cost (£)', axis = 1)
    except:
        final_df
        
    if final_df.shape[0] != 0:
        st.write(pd.DataFrame(final_df.iloc[0]).T)

    all_solutions = st.checkbox("Show all solutions", key = 'checkbox for showing all available solutions')
    if all_solutions:
        final_df
