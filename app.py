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

# App Libraries #
import sys
from streamlit import cli as stcli

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

def main():

    # ## INITIAL DASHBOARD LAYOUT AND ADDING SESSION STATE ###

    st.title('Sika R&D Stixall Dynamic Formulation Dashboard')

    state_variables = ['button variable']
    for variable in state_variables:
        if variable not in st.session_state:
            st.session_state[variable] = False

    # state_model_variables = ['model', 'model_search']
    # for variable in state_model_variables:
    #     if variable not in st.session_state:
    #         st.session_state[variable] = None

    # ## UPLOADING DATA ###

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



    # ## SELECT VARIABLES ###

    with st.beta_expander('Unavailable Raw Materials'):

    #     rm_select_container = st.beta_container() #for select all button and multiselect
    #     constants = ['VCRAYVALLACSLT', 'VOMNYA5ML', 'VHAKUENKACCRS', 'VXL10', 'XTITAN2', 'VCAT850', 'VDIDP']
    #     unavailable_rms_options = [rm for rm in non_nan_cols if rm not in constants]
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

    # ## CONSTRAINTS DICTIONARY ###

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


    # if 'VSHI4BTF' and 'VPOLSHI17C' and 'VSTPE10' in unavailable_rms:
    #     constraints_list1 = [{356.7: ('=', ['VSPUR1015LM'])}]
    #     constraints_list2 = [{356.7: ('=', ['VSPUR1015LM'])}]
    #     constraints_list3 = [{356.7: ('=', ['VSPUR1015LM'])}]
    #     constraints_list4 = [{356.7: ('=', ['VSPUR1015LM'])}]        
    #     break
    # if 'VSHI4BTF' and 'VPOLSHI17C' and 'VSPUR1015LM' in unavailable_rms:
    #     constraints_list1 = [{356.7: ('=', ['VSTPE10'])}]
    #     constraints_list2 = [{356.7: ('=', ['VSTPE10'])}]
    #     constraints_list3 = [{356.7: ('=', ['VSTPE10'])}]
    #     constraints_list4 = [{356.7: ('=', ['VSTPE10'])}]        
    #     break
    # if 'VSTPE10' and 'VSPUR1015LM' in unavailable_rms:        
    #     constraints_list1 = [{356.7: ('=', ['VSHI4BTF', 'VPOLSHI17C'])}, {237.8: ('<=', ['VSHI4BTF'])}]
    #     constraints_list2 = [{356.7: ('=', ['VSHI4BTF', 'VPOLSHI17C'])}, {237.8: ('<=', ['VSHI4BTF'])}]
    #     constraints_list3 = [{356.7: ('=', ['VSHI4BTF', 'VPOLSHI17C'])}, {237.8: ('<=', ['VSHI4BTF'])}]
    #     constraints_list4 = [{356.7: ('=', ['VSHI4BTF', 'VPOLSHI17C'])}, {237.8: ('<=', ['VSHI4BTF'])}]        
    #     break
    # if 'VSHI4BTF' and 'VPOLSHI17C' in unavailable_rms:
    #     constraints_list1 = [{356.7: ('=', ['VSTPE10', 'VSPUR1015LM'])}]
    #     constraints_list2 = [{356.7: ('=', ['VSTPE10', 'VSPUR1015LM'])}]
    #     constraints_list3 = [{356.7: ('=', ['VSTPE10', 'VSPUR1015LM'])}]
    #     constraints_list4 = [{356.7: ('=', ['VSTPE10', 'VSPUR1015LM'])}]    
    #     break
    # if 'VSHI4BTF' in unavailable_rms:
    #     constraints_list1 = [{356.7: ('=', ['VPOLSHI17C', 'VSTPE10', 'VSPUR1015LM'])}]
    #     constraints_list2 = [{356.7: ('=', ['VPOLSHI17C', 'VSTPE10', 'VSPUR1015LM'])}]
    #     constraints_list3 = [{356.7: ('=', ['VPOLSHI17C', 'VSTPE10', 'VSPUR1015LM'])}]
    #     constraints_list4 = [{356.7: ('=', ['VPOLSHI17C', 'VSTPE10', 'VSPUR1015LM'])}]
    #     break
    # if 'VPOLSHI17C' in unavailable_rms:    
    #     constraints_list1 = [{356.7: ('=', ['VSHI4BTF', 'VSTPE10', 'VSPUR1015LM'])}, {237.8: ('<=', ['VSHI4BTF'])}]
    #     constraints_list2 = [{356.7: ('=', ['VSHI4BTF', 'VSTPE10', 'VSPUR1015LM'])}, {237.8: ('<=', ['VSHI4BTF'])}]
    #     constraints_list3 = [{356.7: ('=', ['VSHI4BTF', 'VSTPE10', 'VSPUR1015LM'])}, {237.8: ('<=', ['VSHI4BTF'])}]
    #     constraints_list4 = [{356.7: ('=', ['VSHI4BTF', 'VSTPE10', 'VSPUR1015LM'])}, {237.8: ('<=', ['VSHI4BTF'])}] 
    #     break
    # if 'VSTPE10' in unavailable_rms:
    #     constraints_list1 = [{356.7: ('=', ['VSHI4BTF', 'VPOLSHI17C', 'VSPUR1015LM'])}, {237.8: ('<=', ['VSHI4BTF'])}]
    #     constraints_list2 = [{356.7: ('=', ['VSHI4BTF', 'VPOLSHI17C', 'VSPUR1015LM'])}, {237.8: ('<=', ['VSHI4BTF'])}]
    #     constraints_list3 = [{356.7: ('=', ['VSHI4BTF', 'VPOLSHI17C', 'VSPUR1015LM'])}, {237.8: ('<=', ['VSHI4BTF'])}]
    #     constraints_list4 = [{356.7: ('=', ['VSHI4BTF', 'VPOLSHI17C', 'VSPUR1015LM'])}, {237.8: ('<=', ['VSHI4BTF'])}]
    #     break
    # if 'VSPUR1015LM' in unavailable_rms:        
    #     constraints_list1 = [{356.7: ('=', ['VSHI4BTF', 'VPOLSHI17C', 'VSTPE10'])}, {237.8: ('<=', ['VSHI4BTF'])}]
    #     constraints_list2 = [{356.7: ('=', ['VSHI4BTF', 'VPOLSHI17C', 'VSTPE10'])}, {237.8: ('<=', ['VSHI4BTF'])}]
    #     constraints_list3 = [{356.7: ('=', ['VSHI4BTF', 'VPOLSHI17C', 'VSTPE10'])}, {237.8: ('<=', ['VSHI4BTF'])}]
    #     constraints_list4 = [{356.7: ('=', ['VSHI4BTF', 'VPOLSHI17C', 'VSTPE10'])}, {237.8: ('<=', ['VSHI4BTF'])}]        
    #     break
    # else:
    #     constraints_list1 = [{356.7: ('=', ['VSHI4BTF', 'VPOLSHI17C', 'VSTPE10', 'VSPUR1015LM'])}, {237.8: ('<=', ['VSHI4BTF'])}]
    #     constraints_list2 = [{356.7: ('=', ['VSHI4BTF', 'VPOLSHI17C', 'VSTPE10', 'VSPUR1015LM'])}, {237.8: ('<=', ['VSHI4BTF'])}]
    #     constraints_list3 = [{356.7: ('=', ['VSHI4BTF', 'VPOLSHI17C', 'VSTPE10', 'VSPUR1015LM'])}, {237.8: ('<=', ['VSHI4BTF'])}]
    #     constraints_list4 = [{356.7: ('=', ['VSHI4BTF', 'VPOLSHI17C', 'VSTPE10', 'VSPUR1015LM'])}, {237.8: ('<=', ['VSHI4BTF'])}]

    percent_dict = {'VCRAYVALLACSLT': 19.5, 'VOMNYA5ML': 260, 'VHAKUENKACCRS': 780, 'VXL10': 65.0, 'VSHI4BTF':237.8, 'VPOLSHI17C':356.7, 
                     'VSTPE10': 356.7, 'VSPUR1015LM': 356.7, 'XTITAN2': 13.0, 'VCAT850': 3.3, 'VDIDP': 29.3, 'VAMMO': 17.9, 'VDAMO': 14.9, 
                     'XDINP': 241.8, 'VDINCH': 241.8}

    new_dict = {}
    for key, value in percent_dict.items():
        if key is not unavailable_rms:
            new_dict[key] = value

    # percent_terms = pd.DataFrame.from_dict(percent_dict, orient='index').rename(columns = {0:'mass'})


    with st.beta_expander('Raw material conditions'):

        standard_options = ['VSHI4BTF', 'VPOLSHI17C', 'VSTPE10', 'VSPUR1015LM']
        rm_to_constrain_options = [rm for rm in standard_options if rm not in unavailable_rms]

        rm_to_constrain = st.multiselect("Select raw materials to constrain:", options = rm_to_constrain_options, 
                                         key = 'Constrained raw materials', default = None)

        for rm in rm_to_constrain:

            val = st.slider(label = 'Percentage ' + rm + ' of the total polymer, %', min_value = 0, max_value = 100, step = 1, key = 'constraining ' + rm + ' value')

            label_list = [rm +' = ', rm + ' ≥ ', rm + ' ≤ ']

            sign = st.radio(label = '', options =[i + str(val) + ' %' for i in label_list], index = 0, key = 'constraining ' + rm)
            sign = sign.split(' ')[1]
            st.write(sign[0])
            if sign == '≥':
                sign = '≤'
            elif sign == '≤':
                sign = '≥'

            for lizt in constraints_lists:
                lizt.append({val*0.01*new_dict[rm]:(sign, [rm])})

    #         constraints_list1.append({val*0.01*new_dict[rm]:(sign, [rm])})
    #         constraints_list2.append({val*0.01*new_dict[rm]:(sign, [rm])})
    #         constraints_list3.append({val*0.01*new_dict[rm]:(sign, [rm])})
    #         constraints_list4.append({val*0.01*new_dict[rm]:(sign, [rm])})

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
    #         if 'VSTPE10' in rm_to_constrain:
    #             stp_data = stp_data[stp_data['VSTPE10'] == 0]
            try:
                point_formulations = optimiser(stp_data, data).formulation_opt()
            except:
                point_formulations = stp_data
        else:
            point_formulations = optimiser(stp_data, data).formulation_opt() #REMEMBER function drops any rms that do not have a cost associated with them, i.e. if STPE10 does not have a rm cost it will not be included after the function is run.
            point_formulations = point_formulations.drop(['Cost difference compared to standard per kg (£)', 'Total Cost (£)'], axis = 1)
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

        #     constants1 = {key: val for key, val in constants1.items() if key not in unavailable_rms}
    #         for rm in unavailable_rms:
    #             if rm not in constants2:
    #                 lp2frame = stp_lp_optimiser('LP2', constants2, decision_variables, data.fillna(1000000000), constraints_lists[1])
    #                 frame_list.append(lp2frame)
    #             else:
    #                 continue

        #     constants2 = {key: val for key, val in constants2.items() if key not in unavailable_rms}
    #         for rm in unavailable_rms:
    #             if rm not in constants3:
    #                 lp3frame = stp_lp_optimiser('LP3', constants3, decision_variables, data.fillna(1000000000), constraints_lists[2])
    #                 frame_list.append(lp3frame)
    #             else:
    #                 continue
        #     constants3 = {key: val for key, val in constants3.items() if key not in unavailable_rms}
    #         for rm in unavailable_rms:
    #             if rm not in constants4:
    #                 lp4frame = stp_lp_optimiser('LP4', constants4, decision_variables, data.fillna(1000000000), constraints_lists[3])
    #                 frame_list.append(lp4frame)
    #             else:
    #                 continue
        #     constants4 = {key: val for key, val in constants4.items() if key not in unavailable_rms}

        final_df = pd.concat(frame_list, axis=0)
        final_df = pd.concat([final_df, point_formulations])
        cost_df = final_df[['Batch Size (kg)', 'Cost per kg (£)']]
        final_df = final_df.drop(['Batch Size (kg)', 'Cost per kg (£)'], axis = 1)
        final_df = pd.concat([final_df, cost_df], axis = 1) #adds batch size and cost per kg to end of df 
        final_df = final_df.fillna(0)

    #             standard_cost = final_df['Cost per kg (£)'].loc['Standard']

        final_df['Cost difference compared to standard per kg (£)'] = final_df['Cost per kg (£)'] - standard_cost
        final_df['Cost difference compared to standard per kg (£)'] = final_df['Cost difference compared to standard per kg (£)'].round(3)
        final_df = final_df.sort_values('Cost per kg (£)')
        final_df = final_df.drop(['Batch Size (kg)'], axis = 1)

        for rm in nan_cols:
            if rm in final_df.columns:
                final_df = final_df[final_df[rm] == 0]

        for i in constraints_lists:
            constraints_to_consider = i[2:]
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

        if final_df.shape[0] != 0:
            st.write(pd.DataFrame(final_df.iloc[0]).T)

        all_solutions = st.checkbox("Show all solutions", key = 'checkbox for showing all available solutions')
        if all_solutions:
            final_df

if __name__ == "__main__":
    if not st._is_running_with_streamlit: # noqa
        # If not running with streamlit, relaunch.
        # This is necessary because CODI will call `python app.py` to
        # launch the webapp, rather than `streamlit run app.py`.
        sys.argv = ["streamlit", "run", sys.argv[0], "--server.port", "805
        sys.exit(stcli.main())

    main()

#     else:
#         ### LP1 Polymer for constant VAMMO & VDINP ###
#         constants1 = {'VCRAYVALLACSLT': 19.5, 'VOMNYA5ML': 260, 'VHAKUENKACCRS': 780, 'VXL10': 65.0, 
#                      'XTITAN2': 13.0, 'VCAT850': 3.3, 'VDIDP': 29.3, 'VAMMO': 17.9, 'XDINP': 241.8}

#         lp1frame = stp_lp_optimiser('LP1', constants1, decision_variables, data.fillna(1000000000), constraints_list1)

#     #     constants1 = {key: val for key, val in constants1.items() if key not in unavailable_rms}

#     ### LP2 Polymer for constant VAMMO & VDINCH ###
#         constants2 = {'VCRAYVALLACSLT': 19.5, 'VOMNYA5ML': 260, 'VHAKUENKACCRS': 780, 'VXL10': 65.0, 
#                      'XTITAN2': 13.0, 'VCAT850': 3.3, 'VDIDP': 29.3, 'VAMMO': 17.9, 'VDINCH': 241.8}

#         lp2frame = stp_lp_optimiser('LP2', constants2, decision_variables, data.fillna(1000000000), constraints_list2)

#     #     constants2 = {key: val for key, val in constants2.items() if key not in unavailable_rms}

#     ### LP3 Polymer for constant VDAMO & VDINP ###
#         constants3 = {'VCRAYVALLACSLT': 19.5, 'VOMNYA5ML': 260, 'VHAKUENKACCRS': 780, 'VXL10': 65.0, 
#                      'XTITAN2': 13.0, 'VCAT850': 3.3, 'VDIDP': 29.3, 'VDAMO': 14.9, 'XDINP': 241.8}

#         lp3frame = stp_lp_optimiser('LP3', constants3, decision_variables, data.fillna(1000000000), constraints_list3)

#     #     constants3 = {key: val for key, val in constants3.items() if key not in unavailable_rms}

#         ### LP4 Polymer for constant VDAMO & VDINCH ###
#         constants4 = {'VCRAYVALLACSLT': 19.5, 'VOMNYA5ML': 260, 'VHAKUENKACCRS': 780, 'VXL10': 65.0, 
#                      'XTITAN2': 13.0, 'VCAT850': 3.3, 'VDIDP': 29.3, 'VDAMO': 14.9, 'VDINCH': 241.8}

#         lp4frame = stp_lp_optimiser('LP4', constants4, decision_variables, data.fillna(1000000000), constraints_list4)

#     #     constants4 = {key: val for key, val in constants4.items() if key not in unavailable_rms}

#         final_df = pd.concat([lp1frame, lp2frame, lp3frame, lp4frame, point_formulations])
#         cost_df = final_df[['Batch Size (kg)', 'Cost per kg (£)']]
#         final_df = final_df.drop(['Batch Size (kg)', 'Cost per kg (£)'], axis = 1)
#         final_df = pd.concat([final_df, cost_df], axis = 1)
#         final_df = final_df.fillna(0)
# #         standard_cost = final_df['Cost per kg (£)'].loc['Standard']

#         final_df['Cost difference compared to standard per kg (£)'] = final_df['Cost per kg (£)'] - standard_cost
#         final_df['Cost difference compared to standard per kg (£)'] = final_df['Cost difference compared to standard per kg (£)'].round(3)
#         final_df = final_df.sort_values('Cost per kg (£)')

#         for rm in nan_cols:
#             if rm in final_df.columns:
#                 final_df = final_df[final_df[rm] == 0]

#         st.write(pd.DataFrame(final_df.iloc[0]).T)

#         all_solutions = st.checkbox("Show all solutions", key = 'checkbox for showing all available solutions')
#         if all_solutions:
#             final_df

    

    

    

    

    

    

    

    

#     st.header('Data processing')

#     # choosing and cleaning categorical variables

#     with st.beta_expander('My data contains non-numerical variables (optional)'):
#         categorical_vars = st.multiselect(label = 'Select categorical variables:', options = all_variables)
#         continuous_vars = [i for i in all_variables if i not in categorical_vars]

#         for col in categorical_vars:
#             data[col] = data[col].astype(str).str.strip()
#             categorical_vals = sorted(list(set(data[col].dropna())))
#             st.write("Values found in column '" + str(col) + "': " + ", ".join(categorical_vals))

#             st.write('Replace non-numerical values in column ' + str(col))
#             categorical_vals_to_replace = st.multiselect(label = 'select value to replace', options = categorical_vals)
#             for val in categorical_vals_to_replace:
#                 new_val = st.text_input(label = 'replace ' + str(val) + ' with:')
#                 data[col] = data[col].replace(val, new_val)

#             dummy_dict[col] = pd.get_dummies(data[col], prefix = col)
#             data = pd.concat([data, pd.concat([dummy_dict[col] for col in categorical_vars], axis = 1)], axis = 1).drop(categorical_vars, axis = 1)


#     # populating blanks with zeroes

#     with st.beta_expander('Populate blanks in table with zeroes (optional)'):
#         st.write('Rows with any blanks in them are dropped before machine learning is initiated. Alternatively, the blanks can be populated with zeroes to preclude dropping. Only do this if a zero in place of a blank is scientifically reasonable.')
#         fillna_container = st.beta_container()
#         all_fillna_select = st.checkbox("Select all", key = 'checkbox for selecting all independent variables to populate with zeroes')

#         if all_fillna_select:
#             variables_to_fillna = fillna_container.multiselect("Select columns to populate:", [i for i in continuous_vars if i != dependent_variable], [i for i in continuous_vars if i != dependent_variable], key = 'Selecting all independent variables to populate with zeroes')
#         else:
#             variables_to_fillna = fillna_container.multiselect("Select columns to populate:", options = [i for i in continuous_vars if i != dependent_variable], key = 'Selecting independent variables to populate with zeroes')

#         data[variables_to_fillna] = data[variables_to_fillna].fillna(0)


#     # ensuring data numericity

#     with st.beta_expander('Ensuring data numericity for modelling (required)'):
#         try:
#             continuous_variables
#         except:
#             continuous_variables = all_variables

#         for col in continuous_variables:
#             try:
#                 data[col] = data[col].astype(np.float64)
#             except:
#                 for val in data[col]:
#                     try:
#                         np.float64(val)
#                     except:
#                         replace_val = st.number_input('Replace ' + str(val) + ' with')
#                         data[col] = data[col].replace(val, replace_val)

#         data = data[[col for col in all_variables if data[col].sum()!=0]].dropna(how = 'any')
#         numericity_bool = True
#         st.write('Nothing to see here.')


# # ## SELECTING MODEL PARAMETERS ###

#     st.header('Setting up the model')

#     with st.beta_expander('Selecting model parameters'):
#         model_name = st.text_input('Save model as')
#         model_string = st.selectbox(label = 'Select model', options = [
#             'linear', 'regularised linear' , 'nearest neighbours', 'support vector machine', 
#             'relevance vector machine', 'decision tree', 'random forest', 'gradient boosting',
#             'large-data gradient boosting', 'extreme gradient boosting'])
#         gridsearch_time_limit = st.number_input(label = 'Time limit for running model search (s)', min_value = 30, max_value = 2000, value = 120, step = 30)

#         advanced_options = st.checkbox(label = 'Show advanced options')
#         if advanced_options:
#             advanced_options = False
#             refit_time_limit = st.number_input(label = 'Time limit for refitting model (s)', min_value = 1, max_value = 300, value = 10, step = 10)
#             train_test_ratio = st.slider(label = 'Percentage of data allocated to test partition', min_value = 0.01, max_value = 0.5, value = 0.2, step = 0.01)
#             kfoldv = st.slider(label = 'Number of testing rounds per model. Higher values can generate better models at the cost of longer training time', min_value = 2, max_value = 10, value = 5, step = 1)
#         else:
#             train_test_ratio, kfoldv, refit_time_limit = 0.2, 5, 10

#         stratify_select = st.checkbox(label = 'Stratify data')
#         if stratify_select:
#             stratify_select = False
#             stratify = st.selectbox(label = 'Select variable to use for stratified sampling', options = list(data.columns))
#         else:
#             stratify = None


#     init = st.button(label = 'Start modelling')
#     if init:
#         st.session_state['modelling_finish'] = False
#         st.session_state['modelling_init'] = True

#     ### MODEL FITTING ###

#     if st.session_state['modelling_init'] and not st.session_state['modelling_finish']:
# #         st.text('Model fitting initialised')

#         model_search = fit_model(model_name = model_name, model_string = model_string, data = data,
#                                   independent_variables = independent_variables, dependent_variable = dependent_variable,
#                                   train_test_ratio = train_test_ratio, kfoldv = kfoldv, gridsearch_time_limit = gridsearch_time_limit,
#                                   refit_time_limit = refit_time_limit, stratify = stratify)

#         st.session_state ['modelling_finish'] = True

#         try:
#             st.session_state['model'] = model_search.best_model
#             st.session_state['model_search'] = model_search
#             st.success('Model fitting complete')

#         except Exception as e:
#             'No acceptable model found. Please try increasing the time limits or select a different model: ' + str(e)

#     if st.session_state['model'] is not None:

#         ### MODEL ANALYSIS ###

#         st.header('Model analysis')

#         st.set_option('deprecation.showPyplotGlobalUse', False)
#         with st.beta_expander('Model analysis'):
#             #st.write(dir(st.session_state['model_search']))
#             #fig, ax = st.session_state['model_search'].plot_results()
#             st.session_state['model_search'].plot_results()
#             fig, ax = st.session_state['model_search'].plot, st.session_state['model_search'].ax
#             st.pyplot(fig)

#         with st.beta_expander('Linear coefficients for independent variables'):
#             scaled_data = data[independent_variables + [dependent_variable]].apply(zscore)
#             #st.write(dir(st.session_state['model_search']))
#             lin_scaled_model = LinearRegression().fit(scaled_data[independent_variables], scaled_data[dependent_variable])
#             lin_model = LinearRegression().fit(data[independent_variables], data[dependent_variable])
#             coef_frame = pd.DataFrame(columns = independent_variables, data = [lin_model.coef_, lin_scaled_model.coef_], index = ['normalised influence', 'denormalised influence']).T
#             #fig, (ax1, ax2) = plt.subplots(ncols = 2)
#             fig, ax = plt.subplots()
#             fig.set_size_inches(15,20)
#             ax.xaxis.tick_top()
# #             ax2.xaxis.tick_top()
#             sns.heatmap(coef_frame[['normalised influence']].sort_values(by = 'normalised influence', ascending = False), ax = ax, annot = True, cmap = 'plasma') #cbar = False)
#             #sns.heatmap(coef_frame[['denormalised influence']], ax = ax2, annot = True, cmap = 'plasma', cbar = False)
#             #ax2.get_yaxis().set_visible(False)
#             plt.subplots_adjust(wspace = 0)
#             st.write(fig)


#         ### PREDICTING USING THE MODEL ###

#         st.header('Model predictions')

#         with st.beta_expander('Predicting dependent from single-entry independent variables'):
#             with st.form('independents for foward prediction'):
#                 input_formula = []
#                 for variable in independent_variables:
#                     input_formula.append(st.number_input(label = variable, value = 0, min_value = min([0, data[variable].min()])))

#                 normalise_bool = st.checkbox('Normalise variables')
#                 if normalise_bool:
#                     normalise_container = st.beta_container()
#                     all_normalise_select = st.checkbox("Select all", key = 'checkbox for normalising prediction inputs')
#                     if all_normalise_select:
#                         variables_to_fillna = normalise_container.multiselect("Select columns to populate:", independent_variables, independent_variables, key = 'Selecting all independent variables to normalise')
#                     else:
#                         variables_to_fillna = normalise_container.multiselect("Select columns to populate:", options = independent_variables, key = 'Selecting independent variables to normalise')
#                     normalise_sum = st.number_input(label = 'Normalise sum of variables to:', value = 1)

#                 predict_bool = st.form_submit_button(label = 'Predict ' + str(dependent_variable))

#             if predict_bool:
#                 st.session_state['predict_init'] = True

#             if st.session_state['predict_init']:
#                 try:
#                     input_formula = np.array(input_formula)
#                     input_formula = input_formula/input_formula.sum()
#                     prediction = st.session_state['model'].predict(input_formula)
#                     st.write(prediction)
#                     st.stop()

#                 except Exception as f:
#                     'Error: ' + str(f)
