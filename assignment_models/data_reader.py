import pandas as pd
import numpy as np
from collections import defaultdict
from datetime import datetime

def load_data(input_file):

    start = datetime.now()
    input_data = {}
    centerDemand = pd.read_excel(input_file, sheet_name='CenterDemand')
    specialistHourlyCapacity = pd.read_excel(input_file, sheet_name='SpecialistHourlyCapacity')
    specialistAvailability = pd.read_excel(input_file, sheet_name='SpecialistAvailability')
    specialistPreference = pd.read_excel(input_file, sheet_name='SpecialistPreference')
    centerMaxBacklogSize = pd.read_excel(input_file, sheet_name='CenterMaxBacklogSize')
    centerFullTimeCoverPerc = pd.read_excel(input_file, sheet_name='CenterFullTimeCoverPerc')
    hourIsMixed = pd.read_excel(input_file, sheet_name='HourIsMixed')


    D = centerDemand['d'].unique().tolist()
    H = centerDemand['h'].unique().tolist()
    C = centerDemand['c'].unique().tolist()
    S = specialistPreference['s'].unique().tolist()
    SC = list(zip(specialistPreference['s'],specialistPreference['c']))

    filtered_DHS = specialistAvailability.merge(
        specialistPreference[['s']].drop_duplicates(), on='s'
    ).merge(
        centerDemand[['h']].drop_duplicates(), on='h'
    )
    DHS = list(zip(filtered_DHS['d'], filtered_DHS['h'], filtered_DHS['s']))


    end3 = datetime.now()

    input_data['centerDemand'] = centerDemand.set_index(['d', 'h', 'c'])['CenterDemand'].to_dict()
    input_data['specialistPreference'] = specialistPreference.set_index(['s','c'])['SpecialistPreference'].to_dict()

    specialistHourlyCapacity = specialistHourlyCapacity[specialistHourlyCapacity['s'].isin(S)]
    input_data['specialistHourlyCapacity'] = specialistHourlyCapacity.set_index(['s'])['SpecialistHourlyCapacity'].to_dict()
    input_data['specialistIsFullTime'] = specialistHourlyCapacity.set_index(['s'])['SpecialistIsFullTime'].to_dict()

    input_data['specialistAvailability'] = specialistAvailability.set_index(['d', 'h', 's'])['SpecialistAvailability'].to_dict()
    input_data['centerMaxBacklogSize'] = centerMaxBacklogSize.set_index(['d', 'h', 'c'])['CenterMaxBacklogSize'].to_dict()
    input_data['centerFullTimeCoverPerc'] = centerFullTimeCoverPerc.set_index(['c'])['CenterFullTimeCoverPerc'].to_dict()

    input_data['hourIsMixed'] = hourIsMixed .set_index(['h'])['HourIsMixed'].to_dict()



    end = datetime.now()

    print(f"{datetime.now()} finish reading {end - start}")

    return D,H,C,S,DHS,SC,input_data

def load_result(file_path):
    result_data = {}
    specialistAtCenterByHour = pd.read_excel(file_path, sheet_name='SpecialistAtCenterByHour')
    result_data['specialistAtCenterByHour'] = specialistAtCenterByHour.set_index(['d', 'h', 's','c'])['specialistAtCenterByHour'].to_dict()
    return result_data

if __name__ == "__main__":
    input_file = 'Data/input.xlsx'
    D,H,C,S,DHS,SC,input_data = load_data(input_file)