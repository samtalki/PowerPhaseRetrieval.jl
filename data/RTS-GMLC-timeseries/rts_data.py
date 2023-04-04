import numpy as np
import pandas as pd

real_time_load_data = "/home/sam/Research/PowerPhaseRetrieval/data/RTS-GMLC-timeseries/REAL_TIME_regional_Load.csv"
day_ahead_load_data = "/home/sam/Research/PowerPhaseRetrieval/data/RTS-GMLC-timeseries/DAY_AHEAD_regional_Load.csv"
bus_file = "/home/sam/Research/PowerPhaseRetrieval/data/RTS-GMLC-timeseries/bus.csv"

def get_rts_load_fractions(bus_df):
    fractions  = pd.DataFrame(columns=["Bus ID","Area","MW Fraction","MVAR Fraction"])
    for area in bus_df["Area"].unique():
        area_load = bus_df[bus_df["Area"]==area]
        area_load["MW Fraction"] = area_load["MW Load"]/area_load["MW Load"].sum()
        area_load["MVAR Fraction"] = area_load["MVAR Load"]/area_load["MVAR Load"].sum()
        fractions = fractions.append(area_load[["Bus ID","Area","MW Fraction","MVAR Fraction"]])
    return fractions

def get_rts_power_factors(bus_df):
    entries = []
    for bus_id in bus_df["Bus ID"]:
        bus_mw = bus_df[bus_df["Bus ID"]==bus_id]["MW Load"].values[0]
        bus_mvar = bus_df[bus_df["Bus ID"]==bus_id]["MVAR Load"].values[0]
        if bus_mw == 0 and bus_mvar == 0:
            bus_pf = 1
        elif bus_mvar == 0:
            bus_pf = 1
        else:
            bus_pf = bus_mw/np.sqrt(bus_mw**2+bus_mvar**2)
        entries.append({"Bus ID":bus_id,"Power Factor":bus_pf})
    return pd.DataFrame(entries,columns=["Bus ID","Power Factor"])

def get_rts_nodal_data_realtime(bus_file=bus_file,real_time_load_data=real_time_load_data):
    # Read in the data
    bus_df = pd.read_csv(bus_file)
    load_df = pd.read_csv(real_time_load_data)
    # Get the bus fractions and power factors
    fractions = get_rts_load_fractions(bus_df)    
    power_factors = get_rts_power_factors(bus_df)
    # Setup nodal dataframes
    nodal_mw_data = pd.DataFrame(columns=bus_df["Bus ID"],index=load_df.index)
    nodal_mvar_data = pd.DataFrame(columns=bus_df["Bus ID"],index=load_df.index)
    # Get the nodal data
    for bus_id in bus_df["Bus ID"]:
        bus_zone = bus_df[bus_df["Bus ID"]==bus_id]["Area"].values[0]
        bus_power_factor = power_factors[power_factors["Bus ID"]==bus_id]["Power Factor"].values[0]
        fraction = fractions[fractions["Bus ID"]==bus_id]["MW Fraction"].values[0]
        nodal_mw_data[bus_id] = fraction*load_df[str(bus_zone)] 
        nodal_mvar_data[bus_id] = nodal_mw_data[bus_id]*np.tan(np.arccos(bus_power_factor))
    return nodal_mw_data,nodal_mvar_data


def get_rts_nodal_data_day_ahead(bus_file=bus_file,day_ahead_load_data=day_ahead_load_data):
    # Read in the data
    bus_df = pd.read_csv(bus_file)
    load_df = pd.read_csv(day_ahead_load_data)
    # Get the bus fractions and power factors
    fractions = get_rts_load_fractions(bus_df)    
    power_factors = get_rts_power_factors(bus_df)
    # Setup nodal dataframes
    nodal_mw_data = pd.DataFrame(columns=bus_df["Bus ID"],index=load_df.index)
    nodal_mvar_data = pd.DataFrame(columns=bus_df["Bus ID"],index=load_df.index)
    # Get the nodal data
    for bus_id in bus_df["Bus ID"]:
        bus_zone = bus_df[bus_df["Bus ID"]==bus_id]["Area"].values[0]
        bus_power_factor = power_factors[power_factors["Bus ID"]==bus_id]["Power Factor"].values[0]
        fraction = fractions[fractions["Bus ID"]==bus_id]["MW Fraction"].values[0]
        nodal_mw_data[bus_id] = fraction*load_df[str(bus_zone)] 
        nodal_mvar_data[bus_id] = nodal_mw_data[bus_id]*np.tan(np.arccos(bus_power_factor))
    return nodal_mw_data,nodal_mvar_data
