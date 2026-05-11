import pandas as pd
import numpy as np

# Global cache to prevent redundant processing on Render (Free Tier RAM optimization)
_V2_CACHE = None

def get_v2_data():
    global _V2_CACHE
    if _V2_CACHE is not None:
        return _V2_CACHE

    print("Loading 2025 Seasonal Baseline into memory...")
    
    df_arr = pd.read_csv('data/inputs/DAA_Weekly_Flight_Demand_Arrivals_2025.csv')
    df_dep = pd.read_csv('data/inputs/DAA_Weekly_Flight_Demand_Departures_2025.csv')
    df_util = pd.read_csv('data/inputs/DAA_Weekly_Stand_Utilisation_2025.csv')

    df_arr['Week_Start'] = pd.to_datetime(df_arr['Week_Start'])
    df_dep['Week_Start'] = pd.to_datetime(df_dep['Week_Start'])
    df_util['Week_Start'] = pd.to_datetime(df_util['Week_Start'])
    
    # ── S1 Demand ──
    df_combined = pd.concat([df_arr, df_dep])
    df_combined['Month'] = df_combined['Week_Start'].dt.strftime('%b')
    df_arr['Month'] = df_arr['Week_Start'].dt.strftime('%b')
    df_dep['Month'] = df_dep['Week_Start'].dt.strftime('%b')
    df_arr['Month_Num'] = df_arr['Week_Start'].dt.month
    df_dep['Month_Num'] = df_dep['Week_Start'].dt.month
    df_combined['Month_Num'] = df_combined['Week_Start'].dt.month
    
    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    MONTHS_ORDER = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]
    
    dep_m = df_dep.groupby('Month_Num')['Weekly_Movements'].sum().sort_index().astype(int).tolist()
    arr_m = df_arr.groupby('Month_Num')['Weekly_Movements'].sum().sort_index().astype(int).tolist()
    
    wk_weekly_dep = df_dep.groupby('Week_Start')['Weekly_Movements'].sum().sort_index()
    wk_weekly_arr = df_arr.groupby('Week_Start')['Weekly_Movements'].sum().sort_index()
    
    WK_LBL = [d.strftime('%d %b') for d in wk_weekly_dep.index]
    WK_DEP = wk_weekly_dep.astype(int).tolist()
    WK_ARR = wk_weekly_arr.astype(int).tolist()
    
    MWD = {}
    for ym in MONTHS_ORDER:
        m = int(ym.split('-')[1])
        # Find weeks that start in this month
        mask = wk_weekly_dep.index.month == m
        labels = [d.strftime('W%V') for d in wk_weekly_dep.index[mask]]
        dep_vals = wk_weekly_dep[mask].astype(int).tolist()
        arr_vals = wk_weekly_arr[mask].astype(int).tolist()
        MWD[ym] = {"labels": labels, "dep": dep_vals, "arr": arr_vals}
        
    # ── S2 Ops ──
    top_airlines = df_combined.groupby('Airline_Name')['Weekly_Movements'].sum().nlargest(10).index.tolist()
    AIR_N = top_airlines
    
    air_dep_s = df_dep.groupby('Airline_Name')['Weekly_Movements'].sum()
    air_arr_s = df_arr.groupby('Airline_Name')['Weekly_Movements'].sum()
    
    AIR_DEP = [int(air_dep_s.get(a, 0)) for a in AIR_N]
    AIR_ARR = [int(air_arr_s.get(a, 0)) for a in AIR_N]
    
    # Mkt Share
    top5_air = df_combined.groupby('Airline_Name')['Weekly_Movements'].sum().nlargest(5)
    MS_L = top5_air.index.tolist() + ["Others"]
    MS_V = top5_air.astype(int).tolist() + [int(df_combined['Weekly_Movements'].sum() - top5_air.sum())]
    
    # Categories
    cat_s = df_combined.groupby('Flight_Category')['Weekly_Movements'].sum().sort_values(ascending=False)
    CAT_L = cat_s.index.tolist()
    CAT_V = cat_s.astype(int).tolist()
    
    # Fleet
    flt_s = df_combined.groupby('Aircraft_Family')['Weekly_Movements'].sum().sort_values(ascending=False).head(7)
    FLT_L = flt_s.index.tolist()
    FLT_V = flt_s.astype(int).tolist()
    
    # CBP
    cbp_mask_arr = df_arr['Flight_Category'] == 'Transatlantic CBP'
    cbp_mask_dep = df_dep['Flight_Category'] == 'Transatlantic CBP'
    cbp_air = pd.concat([df_arr[cbp_mask_arr], df_dep[cbp_mask_dep]]).groupby('Airline_Name')['Weekly_Movements'].sum().sort_values(ascending=False).index.tolist()
    CBP_A = sorted(cbp_air)
    CBP_D = [int(df_dep[cbp_mask_dep & (df_dep['Airline_Name'] == a)]['Weekly_Movements'].sum()) for a in CBP_A]
    CBP_AR = [int(df_arr[cbp_mask_arr & (df_arr['Airline_Name'] == a)]['Weekly_Movements'].sum()) for a in CBP_A]
    
    # ── S3 Pressure ──
    pier_s = df_util.groupby('Pier')['Avg_Daily_Occupancy_Hrs'].mean().sort_values(ascending=False)
    PIER_N = pier_s.index.tolist()
    PIER_U = pier_s.round(1).tolist()
    
    gw_s = df_util.groupby('Week_Start')['Avg_Daily_Occupancy_Hrs'].mean().sort_index()
    GW_LBL = [d.strftime('%d %b') for d in gw_s.index]
    GW_VAL = gw_s.round(2).tolist()
    
    # ── S4 T1 vs T2 ──
    t1_dep = df_dep[df_dep['Terminal'] == 'T1']
    t1_arr = df_arr[df_arr['Terminal'] == 'T1']
    t1_mov = pd.concat([t1_dep, t1_arr])
    
    t2_dep = df_dep[df_dep['Terminal'] == 'T2']
    t2_arr = df_arr[df_arr['Terminal'] == 'T2']
    t2_mov = pd.concat([t2_dep, t2_arr])
    
    T1_M = t1_mov.groupby('Month_Num')['Weekly_Movements'].sum().sort_index().astype(int).tolist()
    T2_M = t2_mov.groupby('Month_Num')['Weekly_Movements'].sum().sort_index().astype(int).tolist()
    
    ALL_CAT = sorted(df_combined['Flight_Category'].dropna().unique().tolist())
    T1_CAT = [int(t1_mov[t1_mov['Flight_Category'] == c]['Weekly_Movements'].sum()) for c in ALL_CAT]
    T2_CAT = [int(t2_mov[t2_mov['Flight_Category'] == c]['Weekly_Movements'].sum()) for c in ALL_CAT]
    
    _V2_CACHE = {
        "MONTHS": MONTHS, "MONTHS_ORDER": MONTHS_ORDER,
        "DEP_M": dep_m, "ARR_M": arr_m,
        "WK_LBL": WK_LBL, "WK_DEP": WK_DEP, "WK_ARR": WK_ARR,
        "MWD": MWD,
        "AIR_N": AIR_N, "AIR_DEP": AIR_DEP, "AIR_ARR": AIR_ARR,
        "MS_L": MS_L, "MS_V": MS_V,
        "CAT_L": CAT_L, "CAT_V": CAT_V,
        "FLT_L": FLT_L, "FLT_V": FLT_V,
        "CBP_A": CBP_A, "CBP_D": CBP_D, "CBP_AR": CBP_AR,
        "PIER_N": PIER_N, "PIER_U": PIER_U,
        "GW_LBL": GW_LBL, "GW_VAL": GW_VAL,
        "ALL_CAT": ALL_CAT, "T1_M": T1_M, "T2_M": T2_M, "T1_CAT": T1_CAT, "T2_CAT": T2_CAT
    }
    return _V2_CACHE
