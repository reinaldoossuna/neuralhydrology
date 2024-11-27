from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import xarray
import calendar

from neuralhydrology.datasetzoo.basedataset import BaseDataset
from neuralhydrology.utils.config import Config


class CabraMS(BaseDataset):
    def __init__(self,
                 cfg: Config,
                 is_train: bool,
                 period: str,
                 basin: str = None,
                 additional_features: List[Dict[str, pd.DataFrame]] = [],
                 id_to_int: Dict[str, int] = {},
                 scaler: Dict[str, Union[pd.Series, xarray.DataArray]] = {}):
        super(CabraMS, self).__init__(cfg=cfg,
                                       is_train=is_train,
                                       period=period,
                                       basin=basin,
                                       additional_features=additional_features,
                                       id_to_int=id_to_int,
                                       scaler=scaler)

    def _load_basin_data(self, basin: str) -> pd.DataFrame:
        """Load input and output data from text files."""
        # get forcings
        dfs = []
        for forcing in self.cfg.forcings:
            df, area = load_cabra_forcings(self.cfg.data_dir, basin, forcing)

            # rename columns
            if len(self.cfg.forcings) > 1:
                df = df.rename(columns={col: f"{col}_{forcing}" for col in df.columns})
            dfs.append(df)
        df = pd.concat(dfs, axis=1)

        # add discharge
        print(df)
        df['QObs(mm/d)'] = load_cabra_discharge(self.cfg.data_dir, basin, area)

        # replace invalid discharge values by NaNs
        qobs_cols = [col for col in df.columns if "qobs" in col.lower()]
        for col in qobs_cols:
            df.loc[df[col] < 0, col] = np.nan
        
        print("missing Qobs all data: ", df['QObs(mm/d)'].isna().sum()/len(df))
        print("missing PREC all data: ", df['p_ens'].isna().sum()/len(df))
        print(basin)
            
	    	
        return df

    def _load_attributes(self) -> pd.DataFrame:
        return load_cabra_attributes(self.cfg.data_dir, basins=self.basins)


def basin_area(path: Path,basin: str)->int:
    data=pd.read_csv(path,encoding= 'unicode_escape',sep='\t')
    data.columns = data.columns.str.replace(' ', '')
    data=data.drop(0).astype('float32')
    data=data.reset_index().drop("index",axis=1)
    area=data.loc[data['CABra_MS_ID']==float(basin)]['catch_area'].values
    return int(area)

def type_converter(data):
    for c in data.columns.to_list():
        try:
            data[c] = data[c].astype('float64')
        except:
            data[c] = data[c]
    return data

def load_cabra_attributes(data_dir: Path, basins: List[str] = []) -> pd.DataFrame:
    attributes_path = Path(data_dir) / 'CABra_MS/CABra_MS_attributes'
    if not attributes_path.exists():
        raise RuntimeError(f"Attribute folder not found at {attributes_path}")

    txt_files = attributes_path.glob('CABra_MS_*.txt')

    dfs = []
    for txt_file in txt_files:
        df_temp = pd.read_csv(txt_file,encoding= 'unicode_escape',sep='\t')
        df_temp.columns = df_temp.columns.str.replace(' ', '')
        df_temp = df_temp.drop(0)
        df_temp = type_converter(df_temp)
        df_temp.CABra_MS_ID = df_temp.CABra_MS_ID.astype('int')
        df_temp.CABra_MS_ID = df_temp.CABra_MS_ID.astype('str')
        df_temp = df_temp.reset_index().drop("index",axis=1).set_index('CABra_MS_ID')
        df_temp = df_temp.fillna(0)
        dfs.append(df_temp)

    df = pd.concat(dfs, axis=1)

    if basins:
        if any(b not in df.index for b in basins):
            raise ValueError('Some basins are missing static attributes.')
        df = df.loc[basins]

    return df

def load_cabra_forcings(data_dir: Path, basin: str, forcings: str) -> Tuple[pd.DataFrame, int]:   
    forcing_path = data_dir / 'CABra_MS/climate' / forcings
    attributes_path = data_dir / 'CABra_MS/CABra_MS_attributes/CABra_MS_topography_attributes.txt'

    if not forcing_path.is_dir():
        raise OSError(f"{forcing_path} does not exist")
    
    if int(basin) in range(1,150):
        file_path = Path(str(forcing_path)+'/CABra_MS_'+ basin+'_climate.txt')
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {forcing_path}')
    forcing_df = pd.read_csv(file_path,skiprows=0,encoding= 'unicode_escape',sep='\t')

    forcing_df.columns = forcing_df.columns.str.replace(' ', '')
    forcing_df = forcing_df.drop(0).astype('float32')
    dates = (forcing_df.Year.map(int).map(str) + "/" + forcing_df.Month.map(int).map(str) + "/"+  forcing_df.Day.map(int).map(str))
    forcing_df["date"] = pd.to_datetime(dates, format="%Y/%m/%d")
    forcing_df = forcing_df.set_index("date")
    area = basin_area(attributes_path,int(basin))

    return forcing_df.loc, area


def fill_discharge(df):
    df['doy']=df.date.dt.dayofyear
    doy_dict = dict(zip(df.doy, df.groupby(df.doy).mean()['Streamflow(m³s)']))
    df['Streamflow(m³s)'] = df['Streamflow(m³s)'].fillna(df['doy'].map(doy_dict))
    return df

def load_cabra_discharge(data_dir: Path, basin: str, area: int) -> pd.Series:
    print('*********************************** loading discharge - 1 ***********************************')
    discharge_path = data_dir / 'CABra_MS/streamflow'
    if not discharge_path.is_dir():
        raise OSError(f"{discharge_path} does not exist")
    
    if int(basin) in range(1,736):
        file_path = Path(str(discharge_path)+'/CABra_MS_'+ basin+'_streamflow.txt')
    else:
        raise FileNotFoundError(f'No file for Basin {basin} at {discharge_path}')

    col_names = ["Year",'Month','Day','Streamflow(m³s)','Quality']
    print('*********************************** loading discharge - 2 ***********************************')
    discharge = pd.read_csv(file_path, skiprows=1,encoding= 'unicode_escape',sep='\t',names=col_names)

    print('*********************************** loading discharge - 3 ***********************************')
    discharge.columns = discharge.columns.str.replace(' ', '')
    print('*********************************** loading discharge - 4 ***********************************')
    discharge=discharge.astype('float32')
    dates = (discharge.Year.map(int).map(str) + "/" + discharge.Month.map(int).map(str) + "/"+  discharge.Day.map(int).map(str))
    discharge['date'] = pd.to_datetime(dates, format="%Y/%m/%d")

    print('*********************************** loading discharge - 5 ***********************************')
    discharge = fill_discharge(discharge)
    print('*********************************** loading discharge - 6 ***********************************')
    discharge = discharge.set_index("date").loc['1980-01-01':'2025-12-31']
    # discharge = discharge.set_index("date").loc['1980-01-01':'2023-09-12']
    print('*********************************** loading discharge - 7 ***********************************')
    # discharge = discharge.drop_duplicates(subset=['date'], keep='first')
    print('*********************************** loading discharge - 8 ***********************************')
    #area = basin_area(attributes_path,int(basin))
    #discharge=discharge.dropna(axis=0)'])
    #discharge=discharge.fillna(discharge.mean(skipna=True))
    

    # normalize discharge from cubic meter per second to mm per day
    discharge['QObs'] = (discharge['Streamflow(m³s)']/(area*10**6))*86400*1000

    # # normalize discharge from cubic meter per second to mm per month ###################### USE THIS FOR MONTHLY FORECAST
    # discharge['QObs'] = (discharge['Streamflow(m³s)']/(area*10**6))*86400*1000*(calendar.monthrange(discharge['Year'], discharge['Month'])[1])
    print('*********************************** loading discharge - 9 ***********************************')

    return discharge.QObs
