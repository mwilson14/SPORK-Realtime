import numpy as np
from metpy.units import units
from metpy.plots import StationPlot
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS
from datetime import datetime, timedelta
import netCDF4
from metpy.calc import wind_direction

def get_RAP_data_realtime(radar_lon, radar_lat, time_start):
    #This definition will grab RAP data from around a given radar site for the ZDR arc algorithm.
    #cat = TDSCatalog('https://www.ncei.noaa.gov/thredds/catalog/model-rap130anl-old/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/catalog.html?dataset=rap130anl-old/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/rap_130_'+str(time_start.year)+str(month)+str(day)+'_'+str(hour)+'00_000.grb2')
    #cat = TDSCatalog('https://thredds.ucar.edu/thredds/catalog/grib/NCEP/RAP/CONUS_13km/catalog.html')
    print(radar_lon, radar_lat)
    hour = time_start.hour
    if hour < 10:
        hour = '0'+str(hour)
    day = time_start.day
    if day < 10:
        day = '0'+str(day)
    month = time_start.month
    if month < 10:
        month = '0'+str(month)
    
    file = 0

    ## 1    
    if file == 0:
        try:
            cat = TDSCatalog('https://www.ncei.noaa.gov/thredds/catalog/model-rap130anl-old/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/catalog.html?dataset=rap130anl-old/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/rap_130_'+str(time_start.year)+str(month)+str(day)+'_'+str(hour)+'00_000.grb2')
            file = 1
            print("We have the file we need1!")
        except:
            file = 0
            print("No file yet...")

    ## 2
    if file == 0:
        try:
            cat = TDSCatalog('https://www.ncei.noaa.gov/thredds/catalog/model-ruc130anl-old/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/catalog.html?dataset=ruc130anl-old/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/ruc_130_'+str(time_start.year)+str(month)+str(day)+'_'+str(hour)+'00_000.grb2')
            file = 1
            print("We have the file we need2!")
        except:
            file = 0
            print("No file yet...")

    ## 3
    if file == 0:
        try:
            cat = TDSCatalog('https://www.ncei.noaa.gov/thredds/catalog/model-rap252anl-old/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/catalog.html?dataset=rap252anl-old/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/rap_252_'+str(time_start.year)+str(month)+str(day)+'_'+str(hour)+'00_000.grb2')
            file = 1
            print("We have the file we need3!")
        except:
            file = 0
            print("No file yet...")

    ## 4
    if file == 0:
        try:
            cat = TDSCatalog('https://www.ncei.noaa.gov/thredds/catalog/model-ruc252anl-old/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/catalog.html?dataset=ruc252anl-old/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/ruc_252_'+str(time_start.year)+str(month)+str(day)+'_'+str(hour)+'00_000.grb2')
            file = 1
            print("We have the file we need4!")
        except:
            file = 0
            print("No file yet...")        

    ## 1b
    if file == 0:
        try:
            cat = TDSCatalog('https://www.ncei.noaa.gov/thredds/catalog/model-rap130anl/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/catalog.html?dataset=rap130anl/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/rap_130_'+str(time_start.year)+str(month)+str(day)+'_'+str(hour)+'00_000.grb2')
            file = 1
            print("We have the file we need1b!")
        except:
            file = 0
            print("No file yet...")

    ## 2b
    if file == 0:
        try:
            cat = TDSCatalog('https://www.ncei.noaa.gov/thredds/catalog/model-ruc130anl/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/catalog.html?dataset=ruc130anl/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/ruc_130_'+str(time_start.year)+str(month)+str(day)+'_'+str(hour)+'00_000.grb2')
            file = 1
            print("We have the file we need2b!")
        except:
            file = 0
            print("No file yet...")

    ## 3b
    if file == 0:
        try:
            cat = TDSCatalog('https://www.ncei.noaa.gov/thredds/catalog/model-rap252anl/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/catalog.html?dataset=rap252anl/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/rap_252_'+str(time_start.year)+str(month)+str(day)+'_'+str(hour)+'00_000.grb2')
            file = 1
            print("We have the file we need3b!")
        except:
            file = 0
            print("No file yet...")

    ## 4b
    if file == 0:
        try:
            cat = TDSCatalog('https://www.ncei.noaa.gov/thredds/catalog/model-ruc252anl/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/catalog.html?dataset=ruc252anl/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/ruc_252_'+str(time_start.year)+str(month)+str(day)+'_'+str(hour)+'00_000.grb2')
            file = 1
            print("We have the file we need4b!")
        except:
            file = 0
            print("No file yet...")


    ## 5
    #if file == 0:
    #    try:
    #        cat = TDSCatalog('https://www.ncdc.noaa.gov/thredds/catalog/model-rap130anl-old/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/catalog.html?dataset=rap130anl-old/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/rap_130_'+str(time_start.year)+str(month)+str(day)+'_'+str(hour)+'00_000.grb2')
    #        file = 1
    #        print("We have the file we need!")
    #    except:
    #        file = 0
    #        print("No file yet...")

    ## 6
    #if file == 0:
    #    try:
    #        cat = TDSCatalog('https://www.ncdc.noaa.gov/thredds/catalog/model-rap130anl/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/catalog.html?dataset=rap130anl-old/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/rap_130_'+str(time_start.year)+str(month)+str(day)+'_'+str(hour)+'00_000.grb2')
    #        file = 1
    #        print("We have the file we need...rap130anal!")
    #    except:
    #        file = 0
    #        print("No file yet...rap130anal")

    ## 7
    #if file == 0:
    #    try:
    #        cat = TDSCatalog('https://www.ncdc.noaa.gov/thredds/catalog/model-rap252anl/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/catalog.html?dataset=rap130anl-old/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/rap_130_'+str(time_start.year)+str(month)+str(day)+'_'+str(hour)+'00_000.grb2')
    #        file = 1
    #        print("We have the file we need...rap252anal!")
    #    except:
    #        file = 0
    #        print("No file yet...rap252anal")

    ## 8
    #if file == 0:
    #    try:
    #        cat = TDSCatalog('https://www.ncei.noaa.gov/thredds/catalog/model-rap130anl-old/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/catalog.html?dataset=rap130anl-old/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/rap_130_'+str(time_start.year)+str(month)+str(day)+'_'+str(hour)+'00_000.grb2')
    #        file = 1
    #        print("We have the file we need!")
    #    except:
    #        file = 0
    #        print("No file yet...")


    ## 9
    if file == 0:
        try:
            cat = TDSCatalog('http://nomads.ncdc.noaa.gov/thredds/catalog/rap130/'+str(time_start.year)+'0'+str(time_start.month)+'/'+str(time_start.year)+'0'+str(time_start.month)+str(time_start.day)+'/catalog.html?dataset=rap130/'+str(time_start.year)+'0'+str(time_start.month)+'/'+str(time_start.year)+'0'+str(time_start.month)+str(time_start.day)+'/rap_130_'+str(time_start.year)+'0'+str(time_start.month)+str(time_start.day)+'_'+str(time_start.hour)+'00_001.grb2')
            file = 1
            print("We have the file we need!")
        except:
            file = 0
            print("No file yet...")

    ## 10
    if file == 0:
        try:
            cat = TDSCatalog('http://nomads.ncdc.noaa.gov/thredds/catalog/rap130/'+str(year)+str(month)+'/'+str(year)+str(month)+str(day)+'/catalog.html?dataset=rap130/'+str(year)+str(month)+'/'+str(year)+str(month)+str(day)+'/rap_130_'+str(year)+str(month)+str(day)+'_'+str(UTC)+'_000.grb2')
            file = 1
            print("We have the file we need!")
        except:
            file = 0
            print("No file yet...")

    ## 11
    if file == 0:
        try:
            cat = TDSCatalog('https://www.ncei.noaa.gov/thredds/catalog/rap130anl/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/catalog.html?dataset=rap130anl/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/rap_130_'+str(time_start.year)+str(month)+str(day)+'_'+str(hour)+'00_000.grb2')
            file = 1
            print("We have the file we need!")
        except:
            file = 0
            print("No file yet...")

    ## 12
    if file == 0:
        try:
            cat = TDSCatalog('https://www.ncdc.noaa.gov/thredds/catalog/model-rap252anl/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/catalog.html?dataset=rap252anl/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/rap_252_'+str(time_start.year)+str(month)+str(day)+'_'+str(hour)+'00_000.grb2')
            file = 1
            print("We have the file we need!")
        except:
            file = 0
            print("No file yet...")

    ## 13
    #if file == 0:
    #    try:
    #        cat = TDSCatalog('https://www.ncei.noaa.gov/thredds/catalog/model-rap252anl-old/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/catalog.html?dataset=rap252anl-old/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/rap_252_'+str(time_start.year)+str(month)+str(day)+'_'+str(hour)+'00_000.grb2') 
    #        file = 1
    #        print("We have the file we need!")
    #    except:
    #        file = 0
    #        print("No file yet...")

    ## 14
    if file == 0:
        try:
            cat = TDSCatalog('https://www.ncei.noaa.gov/thredds/catalog/model-ruc130anl/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/catalog.html?dataset=ruc130anl/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/ruc2anl_130_'+str(time_start.year)+str(month)+str(day)+'_'+str(hour)+'00_000.grb2') 
            file = 1
            print("We have the file we need!")
        except:
            file = 0
            print("No file yet...")

    ## 15
    if file == 0:
        try:
            cat = TDSCatalog('https://www.ncei.noaa.gov/thredds/catalog/model-ruc252anl/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/catalog.html?dataset=ruc252anl/'+str(time_start.year)+str(month)+'/'+str(time_start.year)+str(month)+str(day)+'/ruc2anl_252_'+str(time_start.year)+str(month)+str(day)+'_'+str(hour)+'00_000.grb2')
            file = 1
            print("We have the file we need!")
        except:
            file = 0
            print("We didn't get any file!")
        #cat = TDSCatalog('http://nomads.ncdc.noaa.gov/thredds/catalog/rap130/'+str(year)+str(month)+'/'+str(year)+str(month)+str(day)+'/catalog.html?dataset=rap130/'+str(year)+str(month)+'/'+str(year)+str(month)+str(day)+'/rap_130_'+str(year)+str(month)+str(day)+'_'+str(UTC)+'_000.grb2')
    latest_ds = list(cat.datasets.values())[0]
    print(latest_ds.access_urls)
    #ncss = NCSS(latest_ds.access_urls['NetcdfServer'])
    ncss = NCSS(latest_ds.access_urls['NetcdfSubset'])
    query = ncss.query()
    query.variables('Convective_available_potential_energy_surface').variables('u-component_of_wind_isobaric').variables('v-component_of_wind_isobaric').variables('Pressure_surface').variables('Geopotential_height_isobaric').variables('u-component_of_wind_height_above_ground').variables('v-component_of_wind_height_above_ground').variables('MSLP_MAPS_System_Reduction_msl').variables('Geopotential_height_zeroDegC_isotherm')
    query.add_lonlat().lonlat_box(radar_lon-1.8, radar_lon+1.8, radar_lat-1.8, radar_lat+1.8)
    data1 = ncss.get_data(query)
    dtime = data1.variables['Convective_available_potential_energy_surface'].dimensions[0]
    dlat = data1.variables['Convective_available_potential_energy_surface'].dimensions[1]
    dlev = data1.variables['Geopotential_height_isobaric'].dimensions[1]
    dlon = data1.variables['Convective_available_potential_energy_surface'].dimensions[2]
    SFCP = np.asarray(data1.variables['Pressure_surface'][:]/100.) * units('hPa')
    hgt = np.asarray(data1.variables['Geopotential_height_isobaric'][:]) * units('meter')
    #sfc_hgt = np.asarray(data1.variables['Geopotential_height_surface'][:]) * units('meter')
    uwnd = np.asarray(data1.variables['u-component_of_wind_isobaric'][:]) * units('m/s')
    vwnd = np.asarray(data1.variables['v-component_of_wind_isobaric'][:]) * units('m/s')
    usfc = np.asarray(data1.variables['u-component_of_wind_height_above_ground'][:]) * units('m/s')
    vsfc = np.asarray(data1.variables['v-component_of_wind_height_above_ground'][:]) * units('m/s')
    MSLP = np.asarray((data1.variables['MSLP_MAPS_System_Reduction_msl'][:]/100.)) * units('hPa')
    Z0C = np.asarray(data1.variables['Geopotential_height_zeroDegC_isotherm']) * units('m')
    # Get the dimension data
    lats_r = data1.variables[dlat][:]
    lons_r= data1.variables[dlon][:]
    lev = (data1.variables[dlev][:]/100.) * units('hPa')
    # Set up our array of latitude and longitude values and transform to 
    # the desired projection.
    crs = ccrs.PlateCarree()
    crlons, crlats = np.meshgrid(lons_r[:]*1000, lats_r[:]*1000)
    trlatlons = crs.transform_points(ccrs.LambertConformal(central_longitude=265, central_latitude=25, standard_parallels=(25.,25.)),crlons,crlats)
    trlons = trlatlons[:,:,0]
    trlats = trlatlons[:,:,1]
    
    #Next step: get sfc-400mb shear across region of interest to dynamically set the FFD angle.
    u400 = uwnd[0,12,:,:]
    v400 = vwnd[0,12,:,:]
    u10m = usfc[0,0,:,:]
    v10m = vsfc[0,0,:,:]

    us400 = u400-u10m
    vs400 = v400-v10m
    shr_dir = wind_direction(us400, vs400)

    
    return trlons, trlats, shr_dir, Z0C