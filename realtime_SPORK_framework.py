import matplotlib.pyplot as plt
import pyart
import numpy as np
import numpy.ma as ma
from metpy.units import check_units, concatenate, units
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from siphon.radarserver import RadarServer
#rs = RadarServer('http://thredds-aws.unidata.ucar.edu/thredds/radarServer/nexrad/level2/S3/')
#rs = RadarServer('http://thredds.ucar.edu/thredds/radarServer/nexrad/level2/IDD/')
from datetime import datetime, timedelta
from siphon.cdmr import Dataset
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from metpy.units import check_units, concatenate, units
from shapely.geometry import polygon as sp
import pyproj 
import shapely.ops as ops
from shapely.ops import transform
from shapely.geometry.polygon import Polygon
from functools import partial
from shapely import geometry
import netCDF4
from scipy import ndimage as ndi
#from skimage.feature import peak_local_max
#from skimage import data, img_as_float
from pyproj import Geod
from metpy.calc import wind_direction, wind_speed, wind_components
import matplotlib.lines as mlines
import pandas as pd
import scipy.stats as stats
import csv
import pickle
from sklearn.ensemble import RandomForestClassifier
import nexradaws
import os
#from grid_section import gridding
#from grid_section_spin import gridding_spin
from grid_section_SPORK_stamps import gridding_spin_fast
from kdp_section import kdp_genesis
from gradient_section import grad_mask
#from ungridded_section import quality_control
from ungridded_section_spin import quality_control_spin
#from stormid_section import storm_objects
from stormid_section_xtrap import storm_objects_new
from zdr_arc_trendss import zdrarc
from hail_section import hail_objects
from zhh_section import zhh_objects
from kdpfoot_section import kdp_objects
from zdr_col_section import zdrcol
from rotation_stuff import get_rotation
from rotation_matching_qc import rot_storm_matcher_qc
from realtime_radar_SPORK import get_latest_scan_SPORK
from RAPdataSPORK import get_RAP_data_realtime
from metpy.interpolate import interpolate_to_points
import threading

def spork_realtime(zdrlev, kdplev, REFlev, REFlev1, big_storm, calibration, station, Bunkers_m, track_dis=10, GR_mins=5.0):
    #Set vector perpendicular to FFD Z gradient
    #storm_relative_dir = storm_relative_dir
    #Set storm motion
    Zint = 20
    Bunkers_m = Bunkers_m
    #Set ZDR Threshold for outlining arcs
    zdrlev = [zdrlev]
    #Set KDP Threshold for finding KDP feet
    kdplev = [kdplev]
    #Set reflectivity thresholds for storm tracking algorithm
    REFlev = [REFlev]
    REFlev1 = [REFlev1]
    #Set storm size threshold that triggers subdivision of big storms
    big_storm = big_storm #km^2
    Z0C = 3200
    Outer_r = 30 #km
    Inner_r = 6 #km
    #Set trigger to ignore strangely-formatted files right before 00Z
    #Pre-SAILS #: 17
    #SAILS #: 25
    zero_z_trigger = 300
    storm_to_track = -999
    zdr_outlines = []
    #Here, set the initial time of the archived radar loop you want.
    #Our specified time
#     dt = datetime(year,month, day, hour, start_min)
    station = station
    dt = datetime.utcnow()
#     end_dt = dt + timedelta(hours=duration)

#     #Set up nexrad interface
#     conn = nexradaws.NexradAwsInterface()
#     scans = conn.get_avail_scans_in_range(dt,end_dt,station)
#     results = conn.download(scans, 'RadarFolder')

    #Setting counters for figures and Pandas indices
    f = 27
    n = 1
    storm_index = 0
    scan_index = 0
    tracking_index = 0
    #Create geod object for later distance and area calculations
    g = Geod(ellps='sphere')
    #Open the placefile
    f = open("SPORKRF1"+station+str(dt.year)+str(dt.month)+str(dt.day)+str(dt.hour)+str(dt.minute)+"_Placefile.txt", "w+")
    f.write("Title: SPORK Placefile \n")
    #f.write("Refresh: 8 \n \n")
    f.write("RefreshSeconds: 10 \n \n")
    f.flush()
    
    #Load ML algorithm
    forest_loaded = pickle.load(open('NewDataRandomForest.pkl', 'rb'))
    forest_loaded_col = pickle.load(open('NewDataRandomForest_COLUMNS.pkl', 'rb'))
    forest_loaded_mesos = pickle.load(open('NewDataRandomForest_MESOS.pkl', 'rb'))

    #Actual algorithm code starts here
    #Create a list for the lists of arc outlines
    zdr_out_list = []
    tracks_dataframe = []
    
    #Section of code to set up continuous loop of 
    #getting new radar files
    
    WAIT_TIME_SECONDS = 30

    ticker = threading.Event()
    f_l = 0
    goodscan_flag = 0
    while not ticker.wait(WAIT_TIME_SECONDS):
    #     print('time')
    #     print(time1)
        try:
            radar4, time_start = get_latest_scan_SPORK(station)
        except:
            continue
        print(time_start.hour)
        print(time_start.minute)
        print(time_start.second)
        if f_l == 0:
            if radar4.azimuth['data'].shape[0] > 700:
                ts_h1 = time_start.hour
                ts_m1 = time_start.minute
                ts_s1 = time_start.second
                goodscan_flag = 1
            else:
                goodscan_flag = 0
        else:
            ts_h = time_start.hour
            ts_m = time_start.minute
            ts_s = time_start.second
            if ts_h == ts_h1 and ts_m == ts_m1 and ts_s == ts_s1:
                print('duplicate scan')
                goodscan_flag = 0
            else:
                print('new scan')
                if radar4.azimuth['data'].shape[0] > 700:
                    ts_h1 = time_start.hour
                    ts_m1 = time_start.minute
                    ts_s1 = time_start.second
                    goodscan_flag = 1
                    ts_h1 = ts_h
                    ts_m1 = ts_m
                    ts_s1 = ts_s
                else:
                    goodscan_flag = 0
    
    
    
    
    
        if goodscan_flag==1:
        #Local file option:
            #Loop over all files in the dataset and pull out each 0.5 degree tilt for analysis
#             try:
#                 radar1 = scan.open_pyart()
#             except:
#                 print('bad radar file')
#                 continue
            #Local file option
            print('File Reading')
            #Make sure the file isn't a strange format
            if radar4.nsweeps > zero_z_trigger:
                continue

            #Calling ungridded_section; Pulling apart radar sweeps and creating ungridded data arrays
            [radar,radar_v,n,range_2d,last_height,rlons_h,rlats_h,ungrid_lons,ungrid_lats] = quality_control_spin(radar4,n,calibration)

            time_start = netCDF4.num2date(radar.time['data'][0], radar.time['units'])
            object_number=0.0
            month = time_start.month
            if month < 10:
                month = '0'+str(month)
            hour = time_start.hour
            if hour < 10:
                hour = '0'+str(hour)
            minute = time_start.minute
            if minute < 10:
                minute = '0'+str(minute)
            day = time_start.day
            if day < 10:
                day = '0'+str(day)
            time_beg = time_start - timedelta(minutes=0.1)
            time_end = time_start + timedelta(minutes=GR_mins)
            sec_beg = time_beg.second
            sec_end = time_end.second
            min_beg = time_beg.minute
            min_end = time_end.minute
            h_beg = time_beg.hour
            h_end = time_end.hour
            d_beg = time_beg.day
            d_end = time_end.day
            if sec_beg < 10:
                sec_beg = '0'+str(sec_beg)
            if sec_end < 10:
                sec_end = '0'+str(sec_end)
            if min_beg < 10:
                min_beg = '0'+str(min_beg)
            if min_end < 10:
                min_end = '0'+str(min_end)
            if h_beg < 10:
                h_beg = '0'+str(h_beg)
            if h_end < 10:
                h_end = '0'+str(h_end)
            if d_beg < 10:
                d_beg = '0'+str(d_beg)
            if d_end < 10:
                d_end = '0'+str(d_end)

            #Calling kdp_section; Using NWS method, creating ungridded, smoothed KDP field
            kdp_nwsdict = kdp_genesis(radar)

            #Add field to radar
            radar.add_field('KDP', kdp_nwsdict)
            kdp_ungridded_nws = radar.fields['KDP']['data']


            #Calling grid_section; Now let's grid the data on a ~250 m x 250 m grid
            [Zint_a,REF,KDP,CC,CC_c,CCall,ZDRmasked1,ZDRrmasked1_a,REFmasked,REFrmasked_a,KDPmasked,KDPrmasked_a,rlons,rlats,rlons_2d,rlats_2d,cenlat,cenlon, VEL, REFall, ZDRall, KDPall] = gridding_spin_fast(radar,radar_v,Z0C)
            
            #Get RAP data for FFD angles
            
            if f_l == 0:
                ffd_lons, ffd_lats, ffd_Rangles, Z0C_arl1 = get_RAP_data_realtime(cenlon, cenlat)
                
                radar_z = radar.altitude['data']
                Z0C_arl = Z0C_arl1 - radar_z*units('m')

                ffd_grid = interpolate_to_points(np.asarray([np.ndarray.flatten(ffd_lons), np.ndarray.flatten(ffd_lats)]).T,
                                                np.ndarray.flatten(ffd_Rangles.magnitude-86),
                                                np.asarray([np.ndarray.flatten(rlons_2d), np.ndarray.flatten(rlats_2d)]).T)
                ffd_sgrid = ffd_grid.reshape(rlons_2d.shape)
                storm_relative_dir = ffd_sgrid
                
                Z0C_grid = interpolate_to_points(np.asarray([np.ndarray.flatten(ffd_lons), np.ndarray.flatten(ffd_lats)]).T,
                                  np.ndarray.flatten(Z0C_arl[0,:,:].magnitude),
                                  np.asarray([np.ndarray.flatten(rlons_2d), np.ndarray.flatten(rlats_2d)]).T)
                Z0C_sgrid = Z0C_grid.reshape(rlons_2d.shape)
                ZINT_sgrid = np.round(Z0C_sgrid/250)
                
                time_RAPscan = time_start
            else:
                diff_seconds = (time_start-time_RAPscan).seconds
                if diff_seconds > 3600:
                    ffd_lons, ffd_lats, ffd_Rangles, Z0C_arl1 = get_RAP_data_realtime(cenlon, cenlat)
                    
                    radar_z = radar.altitude['data']
                    Z0C_arl = Z0C_arl1 - radar_z*units('m')

                    ffd_grid = interpolate_to_points(np.asarray([np.ndarray.flatten(ffd_lons), np.ndarray.flatten(ffd_lats)]).T,
                                                    np.ndarray.flatten(ffd_Rangles.magnitude-86),
                                                    np.asarray([np.ndarray.flatten(rlons_2d), np.ndarray.flatten(rlats_2d)]).T)
                    ffd_sgrid = ffd_grid.reshape(rlons_2d.shape)
                    storm_relative_dir = ffd_sgrid
                    
                    Z0C_grid = interpolate_to_points(np.asarray([np.ndarray.flatten(ffd_lons), np.ndarray.flatten(ffd_lats)]).T,
                                  np.ndarray.flatten(Z0C_arl[0,:,:].magnitude),
                                  np.asarray([np.ndarray.flatten(rlons_2d), np.ndarray.flatten(rlats_2d)]).T)
                    Z0C_sgrid = Z0C_grid.reshape(rlons_2d.shape)
                    ZINT_sgrid = np.round(Z0C_sgrid/250)
                    
                    time_RAPscan = time_start
                else:
                    print('not getting new RAP data')
                    
            #Get ZDR 1km above the freezing level
            #ZDR_1km_g = np.copy(ZDRall)
            REF_1km_g = np.copy(REFall)
            KDP_1km_g = np.copy(KDPall)
            CC_1km_g = np.copy(CCall)
            for i in range(REFall.shape[0]):
#                 ZDR_1km_g1 = ZDR_1km_g[i,:,:]
#                 ZDR_1km_g1[i!=(ZINT_sgrid+4)] = 0
#                 ZDR_1km_g[i,:,:] = ZDR_1km_g1
                
                REF_1km_g1 = REF_1km_g[i,:,:]
                REF_1km_g1[i!=(ZINT_sgrid+4)] = 0
                REF_1km_g[i,:,:] = REF_1km_g1
                
                KDP_1km_g1 = KDP_1km_g[i,:,:]
                KDP_1km_g1[i!=(ZINT_sgrid+4)] = 0
                KDP_1km_g[i,:,:] = KDP_1km_g1
                
                CC_1km_g1 = CC_1km_g[i,:,:]
                CC_1km_g1[i!=(ZINT_sgrid+4)] = 0
                CC_1km_g[i,:,:] = CC_1km_g1
                
#             ZDR_1km = np.max(ZDR_1km_g, axis=0)
            REFrmasked = np.max(REF_1km_g, axis=0)
            KDPrmasked = np.max(KDP_1km_g, axis=0)
            CC_c = np.max(CC_1km_g, axis=0)
            
            #Get TRENDSS detections from the 1km ARL data
            #Calculating TRENDSS ZDR anomalies
            NORM_ZDR = np.copy(ZDRmasked1)
            NORM_ZDR = ma.masked_where(REF < 20, NORM_ZDR)
            refl_bins = np.arange(20, 60, 5)
            for refl_b in refl_bins:
                #print(refl_b)
                #print(REFW[0][(refl_b < REFW[0]) & (REFW[0] < (refl_b+5))])
                if np.shape(REF[(refl_b < REF) & (REF < refl_b + 5)])[0] > 20:
                    #print(np.shape(REFW[i][(refl_b < REFW[i]) & (REFW[i] < refl_b + 5)]))
                    #print(np.min(REFW[0][(refl_b < REFW[0]) & (REFW[0] < refl_b + 5)]))
                    #print(np.nanmean(ZDRW_C[i][(refl_b < REFW[i]) & (REFW[i] < refl_b + 5)]))
                    mu = np.nanmean(ZDRmasked1[(refl_b < REF) & (REF < refl_b + 5)])
                    sigma = np.nanstd(ZDRmasked1[(refl_b < REF) & (REF < refl_b + 5)])
                    #print(mu, sigma)
                    points_c = ZDRmasked1[(refl_b < REF) & (REF < refl_b + 5)]
                    points_n1 = points_c - mu
                    points_n = points_n1 / sigma
                    NORM_ZDR[(refl_b < REF) & (REF < refl_b + 5)] = np.asarray(points_n)

                else:
                    print('not enough points')
                    mu = 10 ** ((-2.6857*(10**-4)*((refl_b+2.5)**2)) + (0.04892 * (refl_b + 2.5)) - 1.4287)
                    sigma = 0.5
                    print(mu, sigma)
                    points_c = ZDRmasked1[(refl_b < REF) & (REF < refl_b + 5)]
                    points_n1 = points_c - mu
                    points_n = points_n1 / sigma
                    NORM_ZDR[(refl_b < REF) & (REF < refl_b + 5)] = np.asarray(points_n)

            #Calling gradient_section; Determining gradient direction and masking some Zhh and Zdr grid fields
            [grad_mag,grad_ffd,ZDRmasked,ZDRallmasked,ZDRrmasked_b] = grad_mask(Zint,REFmasked,REF,storm_relative_dir,NORM_ZDR,ZDRrmasked1_a,CC,CCall)
            
            
            
            #Get ZDR 1km above the freezing level
            ZDR_1km_g = np.copy(ZDRallmasked)
            for i in range(ZDRallmasked.shape[0]):
                ZDR_1km_g1 = ZDR_1km_g[i,:,:]
                ZDR_1km_g1[i!=(ZINT_sgrid+4)] = 0
                ZDR_1km_g[i,:,:] = ZDR_1km_g1
            
            ZDRrmasked = np.max(ZDR_1km_g, axis=0)
            

            if np.max(VEL) > 0:
                #Calculate rotation from the velocity field
                [az_masked, shear_maxes1, shear_maxes15, shear_maxes2, shear_maxes25, shear_lats1, shear_lats15, shear_lats2, shear_lats25, shear_lons1, shear_lons15, shear_lons2, shear_lons25] = get_rotation(VEL, REFall, rlons_2d, rlats_2d, bin_size=7)

            else:
                az_masked = []
                shear_maxes1 = []
                shear_maxes15 = []
                shear_maxes2 = []
                shear_maxes25 = []
                shear_lats1 = []
                shear_lats15 = []
                shear_lats2 = []
                shear_lats25 = []
                shear_lons1 = []
                shear_lons15 = []
                shear_lons2 = []
                shear_lons25 = []


            #Let's create the ZDR column depth field as in Snyder et al. (2015)
            ZDR_count = np.copy(ZDRallmasked)
            ZDR_count[ZDR_count > 1.0] = 1
            ZDR_count[ZDR_count < 1.0] = 0

#             ZDR_sum_stuff = np.zeros((ZDR_count.shape[1], ZDR_count.shape[2]))
#             ZDR_top = np.copy(ZDR_count[(Zint-4):,:,:])
#             for i in range(ZDR_top.shape[0]):
#                 ZDR_new_sum = ZDR_sum_stuff + ZDR_top[i,:,:]
#                 ZDR_same = np.where(ZDR_new_sum-ZDR_sum_stuff==0)
#                 ZDR_top[i:,ZDR_same[0],ZDR_same[1]] = 0
#                 ZDR_sum_stuff = ZDR_new_sum

            ZDR_olaf = np.copy(ZDR_count)
            #Starting from the bottm up, set all count values below the freezing level to 0
            for i in range(ZDR_olaf.shape[0]):
                #print(i)
                ZDR_pancake = ZDR_olaf[i,:,:]
                ZDR_pancake[ZINT_sgrid>i] = 0.0
                #print(np.max(ZDR_pancake))
                ZDR_olaf[i,:,:] = ZDR_pancake
                
            ZDR_sum_stuff = np.zeros((ZDR_count.shape[1], ZDR_count.shape[2]))
            ZDR_top_new = np.copy(ZDR_olaf)
            for i in range(ZDR_top_new.shape[0]):
                ZDR_levthing = np.zeros((ZDR_count.shape[1], ZDR_count.shape[2]))
                ZDR_new_sum1 = ZDR_sum_stuff + ZDR_top_new[i,:,:]
                ZDR_levthing[ZINT_sgrid>i] = 1
                #print(np.max(ZDR_levthing))
                ZDR_same_new = np.where(((ZDR_new_sum1-ZDR_sum_stuff+ZDR_levthing))==0)
                ZDR_top_new[i:,ZDR_same_new[0],ZDR_same_new[1]] = 0
                ZDR_sum_stuff = ZDR_new_sum1
            #print(np.nanmax(ZDR_sum_new))

            #Let's create a field for inferred hail
            REF_Hail = np.copy(REFmasked)
            REF_Hail1 = ma.masked_where(ZDRmasked1 > 1.0, REF_Hail)
            REF_Hail2 = ma.masked_where(CC > 1.0, REF_Hail1)
            REF_Hail2 = ma.filled(REF_Hail2, fill_value = 1)

            #Let's set up the map projection!
            crs = ccrs.LambertConformal(central_longitude=-100.0, central_latitude=45.0)

            #Set up our array of latitude and longitude values and transform our data to the desired projection.
            tlatlons = crs.transform_points(ccrs.LambertConformal(central_longitude=265, central_latitude=25, standard_parallels=(25.,25.)),rlons[0,:,:],rlats[0,:,:])
            tlons = tlatlons[:,:,0]
            tlats = tlatlons[:,:,1]

            #Limit the extent of the map area, must convert to proper coords.
            LL = (cenlon-1.0,cenlat-1.0,ccrs.PlateCarree())
            UR = (cenlon+1.0,cenlat+1.0,ccrs.PlateCarree())
            print(LL)

            #Get data to plot state and province boundaries
            states_provinces = cfeature.NaturalEarthFeature(
                    category='cultural',
                    name='admin_1_states_provinces_lakes',
                    scale='50m',
                    facecolor='none')
            #Make sure these shapefiles are in the same directory as the script
            fname = 'cb_2016_us_county_20m/cb_2016_us_county_20m.shp'
            fname2 = 'cb_2016_us_state_20m/cb_2016_us_state_20m.shp'
            counties = ShapelyFeature(Reader(fname).geometries(),ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black')
            states = ShapelyFeature(Reader(fname2).geometries(),ccrs.PlateCarree(), facecolor = 'none', edgecolor = 'black')

            #Create a figure and plot up the initial data and contours for the algorithm
            fig=plt.figure(n,figsize=(30.,25.))
            ax = plt.subplot(111,projection=ccrs.PlateCarree())
            ax.coastlines('50m',edgecolor='black',linewidth=0.75)
            ax.add_feature(counties, edgecolor = 'black', linewidth = 0.5)
            ax.add_feature(states, edgecolor = 'black', linewidth = 1.5)
            ax.set_extent([LL[0],UR[0],LL[1],UR[1]])
            REFlevels = np.arange(20,73,2)
            depth_levels= np.arange(0.01,23,1)

            #Options for Z backgrounds/contours
            #refp = ax.pcolormesh(ungrid_lons, ungrid_lats, ref_c, cmap=plt.cm.gist_ncar, vmin = 10, vmax = 73)
            #refp = ax.pcolormesh(ungrid_lons, ungrid_lats, ref_ungridded_base, cmap='HomeyerRainbow', vmin = 10, vmax = 73)
            #refp = ax.pcolormesh(rlons_2d, rlats_2d, REFrmasked, cmap=pyart.graph.cm_colorblind.HomeyerRainbow, vmin = 10, vmax = 73)
            refp2 = ax.contour(rlons_2d, rlats_2d, REFmasked, [40], colors='grey', linewidths=5, zorder=1)
            #refp3 = ax.contour(rlons_2d, rlats_2d, REFmasked, [45], color='r')
            #plt.contourf(rlons_2d, rlats_2d, ZDR_sum_stuff, depth_levels, cmap=plt.cm.viridis)

            #Option to have a ZDR background instead of Z:
            #zdrp = ax.pcolormesh(ungrid_lons, ungrid_lats, zdr_c, cmap=plt.cm.nipy_spectral, vmin = -2, vmax = 6)

            #Storm tracking algorithm starts here
            #Reflectivity smoothed for storm tracker
            smoothed_ref = ndi.gaussian_filter(REFmasked, sigma = 3, order = 0)
            #1st Z contour plotted
            refc = ax.contour(rlons[0,:,:],rlats[0,:,:],smoothed_ref,REFlev, colors= 'grey', alpha=.01)

            #Set up projection for area calculations
            proj = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'),
                       pyproj.Proj(init='epsg:3857'))

            #Main part of storm tracking algorithm starts by looping through all contours looking for Z centroids
            #This method for breaking contours into polygons based on this stack overflow tutorial:
            #https://gis.stackexchange.com/questions/99917/converting-matplotlib-contour-objects-to-shapely-objects
            #Calling stormid_section
            [storm_ids,max_lons_c,max_lats_c,ref_areas,storm_index, storm_speeds, storm_dirs] = storm_objects_new(refc,proj,REFlev,REFlev1,big_storm,smoothed_ref,ax,rlons,rlats,storm_index,tracking_index,scan_index,tracks_dataframe, track_dis, time_start)
            
            #Interpolate FFD angles to the storm locations
            ffd_storms = interpolate_to_points(np.asarray([np.ndarray.flatten(rlons_2d), np.ndarray.flatten(rlats_2d)]).T,
                                                    np.ndarray.flatten(ffd_sgrid),
                                                    np.asarray([np.ndarray.flatten(max_lons_c), np.ndarray.flatten(max_lats_c)]).T)

            #Setup tracking index for storm of interest
            tracking_ind=np.where(np.asarray(storm_ids)==storm_to_track)[0]
            max_lons_c = np.asarray(max_lons_c)
            max_lats_c = np.asarray(max_lats_c)
            ref_areas = np.asarray(ref_areas)
            #Create the ZDR and KDP contours which will later be broken into polygons
            if np.max(ZDRmasked) > zdrlev:
                zdrc = ax.contour(rlons[0,:,:],rlats[0,:,:],ZDRmasked,zdrlev,linewidths = 2, colors='purple', alpha = .1)
            else:
                zdrc=[]
            if np.max(ZDRrmasked) > 1.0:
                zdrrc = ax.contour(rlons[0,:,:],rlats[0,:,:],ZDRrmasked,[1.0],linewidths = 4, colors='cyan', alpha = 0.1)
            else:
                zdrrc=[]
            if np.max(KDPmasked) > kdplev:
                kdpc = ax.contour(rlons[0,:,:],rlats[0,:,:],KDPmasked,kdplev,linewidths = 2, colors='green', alpha = 0.01)
            else:
                kdpc=[]
            if np.max(REF_Hail2) > 50.0:
                hailc = ax.contour(rlons[0,:,:],rlats[0,:,:],REF_Hail2,[50],linewidths = 4, colors='pink', alpha = 0.01)
            else:
                hailc=[]
            if np.max(REFmasked) > 35.0:
                zhhc = ax.contour(rlons[0,:,:],rlats[0,:,:],REFmasked,[35.0],linewidths = 3,colors='orange', alpha = 0.01)
            else:
                zhhc=[]
            plt.contour(ungrid_lons, ungrid_lats, range_2d, [73000], linewidths=7, colors='r')
            plt.contour(rlons_h, rlats_h, last_height, [Z0C], linewidths=7, colors='g')
            plt.savefig('testfig.png')
            print('Testfig Saved')

            if len(max_lons_c) > 0:
                #Calling zdr_arc_section; Create ZDR arc objects using a similar method as employed in making the storm objects
                [zdr_storm_lon,zdr_storm_lat,zdr_dist,zdr_forw,zdr_back,zdr_areas,zdr_centroid_lon,zdr_centroid_lat,zdr_mean,zdr_cc_mean,zdr_max,zdr_masks,zdr_outlines,ax,f] = zdrarc(zdrc,ZDRmasked,CC,REF,grad_ffd,grad_mag,KDP,forest_loaded,ax,f,time_start,month,d_beg,h_beg,min_beg,sec_beg,d_end,h_end,min_end,sec_end,rlons,rlats,max_lons_c,max_lats_c,zdrlev,proj,ffd_storms,Outer_r,Inner_r,tracking_ind)


                #Calling hail_section; Identify Hail core objects in a similar way to the ZDR arc objects
                [hail_areas,hail_centroid_lon,hail_centroid_lat,hail_storm_lon,hail_storm_lat,ax,f] = hail_objects(hailc,REF_Hail2,ax,f,time_start,month,d_beg,h_beg,min_beg,sec_beg,d_end,h_end,min_end,sec_end,rlons,rlats,max_lons_c,max_lats_c,proj)


                #Calling zhh_section; Identify 35dBz storm area in a similar way to the ZDR arc objects
                [zhh_areas,zhh_centroid_lon,zhh_centroid_lat,zhh_storm_lon,zhh_storm_lat,zhh_max,zhh_core_avg] = zhh_objects(zhhc,REFmasked,rlons,rlats,max_lons_c,max_lats_c,proj)


                #Calling kdpfoot_section; Identify KDP foot objects in a similar way to the ZDR arc objects
                [kdp_areas,kdp_centroid_lon,kdp_centroid_lat,kdp_storm_lon,kdp_storm_lat,kdp_max,ax,f] = kdp_objects(kdpc,KDPmasked,ax,f,time_start,month,d_beg,h_beg,min_beg,sec_beg,d_end,h_end,min_end,sec_end,rlons,rlats,max_lons_c,max_lats_c,kdplev,proj)


                #Calling zdr_col_section; Identify ZDR columns in a similar way to the ZDR arc objects
                [col_areas,col_maxdepths,col_depths,col_centroid_lon,col_centroid_lat,col_storm_lon,col_storm_lat,ax,col_masks,f] = zdrcol(zdrrc,ZDRrmasked,CC_c,REFrmasked,grad_ffd,grad_mag,KDP,ZDR_sum_stuff,KDPrmasked,depth_levels,forest_loaded_col,ax,f,time_start,month,d_beg,h_beg,min_beg,sec_beg,d_end,h_end,min_end,sec_end,rlons,rlats,max_lons_c,max_lats_c,ref_areas,proj,ffd_storms,tracking_ind,object_number)

                #Getting 1km rotation objects
                [rot_mag1, rot_lat1, rot_lon1, rot_storm_lon_1, rot_storm_lat_1, azarea1] = rot_storm_matcher_qc(shear_maxes1,shear_lats1, shear_lons1,ax,f,time_start,month,d_beg,h_beg,min_beg,sec_beg,d_end,h_end,min_end,sec_end,rlons,rlats,max_lons_c,max_lats_c,proj,tracking_ind, rlons_2d, rlats_2d, REFall, KDPall, CCall, grad_ffd, grad_mag, ZDR_sum_stuff, az_masked, ffd_storms, station, dt, forest_loaded_mesos, 4)

                #Getting 1.5km rotation objects
                [rot_mag15, rot_lat15, rot_lon15, rot_storm_lon_15, rot_storm_lat_15, azarea15] = rot_storm_matcher_qc(shear_maxes15,shear_lats15, shear_lons15,ax,f,time_start,month,d_beg,h_beg,min_beg,sec_beg,d_end,h_end,min_end,sec_end,rlons,rlats,max_lons_c,max_lats_c,proj,tracking_ind, rlons_2d, rlats_2d, REFall, KDPall, CCall, grad_ffd, grad_mag, ZDR_sum_stuff, az_masked, ffd_storms, station, dt, forest_loaded_mesos, 12)

                #Getting 2km rotation objects
                [rot_mag2, rot_lat2, rot_lon2, rot_storm_lon_2, rot_storm_lat_2, azarea2] = rot_storm_matcher_qc(shear_maxes2,shear_lats2, shear_lons2,ax,f,time_start,month,d_beg,h_beg,min_beg,sec_beg,d_end,h_end,min_end,sec_end,rlons,rlats,max_lons_c,max_lats_c,proj,tracking_ind, rlons_2d, rlats_2d, REFall, KDPall, CCall, grad_ffd, grad_mag, ZDR_sum_stuff, az_masked, ffd_storms, station, dt, forest_loaded_mesos, 20)

                #Getting 2.5km rotation objects
                [rot_mag25, rot_lat25, rot_lon25, rot_storm_lon_25, rot_storm_lat_25, azarea25] = rot_storm_matcher_qc(shear_maxes25,shear_lats25, shear_lons25,ax,f,time_start,month,d_beg,h_beg,min_beg,sec_beg,d_end,h_end,min_end,sec_end,rlons,rlats,max_lons_c,max_lats_c,proj,tracking_ind, rlons_2d, rlats_2d, REFall, KDPall, CCall, grad_ffd, grad_mag, ZDR_sum_stuff, az_masked, ffd_storms, station, dt, forest_loaded_mesos, 28)


                #Consolidating the arc objects associated with each storm:
                zdr_areas_arr = np.zeros((len(zdr_areas)))
                zdr_max_arr = np.zeros((len(zdr_max)))
                zdr_mean_arr = np.zeros((len(zdr_mean)))                    
                for i in range(len(zdr_areas)):
                    zdr_areas_arr[i] = zdr_areas[i].magnitude
                    zdr_max_arr[i] = zdr_max[i]
                    zdr_mean_arr[i] = zdr_mean[i]
                zdr_centroid_lons = np.asarray(zdr_centroid_lon)
                zdr_centroid_lats = np.asarray(zdr_centroid_lat)
                zdr_con_areas = []
                zdr_con_maxes = []
                zdr_con_means = []
                zdr_con_centroid_lon = []
                zdr_con_centroid_lat = []
                zdr_con_max_lon = []
                zdr_con_max_lat = []
                zdr_con_storm_lon = []
                zdr_con_storm_lat = []
                zdr_con_masks = []
                zdr_con_dev = []
                zdr_con_10max = []
                zdr_con_mode = []
                zdr_con_median = []
                zdr_masks = np.asarray(zdr_masks)

                #Consolidate KDP objects as well
                kdp_areas_arr = np.zeros((len(kdp_areas)))
                kdp_max_arr = np.zeros((len(kdp_max)))
                for i in range(len(kdp_areas)):
                    kdp_areas_arr[i] = kdp_areas[i].magnitude
                    kdp_max_arr[i] = kdp_max[i]
                kdp_centroid_lons = np.asarray(kdp_centroid_lon)
                kdp_centroid_lats = np.asarray(kdp_centroid_lat)
                kdp_con_areas = []
                kdp_con_maxes = []
                kdp_con_centroid_lon = []
                kdp_con_centroid_lat = []
                kdp_con_max_lon = []
                kdp_con_max_lat = []
                kdp_con_storm_lon = []
                kdp_con_storm_lat = []

                #Consolidate Hail objects as well
                hail_areas_arr = np.zeros((len(hail_areas)))
                for i in range(len(hail_areas)):
                    hail_areas_arr[i] = hail_areas[i].magnitude
                hail_centroid_lons = np.asarray(hail_centroid_lon)
                hail_centroid_lats = np.asarray(hail_centroid_lat)
                hail_con_areas = []
                hail_con_centroid_lon = []
                hail_con_centroid_lat = []
                hail_con_storm_lon = []
                hail_con_storm_lat = []

                #Consolidate Zhh objects as well
                zhh_areas_arr = np.zeros((len(zhh_areas)))
                zhh_max_arr = np.zeros((len(zhh_max)))
                zhh_core_avg_arr = np.zeros((len(zhh_core_avg)))
                for i in range(len(zhh_areas)):
                    zhh_areas_arr[i] = zhh_areas[i].magnitude
                    zhh_max_arr[i] = zhh_max[i]
                    zhh_core_avg_arr[i] = zhh_core_avg[i]
                zhh_centroid_lons = np.asarray(zhh_centroid_lon)
                zhh_centroid_lats = np.asarray(zhh_centroid_lat)
                zhh_con_areas = []
                zhh_con_maxes = []
                zhh_con_core_avg = []
                zhh_con_centroid_lon = []
                zhh_con_centroid_lat = []
                zhh_con_max_lon = []
                zhh_con_max_lat = []
                zhh_con_storm_lon = []
                zhh_con_storm_lat = []

                #Consolidate ZDR Column objects as well
                col_areas_arr = np.zeros((len(col_areas)))
                col_peaks_arr = np.zeros((len(col_areas)))
                col_depths_arr = np.zeros((len(col_areas)))
                for i in range(len(col_areas)):
                    col_areas_arr[i] = col_areas[i].magnitude
                    col_peaks_arr[i] = col_maxdepths[i]
                    col_depths_arr[i] = col_depths[i]
                col_centroid_lons = np.asarray(col_centroid_lon)
                col_centroid_lats = np.asarray(col_centroid_lat)
                col_con_areas = []
                col_con_peaks = []
                col_con_depths = []
                col_con_masks = []
                col_con_centroid_lon = []
                col_con_centroid_lat = []
                col_con_storm_lon = []
                col_con_storm_lat = []
                col_masks = np.asarray(col_masks)

                #Make empty rotation arrays
                rot1_con_mags = []
                rot1_con_storm_lon = []
                rot1_con_storm_lat = []
                rot1_con_lon = []
                rot1_con_lat = []
                rot1_con_area = []

                rot15_con_mags = []
                rot15_con_storm_lon = []
                rot15_con_storm_lat = []
                rot15_con_lon = []
                rot15_con_lat = []
                rot15_con_area = []

                rot2_con_mags = []
                rot2_con_storm_lon = []
                rot2_con_storm_lat = []
                rot2_con_lon = []
                rot2_con_lat = []
                rot2_con_area = []

                rot25_con_mags = []
                rot25_con_storm_lon = []
                rot25_con_storm_lat = []
                rot25_con_lon = []
                rot25_con_lat = []
                rot25_con_area = []

                for i in enumerate(max_lons_c):
                    try:
                        #Find the arc objects associated with this storm:
                        zdr_objects_lons = zdr_centroid_lons[np.where(zdr_storm_lon == max_lons_c[i[0]])]
                        zdr_objects_lats = zdr_centroid_lats[np.where(zdr_storm_lon == max_lons_c[i[0]])]

                        #Get the sum of their areas
                        zdr_con_areas.append(np.sum(zdr_areas_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))
                        #print("consolidated area", np.sum(zdr_areas_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))
                        zdr_con_maxes.append(np.max(zdr_max_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))
                        #print("consolidated max", np.max(zdr_areas_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))
                        zdr_con_means.append(np.mean(zdr_mean_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))
                        #print("consolidated mean", np.mean(zdr_areas_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))
                        zdr_con_max_lon.append(rlons_2d[np.where(ZDRmasked==np.max(zdr_max_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))])
                        zdr_con_max_lat.append(rlats_2d[np.where(ZDRmasked==np.max(zdr_max_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))])

                        #Find the actual centroids
                        weighted_lons = zdr_objects_lons * zdr_areas_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]
                        zdr_con_centroid_lon.append(np.sum(weighted_lons) / np.sum(zdr_areas_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))
                        weighted_lats = zdr_objects_lats * zdr_areas_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]
                        zdr_con_centroid_lat.append(np.sum(weighted_lats) / np.sum(zdr_areas_arr[np.where(zdr_storm_lon == max_lons_c[i[0]])]))
                        zdr_con_storm_lon.append(max_lons_c[i[0]])
                        zdr_con_storm_lat.append(max_lats_c[i[0]])
                        zdr_con_masks.append(np.sum(zdr_masks[np.where(zdr_storm_lon == max_lons_c[i[0]])],axis=0, dtype=bool))
                        mask_con = np.sum(zdr_masks[np.where(zdr_storm_lon == max_lons_c[i[0]])], axis=0, dtype=bool)
                        zdr_con_dev.append(np.std(ZDRmasked[mask_con]))
                        ZDRsorted = np.sort(ZDRmasked[mask_con])[::-1]
                        zdr_con_10max.append(np.mean(ZDRsorted[0:10]))
                        zdr_con_mode.append(stats.mode(ZDRmasked[mask_con]))
                        zdr_con_median.append(np.median(ZDRmasked[mask_con]))
                    except:
                        zdr_con_maxes.append(0)
                        zdr_con_means.append(0)
                        zdr_con_centroid_lon.append(0)
                        zdr_con_centroid_lat.append(0)
                        zdr_con_max_lon.append(0)
                        zdr_con_max_lat.append(0)
                        zdr_con_storm_lon.append(max_lons_c[i[0]])
                        zdr_con_storm_lat.append(max_lats_c[i[0]])
                        zdr_con_masks.append(0)
                        zdr_con_dev.append(0)
                        zdr_con_10max.append(0)
                        zdr_con_mode.append(0)
                        zdr_con_median.append(0)

                    try:
                        #Find the kdp objects associated with this storm:
                        kdp_objects_lons = kdp_centroid_lons[np.where(kdp_storm_lon == max_lons_c[i[0]])]
                        kdp_objects_lats = kdp_centroid_lats[np.where(kdp_storm_lon == max_lons_c[i[0]])]

                        #Get the sum of their areas
                        kdp_con_areas.append(np.sum(kdp_areas_arr[np.where(kdp_storm_lon == max_lons_c[i[0]])]))
                        kdp_con_maxes.append(np.max(kdp_max_arr[np.where(kdp_storm_lon == max_lons_c[i[0]])]))
                        kdp_con_max_lon.append(rlons_2d[np.where(KDPmasked==np.max(kdp_max_arr[np.where(kdp_storm_lon == max_lons_c[i[0]])]))])
                        kdp_con_max_lat.append(rlats_2d[np.where(KDPmasked==np.max(kdp_max_arr[np.where(kdp_storm_lon == max_lons_c[i[0]])]))])
                        #Find the actual centroids
                        weighted_lons_kdp = kdp_objects_lons * kdp_areas_arr[np.where(kdp_storm_lon == max_lons_c[i[0]])]
                        kdp_con_centroid_lon.append(np.sum(weighted_lons_kdp) / np.sum(kdp_areas_arr[np.where(kdp_storm_lon == max_lons_c[i[0]])]))
                        weighted_lats_kdp = kdp_objects_lats * kdp_areas_arr[np.where(kdp_storm_lon == max_lons_c[i[0]])]
                        kdp_con_centroid_lat.append(np.sum(weighted_lats_kdp) / np.sum(kdp_areas_arr[np.where(kdp_storm_lon == max_lons_c[i[0]])]))
                        kdp_con_storm_lon.append(max_lons_c[i[0]])
                        kdp_con_storm_lat.append(max_lats_c[i[0]])
                    except:
                        kdp_con_areas.append(0)
                        kdp_con_maxes.append(0)
                        kdp_con_max_lon.append(0)
                        kdp_con_max_lat.append(0)
                        kdp_con_centroid_lon.append(0)
                        kdp_con_centroid_lat.append(0)
                        kdp_con_storm_lon.append(0)
                        kdp_con_storm_lat.append(0)

                    try:
                        #Find the hail core objects associated with this storm:
                        hail_objects_lons = hail_centroid_lons[np.where(hail_storm_lon == max_lons_c[i[0]])]
                        hail_objects_lats = hail_centroid_lats[np.where(hail_storm_lon == max_lons_c[i[0]])]
                        #Get the sum of their areas
                        hail_con_areas.append(np.sum(hail_areas_arr[np.where(hail_storm_lon == max_lons_c[i[0]])]))
                        #Find the actual centroids
                        weighted_lons_hail = hail_objects_lons * hail_areas_arr[np.where(hail_storm_lon == max_lons_c[i[0]])]
                        hail_con_centroid_lon.append(np.sum(weighted_lons_hail) / np.sum(hail_areas_arr[np.where(hail_storm_lon == max_lons_c[i[0]])]))
                        weighted_lats_hail = hail_objects_lats * hail_areas_arr[np.where(hail_storm_lon == max_lons_c[i[0]])]
                        hail_con_centroid_lat.append(np.sum(weighted_lats_hail) / np.sum(hail_areas_arr[np.where(hail_storm_lon == max_lons_c[i[0]])]))
                        hail_con_storm_lon.append(max_lons_c[i[0]])
                        hail_con_storm_lat.append(max_lats_c[i[0]])
                    except:
                        hail_con_centroid_lon.append(0)
                        hail_con_centroid_lat.append(0)
                        hail_con_storm_lon.append(0)
                        hail_con_storm_lat.append(0)

                    #Change some things to arrays
                    rot_lon1=np.asarray(rot_lon1)
                    rot_lat1=np.asarray(rot_lat1)
                    rot_storm_lon_1=np.asarray(rot_storm_lon_1)
                    rot_storm_lat_1=np.asarray(rot_storm_lat_1)
                    rot_mag1=np.asarray(rot_mag1)
                    azarea1 = np.asarray(azarea1)

                    rot_lon15=np.asarray(rot_lon15)
                    rot_lat15=np.asarray(rot_lat15)
                    rot_storm_lon_15=np.asarray(rot_storm_lon_15)
                    rot_storm_lat_15=np.asarray(rot_storm_lat_15)
                    rot_mag15=np.asarray(rot_mag15)
                    azarea15 = np.asarray(azarea15)

                    rot_lon2=np.asarray(rot_lon2)
                    rot_lat2=np.asarray(rot_lat2)
                    rot_storm_lon_2=np.asarray(rot_storm_lon_2)
                    rot_storm_lat_2=np.asarray(rot_storm_lat_2)
                    rot_mag2=np.asarray(rot_mag2)
                    azarea2 = np.asarray(azarea2)

                    rot_lon25=np.asarray(rot_lon25)
                    rot_lat25=np.asarray(rot_lat25)
                    rot_storm_lon_25=np.asarray(rot_storm_lon_25)
                    rot_storm_lat_25=np.asarray(rot_storm_lat_25)
                    rot_mag25=np.asarray(rot_mag25)
                    azarea25 = np.asarray(azarea25)

                    try:
                        #Find the 1km rotation objects associated with this storm:
                        rot1_objects_lons = rot_lon1[np.where(rot_storm_lon_1 == max_lons_c[i[0]])]
                        rot1_objects_lats = rot_lat1[np.where(rot_storm_lon_1 == max_lons_c[i[0]])]
                        #Get the sum of their areas
                        rot_magstorm1 = rot_mag1[np.where(rot_storm_lon_1 == max_lons_c[i[0]])]
                        rot1_con_mags.append(np.max(rot_magstorm1))
                        rot1_con_storm_lon.append(max_lons_c[i[0]])
                        rot1_con_storm_lat.append(max_lats_c[i[0]])
                        rot1_con_lon.append(rot1_objects_lons[np.where(rot_magstorm1==np.max(rot_magstorm1))])
                        rot1_con_lat.append(rot1_objects_lats[np.where(rot_magstorm1==np.max(rot_magstorm1))])
                        rot1_con_area.append(azarea1[np.where(rot_magstorm1==np.max(rot_magstorm1))])
                    except:
                        rot1_con_mags.append(0)
                        rot1_con_storm_lon.append(0)
                        rot1_con_storm_lat.append(0)
                        rot1_con_lon.append(0)
                        rot1_con_lat.append(0)
                        rot1_con_area.append(0)

                    try:
                        #Find the 1.5km rotation objects associated with this storm:
                        rot15_objects_lons = rot_lon15[np.where(rot_storm_lon_15 == max_lons_c[i[0]])]
                        rot15_objects_lats = rot_lat15[np.where(rot_storm_lon_15 == max_lons_c[i[0]])]
                        #Get the sum of their areas
                        rot_magstorm15 = rot_mag15[np.where(rot_storm_lon_15 == max_lons_c[i[0]])]
                        rot15_con_mags.append(np.max(rot_magstorm15))
                        rot15_con_storm_lon.append(max_lons_c[i[0]])
                        rot15_con_storm_lat.append(max_lats_c[i[0]])
                        rot15_con_lon.append(rot15_objects_lons[np.where(rot_magstorm15==np.max(rot_magstorm15))])
                        rot15_con_lat.append(rot15_objects_lats[np.where(rot_magstorm15==np.max(rot_magstorm15))])
                        rot15_con_area.append(azarea15[np.where(rot_magstorm15==np.max(rot_magstorm15))])
                    except:
                        rot15_con_mags.append(0)
                        rot15_con_storm_lon.append(0)
                        rot15_con_storm_lat.append(0)
                        rot15_con_lon.append(0)
                        rot15_con_lat.append(0)
                        rot15_con_area.append(0)

                    try:
                        #Find the 2km rotation objects associated with this storm:
                        rot2_objects_lons = rot_lon2[np.where(rot_storm_lon_2 == max_lons_c[i[0]])]
                        rot2_objects_lats = rot_lat2[np.where(rot_storm_lon_2 == max_lons_c[i[0]])]
                        #Get the sum of their areas
                        rot_magstorm2 = rot_mag2[np.where(rot_storm_lon_2 == max_lons_c[i[0]])]
                        rot2_con_mags.append(np.max(rot_magstorm2))
                        rot2_con_storm_lon.append(max_lons_c[i[0]])
                        rot2_con_storm_lat.append(max_lats_c[i[0]])
                        rot2_con_lon.append(rot2_objects_lons[np.where(rot_magstorm2==np.max(rot_magstorm2))])
                        rot2_con_lat.append(rot2_objects_lats[np.where(rot_magstorm2==np.max(rot_magstorm2))])
                        rot2_con_area.append(azarea2[np.where(rot_magstorm2==np.max(rot_magstorm2))])
                    except:
                        rot2_con_mags.append(0)
                        rot2_con_storm_lon.append(0)
                        rot2_con_storm_lat.append(0)
                        rot2_con_lon.append(0)
                        rot2_con_lat.append(0)
                        rot2_con_area.append(0)

                    try:
                        #Find the 2.5km rotation objects associated with this storm:
                        rot25_objects_lons = rot_lon25[np.where(rot_storm_lon_25 == max_lons_c[i[0]])]
                        rot25_objects_lats = rot_lat25[np.where(rot_storm_lon_25 == max_lons_c[i[0]])]
                        #Get the sum of their areas
                        rot_magstorm25 = rot_mag25[np.where(rot_storm_lon_25 == max_lons_c[i[0]])]
                        rot25_con_mags.append(np.max(rot_magstorm25))
                        rot25_con_storm_lon.append(max_lons_c[i[0]])
                        rot25_con_storm_lat.append(max_lats_c[i[0]])
                        rot25_con_lon.append(rot25_objects_lons[np.where(rot_magstorm25==np.max(rot_magstorm25))])
                        rot25_con_lat.append(rot25_objects_lats[np.where(rot_magstorm25==np.max(rot_magstorm25))])
                        rot25_con_area.append(azarea25[np.where(rot_magstorm25==np.max(rot_magstorm25))])
                    except:
                        rot25_con_mags.append(0)
                        rot25_con_storm_lon.append(0)
                        rot25_con_storm_lat.append(0)
                        rot25_con_lon.append(0)
                        rot25_con_lat.append(0)
                        rot25_con_area.append(0)

                    try:
                        #Find the zhh objects associated with this storm:
                        zhh_objects_lons = zhh_centroid_lons[np.where(zhh_storm_lon == max_lons_c[i[0]])]
                        zhh_objects_lats = zhh_centroid_lats[np.where(zhh_storm_lon == max_lons_c[i[0]])]
                        #Get the sum of their areas
                        zhh_con_areas.append(np.sum(zhh_areas_arr[np.where(zhh_storm_lon == max_lons_c[i[0]])]))
                        zhh_con_maxes.append(np.max(zhh_max_arr[np.where(zhh_storm_lon == max_lons_c[i[0]])]))
                        zhh_con_core_avg.append(np.max(zhh_core_avg_arr[np.where(zhh_storm_lon == max_lons_c[i[0]])]))
                        zhh_con_max_lon.append(rlons_2d[np.where(REFmasked==np.max(zhh_max_arr[np.where(zhh_storm_lon == max_lons_c[i[0]])]))])
                        zhh_con_max_lat.append(rlats_2d[np.where(REFmasked==np.max(zhh_max_arr[np.where(zhh_storm_lon == max_lons_c[i[0]])]))])
                        #Find the actual centroids
                        weighted_lons_zhh = zhh_objects_lons * zhh_areas_arr[np.where(zhh_storm_lon == max_lons_c[i[0]])]
                        zhh_con_centroid_lon.append(np.sum(weighted_lons_zhh) / np.sum(zhh_areas_arr[np.where(zhh_storm_lon == max_lons_c[i[0]])]))
                        weighted_lats_zhh = zhh_objects_lats * zhh_areas_arr[np.where(zhh_storm_lon == max_lons_c[i[0]])]
                        zhh_con_centroid_lat.append(np.sum(weighted_lats_zhh) / np.sum(zhh_areas_arr[np.where(zhh_storm_lon == max_lons_c[i[0]])]))
                        zhh_con_storm_lon.append(max_lons_c[i[0]])
                        zhh_con_storm_lat.append(max_lats_c[i[0]])
                    except:
                        zhh_con_maxes.append(0)
                        zhh_con_core_avg.append(0)
                        zhh_con_max_lon.append(0)
                        zhh_con_max_lat.append(0)
                        zhh_con_centroid_lon.append(0)
                        zhh_con_centroid_lat.append(0)
                        zhh_con_storm_lon.append(0)
                        zhh_con_storm_lat.append(0)

                    try:
                        #Find the kdp objects associated with this storm:
                        col_objects_lons = col_centroid_lons[np.where(col_storm_lon == max_lons_c[i[0]])]
                        col_objects_lats = col_centroid_lats[np.where(col_storm_lon == max_lons_c[i[0]])]
                        #Get the sum of their areas
                        col_con_storm_lon.append(max_lons_c[i[0]])
                        col_con_storm_lat.append(max_lats_c[i[0]])
                        col_con_areas.append(np.sum(col_areas_arr[np.where(col_storm_lon == max_lons_c[i[0]])]))
                        weighted_lons_col = col_objects_lons * col_areas_arr[np.where(col_storm_lon == max_lons_c[i[0]])]
                        col_con_centroid_lon.append(np.sum(weighted_lons_col) / np.sum(col_areas_arr[np.where(col_storm_lon == max_lons_c[i[0]])]))
                        weighted_lats_col = col_objects_lats * col_areas_arr[np.where(col_storm_lon == max_lons_c[i[0]])]
                        col_con_centroid_lat.append(np.sum(weighted_lats_col) / np.sum(col_areas_arr[np.where(col_storm_lon == max_lons_c[i[0]])]))
                        col_con_peaks.append(np.max(col_peaks_arr[np.where(col_storm_lon == max_lons_c[i[0]])]))
                        mask_con_col = np.sum(col_masks[np.where(col_storm_lon == max_lons_c[i[0]])], axis=0, dtype=bool)
                        col_con_depths.append(np.mean(ZDR_sum_stuff[mask_con_col]))
                        #if len(col_areas_arr[np.where(col_storm_lon == max_lons_c[i[0]])])==0:
                        #    col_con_areas.append(0)
                        #elif col_areas_arr[np.where(col_storm_lon == max_lons_c[i[0]])].shape[0] == 1:
                        #col_con_areas.append(col_areas_arr[np.where(col_storm_lon == max_lons_c[i[0]])])
                        #col_con_maxes.append(np.max(col_max_arr[np.where(col_storm_lon == max_lons_c[i[0]])]))
                        #col_con_max_lon.append(rlons_2d[np.where(KDPmasked==np.max(kdp_max_arr[np.where(kdp_storm_lon == max_lons_c[i[0]])]))])
                        #col_con_max_lat.append(rlats_2d[np.where(KDPmasked==np.max(kdp_max_arr[np.where(kdp_storm_lon == max_lons_c[i[0]])]))])
                        #Find the actual centroids
                        #col_ind = np.where(col_areas_arr[np.where(col_storm_lon == max_lons_c[i[0]])] == np.max(col_areas_arr[np.where(col_storm_lon == max_lons_c[i[0]])]))
                        #col_con_centroid_lon.append(col_objects_lons[col_ind][0])
                        #col_con_centroid_lat.append(col_objects_lats[col_ind][0])
                        #Find the actual centroids

                    except:
                        #col_con_areas.append(0)
                        #kdp_con_maxes.append(0)
                        #kdp_con_max_lon.append(0)
                        #kdp_con_max_lat.append(0)
                        #uncomment
                        #col_con_centroid_lon.append(0)
                        #col_con_centroid_lat.append(0)
                        #col_con_storm_lon.append(0)
                        #col_con_storm_lat.append(0)
                        col_con_peaks.append(0)
                        col_con_depths.append(0)



                    if len(col_con_areas) < len(col_con_centroid_lon):
                        col_con_areas.append(0)



                    #Calculate KDP-ZDR separation
    #             kdp_con_centroid_lons1 = np.asarray(kdp_con_centroid_lon)
    #             kdp_con_centroid_lats1 = np.asarray(kdp_con_centroid_lat)
    #             zdr_con_centroid_lons1 = np.asarray(zdr_con_centroid_lon)
    #             zdr_con_centroid_lats1 = np.asarray(zdr_con_centroid_lat)
    #             #Eliminate consolidated arcs smaller than a specified area
    #             area = 2 #km*2
    #             zdr_con_areas_arr = np.asarray(zdr_con_areas)
    #             zdr_con_centroid_lats = zdr_con_centroid_lats1[zdr_con_areas_arr > area]
    #             zdr_con_centroid_lons = zdr_con_centroid_lons1[zdr_con_areas_arr > area]
    #             kdp_con_centroid_lats = kdp_con_centroid_lats1[zdr_con_areas_arr > area]
    #             kdp_con_centroid_lons = kdp_con_centroid_lons1[zdr_con_areas_arr > area]
    #             zdr_con_max_lons1 = np.asarray(zdr_con_max_lon)[zdr_con_areas_arr > area]
    #             zdr_con_max_lats1 = np.asarray(zdr_con_max_lat)[zdr_con_areas_arr > area]
    #             kdp_con_max_lons1 = np.asarray(kdp_con_max_lon)[zdr_con_areas_arr > area]
    #             kdp_con_max_lats1 = np.asarray(kdp_con_max_lat)[zdr_con_areas_arr > area]
    #             zdr_con_max1 = np.asarray(zdr_con_maxes)[zdr_con_areas_arr > area]
    #             zdr_con_areas1 = zdr_con_areas_arr[zdr_con_areas_arr > area]
                kdp_con_centroid_lat = np.asarray(kdp_con_centroid_lat)
                kdp_con_centroid_lon = np.asarray(kdp_con_centroid_lon)
                zdr_con_centroid_lat = np.asarray(zdr_con_centroid_lat)
                zdr_con_centroid_lon = np.asarray(zdr_con_centroid_lon)

#                 kdp_inds = np.where(kdp_con_centroid_lat*zdr_con_centroid_lat > 0)
#                 distance_kdp_zdr = g.inv(kdp_con_centroid_lon[kdp_inds], kdp_con_centroid_lat[kdp_inds], zdr_con_centroid_lon[kdp_inds], zdr_con_centroid_lat[kdp_inds])
#                 dist_kdp_zdr = distance_kdp_zdr[2] / 1000.
#                 #Now make an array for the distances which will have the same shape as the lats to prevent errors
#                 shaped_dist = np.zeros((np.shape(zdr_con_areas)))
#                 shaped_dist[kdp_inds] = dist_kdp_zdr

#                 #Get separation angle for KDP-ZDR centroids
#                 back_k = distance_kdp_zdr[1]
#                 for i in range(back_k.shape[0]):
#                     if distance_kdp_zdr[1][i] < 0:
#                         back_k[i] = distance_kdp_zdr[1][i] + 360

#                 forw_k = np.abs(back_k - storm_relative_dir)
#                 rawangle_k = back_k - storm_relative_dir
#                 #Account for weird angles
#                 for i in range(back_k.shape[0]):
#                     if forw_k[i] > 180:
#                         forw_k[i] = 360 - forw_k[i]
#                         rawangle_k[i] = (360-forw_k[i])*(-1)

#                 rawangle_k = rawangle_k*(-1)

#                 #Now make an array for the distances which will have the same shape as the lats to prevent errors
#                 shaped_ang = np.zeros((np.shape(zdr_con_areas)))
#                 shaped_ang[kdp_inds] = rawangle_k
#                 shaped_ang = (180-np.abs(shaped_ang))*(shaped_ang/np.abs(shaped_ang))

#                 new_angle_all = shaped_ang + storm_relative_dir

#                 shaped_ang = (new_angle_all - Bunkers_m)* (-1)

#                 shaped_ang = 180 - shaped_ang


                kdp_inds = np.where(kdp_con_centroid_lat*zdr_con_centroid_lat > 0)
                distance_kdp_zdr = g.inv(kdp_con_centroid_lon[kdp_inds], kdp_con_centroid_lat[kdp_inds], zdr_con_centroid_lon[kdp_inds], zdr_con_centroid_lat[kdp_inds])
                dist_kdp_zdr = distance_kdp_zdr[2] / 1000.
                #Now make an array for the distances which will have the same shape as the lats to prevent errors
                shaped_dist = np.zeros((np.shape(zdr_con_areas)))
                shaped_dist[kdp_inds] = dist_kdp_zdr

                #Get separation angle for KDP-ZDR centroids
                back_k = distance_kdp_zdr[1]
                for i in range(back_k.shape[0]):
                    if distance_kdp_zdr[1][i] < 0:
                        back_k[i] = distance_kdp_zdr[1][i] + 360

#                 forw_k = np.abs(back_k - storm_relative_dir)
#                 rawangle_k = back_k - storm_relative_dir
#                 #Account for weird angles
#                 for i in range(back_k.shape[0]):
#                     if forw_k[i] > 180:
#                         forw_k[i] = 360 - forw_k[i]
#                         rawangle_k[i] = (360-forw_k[i])*(-1)

#                 rawangle_k = rawangle_k*(-1)

                #Now make an array for the distances which will have the same shape as the lats to prevent errors
                shaped_ang = np.zeros((np.shape(zdr_con_areas)))
                shaped_ang[kdp_inds] = back_k

                ###Now let's consolidate everything to fit the Pandas dataframe!
                p_zdr_areas = []
                p_zdr_maxes = []
                p_zdr_means = []
                p_zdr_devs = []
                p_zdr_10max = []
                p_zdr_mode = []
                p_zdr_median = []
                p_hail_areas = []
                p_rot1 = []
                p_rot15 = []
                p_rot2 = []
                p_rot25 = []
                p_rot1area = []
                p_rot15area = []
                p_rot2area = []
                p_rot25area = []
                p_zhh_areas = []
                p_zhh_maxes = []
                p_zhh_core_avgs = []
                p_separations = []
                p_sp_angle = []
                p_col_areas = []
                p_col_max_depths = []
                p_col_depths = []
                p_kdp_areas = []
                p_kdp_maxes = []
                for storm in enumerate(max_lons_c):
                    matching_ind = np.flatnonzero(np.isclose(max_lons_c[storm[0]], zdr_con_storm_lon, rtol=1e-05))
                    if matching_ind.shape[0] > 0:
                        p_zdr_areas.append((zdr_con_areas[matching_ind[0]]))
                        p_zdr_maxes.append((zdr_con_maxes[matching_ind[0]]))
                        p_zdr_means.append((zdr_con_means[matching_ind[0]]))
                        p_zdr_devs.append((zdr_con_dev[matching_ind[0]]))
                        p_zdr_10max.append((zdr_con_10max[matching_ind[0]]))
                        p_zdr_mode.append((zdr_con_mode[matching_ind[0]]))
                        p_zdr_median.append((zdr_con_median[matching_ind[0]]))
                        p_separations.append((shaped_dist[matching_ind[0]]))
                        p_sp_angle.append((shaped_ang[matching_ind[0]]))
                    else:
                        p_zdr_areas.append((0))
                        p_zdr_maxes.append((0))
                        p_zdr_means.append((0))
                        p_zdr_devs.append((0))
                        p_zdr_10max.append((0))
                        p_zdr_mode.append((0))
                        p_zdr_median.append((0))
                        p_separations.append((0))
                        p_sp_angle.append((0))

                    matching_ind_hail = np.flatnonzero(np.isclose(max_lons_c[storm[0]], hail_con_storm_lon, rtol=1e-05))
                    if matching_ind_hail.shape[0] > 0:
                        p_hail_areas.append((hail_con_areas[matching_ind_hail[0]]))
                    else:
                        p_hail_areas.append((0))

                    matching_ind_rot1 = np.flatnonzero(np.isclose(max_lons_c[storm[0]], rot1_con_storm_lon, rtol=1e-05))
                    if matching_ind_rot1.shape[0] > 0:
                        p_rot1.append((rot1_con_mags[matching_ind_rot1[0]]))
                        p_rot1area.append((rot1_con_area[matching_ind_rot1[0]]))
                    else:
                        p_rot1.append((0))
                        p_rot1area.append((0))

                    matching_ind_rot15 = np.flatnonzero(np.isclose(max_lons_c[storm[0]], rot15_con_storm_lon, rtol=1e-05))
                    if matching_ind_rot15.shape[0] > 0:
                        p_rot15.append((rot15_con_mags[matching_ind_rot15[0]]))
                        p_rot15area.append((rot15_con_area[matching_ind_rot15[0]]))
                    else:
                        p_rot15.append((0))
                        p_rot15area.append((0))

                    matching_ind_rot2 = np.flatnonzero(np.isclose(max_lons_c[storm[0]], rot2_con_storm_lon, rtol=1e-05))
                    if matching_ind_rot2.shape[0] > 0:
                        p_rot2.append((rot2_con_mags[matching_ind_rot2[0]]))
                        p_rot2area.append((rot2_con_area[matching_ind_rot2[0]]))
                    else:
                        p_rot2.append((0))
                        p_rot2area.append((0))

                    matching_ind_rot25 = np.flatnonzero(np.isclose(max_lons_c[storm[0]], rot25_con_storm_lon, rtol=1e-05))
                    if matching_ind_rot25.shape[0] > 0:
                        p_rot25.append((rot25_con_mags[matching_ind_rot25[0]]))
                        p_rot25area.append((rot25_con_area[matching_ind_rot25[0]]))
                    else:
                        p_rot25.append((0))
                        p_rot25area.append((0))

                    matching_ind_zhh = np.flatnonzero(np.isclose(max_lons_c[storm[0]],zhh_con_storm_lon, rtol=1e-05))
                    if matching_ind_zhh.shape[0] > 0:
                        p_zhh_maxes.append((zhh_con_maxes[matching_ind_zhh[0]]))
                        p_zhh_areas.append((zhh_con_areas[matching_ind_zhh[0]]))
                        p_zhh_core_avgs.append((zhh_con_core_avg[matching_ind_zhh[0]]))
                    else:
                        p_zhh_areas.append((0))
                        p_zhh_maxes.append((0))
                        p_zhh_core_avgs.append((0))

                    matching_ind_kdp = np.flatnonzero(np.isclose(max_lons_c[storm[0]],kdp_con_storm_lon, rtol=1e-05))
                    if matching_ind_kdp.shape[0] > 0:
                        p_kdp_maxes.append((kdp_con_maxes[matching_ind_kdp[0]]))
                        p_kdp_areas.append((kdp_con_areas[matching_ind_kdp[0]]))
                    else:
                        p_kdp_areas.append((0))
                        p_kdp_maxes.append((0))

                    matching_ind_col = np.flatnonzero(np.isclose(max_lons_c[storm[0]], col_con_storm_lon, rtol=1e-05))
                    if matching_ind_col.shape[0] > 0:
                        p_col_areas.append((col_con_areas[matching_ind_col[0]]))
                        p_col_max_depths.append((col_con_peaks[matching_ind_col[0]]))
                        p_col_depths.append((col_con_depths[matching_ind_col[0]]))
                    else:
                        p_hail_areas.append((0))
                        p_col_max_depths.append((0))
                        p_col_depths.append((0))

                #Now start plotting stuff!
                try:
                    LL = (max_lons_c[tracking_ind]-0.3,max_lats_c[tracking_ind]-0.3,ccrs.PlateCarree())
                    UR = (max_lons_c[tracking_ind]+0.3,max_lats_c[tracking_ind]+0.3,ccrs.PlateCarree())
                    ax.set_extent([LL[0],UR[0],LL[1],UR[1]])
                except:
                    print('storm not here yet')
                #Now start plotting stuff!
    #             if np.asarray(zdr_centroid_lon).shape[0] > 0:
    #                 ax.scatter(zdr_centroid_lon, zdr_centroid_lat, marker = '*', s = 100, color = 'black', zorder = 10, transform=ccrs.PlateCarree())
    #             if np.asarray(kdp_centroid_lon).shape[0] > 0:
    #                 ax.scatter(kdp_centroid_lon, kdp_centroid_lat, marker = '^', s = 100, color = 'black', zorder = 10, transform=ccrs.PlateCarree())

                p_rot1a = np.asarray(p_rot1)
                p_rot15a = np.asarray(p_rot15)
                p_rot2a = np.asarray(p_rot2)
                p_rot25a = np.asarray(p_rot25)

                rot_1l = p_rot1a[tracking_ind]/493
                rot_3l = p_rot15a[tracking_ind]/493
                rot_5l = p_rot2a[tracking_ind]/493
                rot_7l = p_rot25a[tracking_ind]/493
                try:
                    if np.asarray(rot1_con_lon).shape[0] > 0:
                        mm_1= ax.scatter(rot1_con_lon, rot1_con_lat, marker = 'v', s = 100, color = 'red', zorder = 20, transform=ccrs.PlateCarree(), label="1km meso: %.4f s$^-$$^1$" %(rot_1l))
                    if np.asarray(rot15_con_lon).shape[0] > 0:
                        mm_3 = ax.scatter(rot15_con_lon, rot15_con_lat, marker = 'v', s = 100, color = 'orange', zorder = 20, transform=ccrs.PlateCarree(), label="3km meso: %.4f s$^-$$^1$" %(rot_3l))
                    if np.asarray(rot2_con_lon).shape[0] > 0:
                        mm_5 = ax.scatter(rot2_con_lon, rot2_con_lat, marker = 'v', s = 100, color = 'yellow', zorder = 20, transform=ccrs.PlateCarree(), label="5km meso: %.4f s$^-$$^1$" %(rot_5l))
                    if np.asarray(rot25_con_lon).shape[0] > 0:
                        mm_7 = ax.scatter(rot25_con_lon, rot25_con_lat, marker = 'v', s = 100, color = 'green', zorder = 20, transform=ccrs.PlateCarree(), label="7km meso: %.4f s$^-$$^1$" %(rot_7l)) 
                except:
                    print('rotation FAILED')
                    if np.asarray(rot1_con_lon).shape[0] > 0:
                        mm_1= ax.scatter(rot1_con_lon, rot1_con_lat, marker = 'v', s = 100, color = 'red', zorder = 20, transform=ccrs.PlateCarree(), label="1km meso")
                    if np.asarray(rot15_con_lon).shape[0] > 0:
                        mm_3 = ax.scatter(rot15_con_lon, rot15_con_lat, marker = 'v', s = 100, color = 'orange', zorder = 20, transform=ccrs.PlateCarree(), label="3km meso")
                    if np.asarray(rot2_con_lon).shape[0] > 0:
                        mm_5 = ax.scatter(rot2_con_lon, rot2_con_lat, marker = 'v', s = 100, color = 'yellow', zorder = 20, transform=ccrs.PlateCarree(), label="5km meso")
                    if np.asarray(rot25_con_lon).shape[0] > 0:
                        mm_7 = ax.scatter(rot25_con_lon, rot25_con_lat, marker = 'v', s = 100, color = 'green', zorder = 20, transform=ccrs.PlateCarree(), label="7km meso")


                #Uncomment to print all object areas
                #for i in enumerate(zdr_areas):
                #    plt.text(zdr_centroid_lon[i[0]]+.016, zdr_centroid_lat[i[0]]+.016, "%.2f km^2" %(zdr_areas[i[0]].magnitude), size = 23)
                    #plt.text(zdr_centroid_lon[i[0]]+.016, zdr_centroid_lat[i[0]]+.016, "%.2f km^2 / %.2f km / %.2f dB" %(zdr_areas[i[0]].magnitude, zdr_dist[i[0]], zdr_forw[i[0]]), size = 23)
                    #plt.annotate(zdr_areas[i[0]], (zdr_centroid_lon[i[0]],zdr_centroid_lat[i[0]]))
                #ax.contourf(rlons[0,:,:],rlats[0,:,:],KDPmasked,KDPlevels1,linewide = .01, colors ='b', alpha = .5)
                #plt.tight_layout()
                #plt.savefig('ZDRarcannotated.png')
                storm_times = []
                for l in range(len(max_lons_c)):
                    storm_times.append((time_start))
                tracking_index = tracking_index + 1

            #If there are no storms, set everything to empty arrays!
            else:
                storm_ids = []
                storm_ids = []
                max_lons_c = []
                max_lats_c = []
                storm_speeds = []
                storm_dirs = []
                p_zdr_areas = []
                p_zdr_maxes = []
                p_zdr_means = []
                p_zdr_devs = []
                p_zdr_10max = []
                p_zdr_mode = []
                p_zdr_median = []
                p_hail_areas = []
                p_rot1 = []
                p_rot15 = []
                p_rot2 = []
                p_rot25 = []
                p_rot1area = []
                p_rot15area = []
                p_rot2area = []
                p_rot25area = []
                p_zhh_areas = []
                p_zhh_maxes = []
                p_zhh_core_avgs = []
                p_separations = []
                p_sp_angle = []
                zdr_con_areas1 = []
                p_col_areas = []
                p_col_max_depths = []
                p_col_depths = []
                p_kdp_areas = []
                p_kdp_maxes = []
                storm_times = time_start


            #One final step: get observed separation angles using the storm motion directions
            #from the tracking algorithm
            o_sp_angle = np.asarray(p_sp_angle) - np.asarray(storm_dirs)
            b_sp_angle = np.asarray(p_sp_angle) - np.asarray(Bunkers_m)
            o_sp_angle[np.asarray(p_sp_angle)==0.0] = np.nan
            
            #Now record all data in a Pandas dataframe.
            new_cells = pd.DataFrame({
                'scan': scan_index,
                'storm_id' : storm_ids,
                'storm_id1' : storm_ids,
                'storm_lon' : max_lons_c,
                'storm_lat' : max_lats_c,
                'storm speed' : storm_speeds,
                'storm direction' : storm_dirs,
                'zdr_area' : p_zdr_areas,
                'zdr_max' : p_zdr_maxes,
                'zdr_mean' : p_zdr_means,
                'zdr_std' : p_zdr_devs,
                'zdr_10max' : p_zdr_10max,
                'zdr_mode' : p_zdr_mode,
                'zdr_median' : p_zdr_median,
                'hail_area' : p_hail_areas,
                'kdp_area' : p_kdp_areas,
                'kdp_max' : p_kdp_maxes,
                'rot_1km' : p_rot1,
                'rot_3km' : p_rot15,
                'rot_5km' : p_rot2,
                'rot_7km' : p_rot25,
                'rot_area1' : p_rot1area,
                'rot_area15' : p_rot15area,
                'rot_area2' : p_rot2area,
                'rot_area25' : p_rot25area,
                'zhh_area' : p_zhh_areas,
                'zhh_max' : p_zhh_maxes,
                'zhh_core_avg' : p_zhh_core_avgs,
                'kdp_zdr_sep' : p_separations,
                'kdp_zdr_angle' : o_sp_angle,
                'column_area' : p_col_areas,
                'column_max_depth' : p_col_max_depths,
                'column_mean_depth' : p_col_depths,
                'times' : storm_times
            })
            new_cells.set_index(['scan', 'storm_id'], inplace=True)
            if scan_index == 0:
                tracks_dataframe = new_cells
            else:
                tracks_dataframe = tracks_dataframe.append(new_cells)
            n = n+1
            scan_index = scan_index + 1

            #Plot the consolidated stuff!
            #Write some text objects for the ZDR arc attributes to add to the placefile
            f.write("Color: 139 000 000 \n")
            f.write('Font: 1, 30, 1,"Arial" \n')
            for y in range(len(max_lats_c)):
                #f.write('Text: '+str(max_lats_c[y])+','+str(max_lons_c[y])+', 1, "X"," Arc Area: '+str(p_zdr_areas[y])+'\\n Arc Mean: '+str(p_zdr_means[y])+'\\n KDP-ZDR Separation: '+str(p_separations[y])+'\\n Separation Angle: '+str(p_sp_angle[y])+'" \n')
                #f.write('Text: '+str(max_lats_c[y])+','+str(max_lons_c[y])+', 1, "X"," Arc Area: %.2f km^2 \\n Arc Mean: %.2f dB \\n Arc 10 Max Mean: %.2f dB \\n KDP-ZDR Separation: %.2f km \\n Separation Angle: %.2f degrees \\n ZDR Column Area: %.2f km^2 \\n ZDR Column Depth: %.2f m \\n Hail Area: %.2f km^2" \n' %(p_zdr_areas[y], p_zdr_means[y], p_zdr_10max[y], p_separations[y], p_sp_angle[y], p_col_areas[y], p_col_max_depths[y]*250, p_hail_areas[y]))
                f.write('Text: '+str(max_lats_c[y])+','+str(max_lons_c[y])+', 1, "X"," Arc Area: %.2f km^2 \\n Separation Angle: %.2f degrees \\n ZDR Column Area: %.2f km^2 \\n ZDR Column Depth: %.2f m \\n Hail Area: %.2f km^2" \n' %(p_zdr_areas[y], o_sp_angle[y], p_col_areas[y], p_col_max_depths[y]*250, p_hail_areas[y]))
                f.flush()


            title_plot = plt.title(station+' SPORK Analysis '+str(time_start.year)+'-'+str(time_start.month)+'-'+str(time_start.day)+
                                       ' '+str(hour)+':'+str(minute)+' UTC', size = 25)

            try:
                plt.plot([zdr_con_centroid_lon[kdp_inds], kdp_con_centroid_lon[kdp_inds]], [zdr_con_centroid_lat[kdp_inds],kdp_con_centroid_lat[kdp_inds]], color = 'k', linewidth = 5, transform=ccrs.PlateCarree())
            except:
                print('Separation Angle Failure')

            ref_centroid_lon = max_lons_c
            ref_centroid_lat = max_lats_c
            if len(max_lons_c) > 0:
                ax.scatter(max_lons_c,max_lats_c, marker = "o", color = 'k', s = 500, alpha = .6)
                for i in enumerate(ref_centroid_lon): 
                    plt.text(ref_centroid_lon[i[0]]+.016, ref_centroid_lat[i[0]]+.016, "storm_id: %.1f" %(storm_ids[i[0]]), size = 25)
            #Comment out this line if not plotting tornado tracks
            #plt.plot([start_torlons, end_torlons], [start_torlats, end_torlats], color = 'purple', linewidth = 5, transform=ccrs.PlateCarree())
            #Add legend stuff
            p_zdr_areas = np.asarray(p_zdr_areas)
            p_hail_areas = np.asarray(p_hail_areas)
            p_col_areas = np.asarray(p_col_areas)
            p_sp_angle = np.asarray(p_sp_angle)

            try:
                print(max_lons_c[tracking_ind], 'tracking lon')
                print(p_zdr_areas[tracking_ind], 'tracking ind areas')
                legend_zdr = p_zdr_areas[tracking_ind]
                legend_hail = p_hail_areas[tracking_ind]
                legend_col = p_col_areas[tracking_ind]
                legend_sep = p_sp_angle[tracking_ind]
            except:
                legend_zdr=0
                legend_hail=0
                legend_col=0
                legend_sep=0
            try:
                zdr_outline = mlines.Line2D([], [], color='blue', linewidth = 5, linestyle = 'solid', label='ZDR Arc Area: %.1f km$^2$' %(legend_zdr))
                kdp_outline = mlines.Line2D([], [], color='green', linewidth = 5,linestyle = 'solid', label='KDP Foot Outline')
                hail_outline = mlines.Line2D([], [], color='gold', linewidth = 5,linestyle = 'solid', label='Hailfall Area: %.1f km$^2$' %(legend_hail))
                column_outline = mlines.Line2D([], [], color='cyan', linewidth = 5,linestyle = 'solid', label='ZDR Column Area: %.1f km$^2$' %(legend_col))
                z_outline = mlines.Line2D([], [], color='grey', linewidth = 5,linestyle = 'solid', label='Z = 40 dBZ')

                separation_vector = mlines.Line2D([], [], color='black', linewidth = 5,linestyle = 'solid', label='KDP-ZDR Separation: %.1f deg.' %(legend_sep))
            except:
                zdr_outline = mlines.Line2D([], [], color='blue', linewidth = 5, linestyle = 'solid', label='ZDR Arc Outline')
                kdp_outline = mlines.Line2D([], [], color='green', linewidth = 5,linestyle = 'solid', label='KDP Foot Outline')
                hail_outline = mlines.Line2D([], [], color='gold', linewidth = 5,linestyle = 'solid', label='Hailfall Area')
                column_outline = mlines.Line2D([], [], color='cyan', linewidth = 5,linestyle = 'solid', label='ZDR Column Outline')
                z_outline = mlines.Line2D([], [], color='grey', linewidth = 5,linestyle = 'solid', label='Z = 40 dBZ')

                separation_vector = mlines.Line2D([], [], color='black', linewidth = 5,linestyle = 'solid', label='KDP-ZDR Separation')
            #tor_track = mlines.Line2D([], [], color='purple', linewidth = 5,linestyle = 'solid', label='Tornado Tracks')
            elevation = mlines.Line2D([], [], color='grey', linewidth = 5,linestyle = 'solid', label='Height AGL (m)')
            rotlevs = np.arange(1, 12, 0.5)
            try:
                if np.max(az_masked[4,:,:]) > 1:
                    plt.contour(rlons_2d, rlats_2d, az_masked[4,:,:], rotlevs, cmap=plt.cm.magma, linewidths=2, zorder=18)
                else: 
                    print('np spin')
            except:
                print('no spinny things')
            print(radar_v, 'radar_v')
            try:
                plt.legend(handles=[zdr_outline, kdp_outline, hail_outline, column_outline, separation_vector, z_outline, mm_1, mm_3, mm_5, mm_7], loc = 3, fontsize = 25)
            except:
                print('oops storm')
            alt_levs = [1000, 2000]
            plt.savefig('Machine_Learning/SPORK_RFN'+station+str(time_start.year)+str(time_start.month)+str(day)+str(hour)+str(minute)+'.png')
            print('Figure Saved')
            print(p_rot1)
            print(p_rot15)
            print(p_rot2)
            print(p_rot25)
            
            print(p_sp_angle, 'p_sp_angles')
            print(o_sp_angle, 'o_sp_angles')
            print(storm_dirs, 'storm_dirs')
            
            #print(rot_mag1, rot_lat1, rot_lon1, rot_storm_lon_1, rot_storm_lat_1)

            #print('consolidated', rot1_con_mags, rot1_con_lon, rot1_con_lat, rot1_con_lon, rot1_con_storm_lon, rot1_con_storm_lat)
            plt.close()
            zdr_out_list.append(zdr_outlines)
            f_l = f_l+1
            #except:
            #    traceback.print_exc()
            #    continue
    f.close()
    plt.show()
    print('Fin')
    #export_csv = tracks_dataframe.to_csv(r'C:\Users\Nick\Downloads\tracksdataframe.csv',index=None,header=True)
    return tracks_dataframe, zdr_out_list, col_con_areas, col_con_centroid_lon, col_con_storm_lon