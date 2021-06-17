from siphon.radarserver import RadarServer
from datetime import datetime, timedelta
import pyart
import netCDF4

def get_latest_scan_SPORK(site):
    rs = RadarServer('http://tds-nexrad.scigw.unidata.ucar.edu/thredds/radarServer/nexrad/level2/S3/')
    query = rs.query()
    query.stations(site).time(datetime.utcnow())
    cat = rs.get_catalog(query)
    cat.datasets
    for item in sorted(cat.datasets.items()):
        # After looping over the list of sorted datasets, pull the actual Dataset object out
        # of our list of items and access over CDMRemote
        try:
            ds = item[1]
            radar1 = pyart.io.nexrad_cdm.read_nexrad_cdm(ds.access_urls['OPENDAP'])
            time_start = netCDF4.num2date(radar1.time['data'][0], radar1.time['units'])
        except:
            print('nope')
    return radar1, time_start