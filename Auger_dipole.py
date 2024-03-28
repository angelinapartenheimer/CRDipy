import numpy as np
import healpy as hp


def healpy2coord(phi, theta, galCoord = False, rotator = None):
    
    if galCoord: 
        theta, phi = rotator(theta, phi)
    b = np.pi/2 - theta
    l = project_angle(phi - np.pi, 2*np.pi) #project onto [0, 2*pi]
    return l, b


def project_angle(angle, interval = 360.):
    if angle >= 0.0:
        return angle - np.floor(angle/interval)*interval
    else:
        return angle - np.ceil(angle/interval)*interval+interval

    
    
def equatorial2horizontal(ra, dec):
    
    lat = np.radians(-35) #latitude of the Pierre Auger Observatory
    slat = np.sin(lat)
    clat = np.cos(lat)
    sdec = np.sin(np.radians(dec))
    cdec = np.cos(np.radians(dec))
    
    #Random hour angle
    ha = 2*np.pi*np.random.rand()
    sha = np.sin(ha)
    cha = np.cos(ha)

    #Elevation and zenith
    el =  np.arcsin( slat * sdec + clat * cdec * cha )
    sel = np.sin(el)
    zenith = np.degrees(np.pi/2 - el)
    
    #Azimuth
    azimuth = np.degrees(np.arctan2(-sha*cdec*clat, sdec - slat*sel) + np.pi/2) #Auger measures azimuth counterclockwise from East
    
    return azimuth, zenith

####### Event generation #######

def generate_events(healpy_map, nside = 128, galCoord = False, number_events = 32000, zenith_cut_deg = 80, outfile = None): 
    
    if galCoord: 
        r = hp.rotator.Rotator(coord = ['G', 'C'])
    else: 
        r = None
    
    healpy_map = healpy_map.astype(float)
    healpy_map /= np.max(healpy_map) #normalize
    event_indices = np.where(healpy_map > (1/number_events))[0]
    
    ras = []
    decs = []
    zens = []
    azs = []
    
    num = 0
    while num < number_events: 
        
        #Choose a random event
        random_event_index = event_indices[np.random.randint(0, len(event_indices))]
        if healpy_map[random_event_index] < np.random.rand():
            continue
        
        #Event coordinates
        theta, phi = hp.pix2ang(nside, random_event_index)
        ra, dec = healpy2coord(phi, theta, galCoord, r)
        ra = np.degrees(ra)
        dec = np.degrees(dec)
        
        #Zenith cut
        az, zen = equatorial2horizontal(ra, dec)
        if zen > zenith_cut_deg: 
            continue
            
        #Zenith-based event selection selection
        if np.cos(np.radians(zen)) < np.random.rand():
            continue
        
        #Event is kept 
        ras.append(ra)
        decs.append(dec)
        zens.append(zen)
        azs.append(az)
        num += 1
    
    if outfile: 
        events = np.array([ras, decs, zens, azs]).transpose()
        np.savetxt(outfile, events, header = 'RA, DEC, ZEN, AZ')
        
    else: 
        events = {'RA': ras, 'DEC': decs, 'ZEN': zens, 'AZ': azs}
        return events

    
    
####### Dipole evaluation #######


def Rayleigh(angle, get_b = False):
    
    a = 2. * np.average(np.cos(np.radians(angle)))
    b = 2. * np.average(np.sin(np.radians(angle)))
    
    if get_b: return b
    
    alpha0 = np.arctan2(b,a)
    r = np.sqrt(a*a + b*b)

    return r, alpha0



def get_dipole(events):
    
    ra = events['RA']
    dec = events['DEC']
    zenith = events['ZEN']
    azimuth = events['AZ']
    
    r, alpha_d = Rayleigh(ra)
    
    alpha_d = np.degrees(alpha_d)
    alpha_d = project_angle(alpha_d)
    
    cosd = np.average(np.cos(np.radians(dec)))
    dperp = r/cosd
    
    lat = np.radians(-35) #latitude of the Pierre Auger observatory
    cosl = np.cos(lat)
    
    b = Rayleigh(azimuth, get_b = True)
    sintheta = np.average(np.sin(np.radians(zenith)))
    
    dz = b/(cosl*sintheta)
    
    d = np.sqrt(dperp**2 + dz**2)
    
    dec_d = np.arctan(dz/dperp)
    dec_d = np.degrees(dec_d)
    
    print("Cosmic ray dipole in equatorial coordinates \n amplitude {}% \n right ascension {}° \n declination {}°".format(np.round(d*100, 2), np.round(alpha_d, 2), np.round(dec_d, 2)))

