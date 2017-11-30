import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Tool_pressure_generator import *

sys.path.append('/data1/caldas/Pytmosph3R/PyRouts/')

from pytransfert import *
print JWST

########################################################################################################################
########################################################################################################################

# Informations diverses sur l'etude

path = "/data1/caldas/Pytmosph3R/"
name_file = "Tools/Files"
name_exo = "HD209458"
name_source = "Source"
opac_file, param_file, stitch_file = 'Opacity', 'Parameters', 'Stitch'
version = 6.2

########################################################################################################################
########################################################################################################################

# Donnees de base

reso_long, reso_lat = 64, 48
t, t_selec, phi_rot, phi_obli, inclinaison = 0, 5, 0.00, 0.00, 0.00
if inclinaison != 0. :
    phi_obli = np.abs(phi_obli+inclinaison-np.pi/2.)

########################################################################################################################

# Proprietes de l'exoplanete

Rp = 15.*R_T
Mp = 220.*M_T

#Rp = 0.246384689105*R_J
#Mp = 0.0206006322445*M_J
g0 = G*Mp/(Rp**2)

# Proprietes de l'etoile hote

Rs = 1.155*R_S
Ts = 6065.
d_al = 154*9.461e+15

#Rs = 0.206470165349*R_S
#Ts = 3000.

# Proprietes en cas de lecture d'un diagfi

alpha_step, delta_step = 2*np.pi/np.float(reso_long), np.pi/np.float(reso_lat)

# Proprietes de l'atmosphere

#n_species = np.array(['H2','He','H2O','CH4','N2','NH3','CO','CO2'])
#n_species_active = np.array(['H2O','CH4','NH3','CO','CO2'])
n_species = np.array(['H2','He','H2O'])
n_species_active = np.array(['H2O'])

# Proprietes de l'atmosphere isotherme

T_iso_array, P_surf, P_tau = np.array([1000.,2000.]), 1.e+6, 1.e+3
x_ratio_species_active = np.array([0.05])
x_ratio_species_inactive = np.array([])
M_species, M, x_ratio_species = ratio(n_species,x_ratio_species_active,IsoComp=True)

# Proprietes des nuages

c_species = np.array([])
c_species_name = np.array([])
rho_p = np.array([])
r_eff = 0.5e-6

########################################################################################################################

# Crossection

n_species_cross = np.array(['H2O','CH4','NH3','CO','CO2'])
m_species = np.array([])
domain, domainn, source = "HR", "HR", "bin10"
dim_bande, dim_gauss = 3000, 16

# Selection des sections efficaces

ind_cross, ind_active = index_active (n_species,n_species_cross,n_species_active)

# Informations generale sur les donnees continuum

cont_tot = np.array(['H2-H2_2011.cia','H2-He_2011.cia','H2O_CONT_SELF.dat','H2O_CONT_FOREIGN.dat'])
cont_species = np.array(['H2','He','H2Os','H2O'])
cont_associations = np.array(['h2h2','h2he','h2oh2o','h2ofor'])
class continuum :
    def __init__(self) :
        self.number = cont_tot.size
        self.associations = cont_associations
        self.species = cont_species
        self.file_name = cont_tot

########################################################################################################################

# Proprietes de maille

h, P_h, n_layers = 1.36e+7, 1.e-4, 100
delta_z, r_step, x_step, theta_number = 3.0e+4, 3.0e+4, 3.0e+4, 96
z_array = np.arange(h/np.float(delta_z)+1)*float(delta_z)
theta_step = 2*np.pi/np.float(theta_number)
Upper = "Isotherme"
number = 3 + n_species.size + m_species.size + c_species.size

# Choix dans la section de la maille

lim_alt, rupt_alt, beta = h, 0.e+0, 5.
beta_rad = beta*2*np.pi/(360.)
lat, long = 24, 47
z_lim = int(lim_alt/delta_z)
z_reso = int(h/delta_z) + 1

# En cas de modulation de la densite

type = np.array(['',0.00])
nest_out_path = '%sOutput/Water_3D_duo_0.5_1/multinest/'%(path)

########################################################################################################################
########################################################################################################################

# Les options choisies lors de l'etude

Tracer = False          ###### S'il y a des marqueurs
Cloudy = False          ###### S'il y a des nuages
Middle = True          ###### Construction de la maille sur le milieu des couches

########################################################################################################################

# Parameters

Profil = True          ###### Reproduit la maille spherique en altitude
Surf = True            ###### Si des donnees de surface sont accessibles
LogInterp = False       ###### Interpolation de la pression via le logarithme
TopPressure = True     ###### Si nous voulons fixer le toit de l'atmosphere par rapport a une pression minimale
Composition = False     ###### Se placer a l'equilibre thermodynamique

Parameters = True

Cylindre = False        ###### Construit la maille cylindrique
Obliquity = False       ###### Si l'exoplanete est inclinee
Layers = True

Corr = False            ###### Traite les parcours optiques
Gravity = False         ###### Pour travailler a gravite constante
Discret = True         ###### Calcul les distances discretes
Integral = False        ###### Effectue l'integration sur les chemins optiques
Ord = False             ###### Si Discreet == False, Ord permet de calculer les indices avec l'integration

Matrix = True          ###### Transposition de la maille spherique dans la maille cylindrique

Convert = True         ###### Lance la fonction convertator qui assure l'interpolation des sections efficaces
Kcorr = False           ###### Sections efficaces ou k-correles
Molecular = True       ###### Effectue les calculs pour l'absorption moleculaire
Cont = True            ###### Effectue les calculs pour l'absorption par les collisions
Scatt = True           ###### Effectue les calculs pour l'absorption par diffusion Rayleigh
Cl = False              ###### Effectue les calculs pour l'absorption par les nuages
Optimal = False         ###### Interpolation optimal (Waldmann et al.)
TimeSelec = True       ###### Si nous etudions un temps precis de la simulation

########################################################################################################################

# Cylindric transfert

Cylindric_transfert_3D = True

Isolated = False        ###### Ne tiens pas compte de l'absorption moleculaire
Continuum = True       ###### Tiens compte de l'absorption par les collisions
Scattering = True      ###### Tiens compte de l'absorption par la diffusion
Clouds = False          ###### Tiens compte de l'absoprtion par les nuages
Single = "no"           ###### Isole une espece de nuage
Rupt = False            ###### Si l'atmosphere est tronquee
LimTop = False         ###### Si on limite l'altitude max
Discreet = True        ###### Calcul discret
Integration = False     ###### Calcul integral
Module = False          ###### Si nous souhaitons moduler la densite de reference

D3Maille = True        ###### Si nous voulons resoudre le transfert dans la maille 3D
TimeSel = True         ###### Si nous etudions un temps precis de la simulation

########################################################################################################################

Script = True          ###### Si nous voulons avoir une version .dat du spectre
ErrOr = True           ###### Si calculons le bruit de photon pour un instrument donne
detection = JWST()
Noise = False           ###### Si nous voulons bruiter le signal a partir du bruit de photon calcule
Pressure_plot = False   ###### Si nous voulons observer les cartes photospheriques
Signature = False       ###### Si on souhaite visualiser les zones radiativement explorees
Distribution = False    ###### Permet de visualiser la distribution de spectre des distributions a posteriori

########################################################################################################################

# Plot

View = False

Radius = True          ###### Spectre rayon effective = f(longueur d'onde)
Flux = True            ###### Spectre flux = f(longueur d'onde)

########################################################################################################################
########################################################################################################################

# Sauvegardes

save_adress = "/data1/caldas/Pytmosph3R/Tools/%s_real_npy/"%(name_exo)
special = ''
stud = stud_type(r_eff,Single,Continuum,Isolated,Scattering,Clouds)
if Composition == False :
    save_name_3D = "%s%s_3D_duo_linear_real_%i_%i_%i_%.2f"%(save_adress,name_exo,np.amin(T_iso_array),np.amax(T_iso_array),beta,P_tau/(1.e+5))
else :
    save_name_3D = "%s%s_3D_duo_linear_real_%i_%i_%i_%.2f_eq"%(save_adress,name_exo,np.amin(T_iso_array),np.amax(T_iso_array),beta,P_tau/(1.e+5))

########################################################################################################################
########################################################################################################################

reso_alt = int(h/1000)
reso_long = int(reso_long)
reso_lat = int(reso_lat)
message_clouds = ''
if Cloudy == True :
    for i in range(c_species.size) :
        message_clouds += '%s (%.2f microns/%.3f)  '%(c_species[i],r_eff*10**6,rho_p[i]/1000.)
    print 'Clouds in the atmosphere (grain radius/density) : %s'%(message_clouds)
else :
    print 'There is no clouds'
if TopPressure != True :
    print 'Width of layers : %i m'%(delta_z)
    print 'Top of the atmosphere : %i m'%(h)
print 'Mean radius of the exoplanet : %i m'%(Rp)
print 'Mean surface gravity : %.2f m/s^2'%(g0)
print 'Mean molar mass : %.5f kg/mol'%(M)
print 'Extrapolation type for the upper atmosphere : %s'%(Upper)
number = 2 + m_species.size + c_species.size + n_species.size + 1
print 'Resolution of the GCM simulation (latitude/longitude) : %i/%i'%(reso_lat,reso_long)

########################################################################################################################





########################################################################################################################
########################################################################################################################
###########################################      PARAMETERS      #######################################################
########################################################################################################################
########################################################################################################################


if Profil == True :

    if Composition == True :
        T_comp = np.load('%s%s/T_comp_%s.npy'%(path,name_source,name_exo))
        P_comp = np.load('%s%s/P_comp_%s.npy'%(path,name_source,name_exo))
        comp = np.load('%s%s/x_species_comp_%s.npy'%(path,name_source,name_exo))
    if TopPressure == True :
        T_Ref = np.amax(T_iso_array)
        alp = R_gp*T_Ref/(g0*M)*np.log(P_h/P_surf)
        h_top = round(-alp/(1+alp/Rp),-5)
        delta_z, r_step, x_step = h_top/np.float(n_layers), h_top/np.float(n_layers), h_top/np.float(n_layers)
        print 'Width of layers : %i m'%(delta_z)
        print 'Top of the atmosphere : %i m'%(h_top)

    data_convert = np.zeros((number,1,n_layers+2,reso_lat+1,reso_long+1))
    T_min, T_max = np.amin(T_iso_array), np.amax(T_iso_array)
    d_lim = (Rp+h)*np.cos(np.pi/2.-beta_rad)
    alp_max = R_gp*T_max/(g0*M)*np.log(P_tau/P_surf)
    n_lim = -alp_max/(1+alp_max/Rp)

    bar = ProgressBar(reso_lat+1,'Data generation')

    for i_lat in range(reso_lat+1) :
        for i_long in range(reso_long+1) :
            z_maxi = 0
            phi_lat = -np.pi/2.+i_lat*np.pi/(np.float(reso_lat))
            phi_long = i_long*2*np.pi/(reso_long)
            x = np.abs((Rp+h)*np.cos(phi_lat)*np.cos(phi_long))

            if x <= d_lim and i_lat != 0  and i_lat != reso_lat and beta_rad >= theta_step :
                if i_long >= 0. and i_long < reso_long/4. :
                    T = T_max - (d_lim - x)*(T_max-T_min)/(2*d_lim)
                if i_long >= 3*reso_long/2. and i_long < reso_long :
                    T = T_max - (d_lim - x)*(T_max-T_min)/(2*d_lim)
                if i_long >= reso_long/4. and i_long < reso_long/2. :
                    T = T_min + (d_lim - x)*(T_max-T_min)/(2*d_lim)
                if i_long >= reso_long/2. and i_long < 3*reso_long/2. :
                    T = T_min + (d_lim - x)*(T_max-T_min)/(2*d_lim)
            else :
                if i_long >= reso_long/4. and i_long < 3.*reso_long/4. :
                    T = T_min
                else :
                    T = T_max
            if i_lat == 0 or i_lat == reso_lat :
                if beta_rad >= theta_step :
                    T = (T_max+T_min)/2.
                else :
                    if i_long >= reso_long/4. and i_long < 3.*reso_long/4. :
                        T = T_min
                    else :
                        T = T_max

            for i_n in range(n_layers+2) :
                if i_n == 0 :
                    z = 0
                else :
                    if i_n == n_layers+1 :
                        z = h_top
                    else :
                        z = (i_n-0.5)*delta_z

                if z < n_lim :
                    data_convert[1,0,i_n,i_lat,i_long] = T_max
                else :
                    if z_maxi == 0 :
                        z_maxi = z - delta_z
                        P_top = data_convert[0,0,i_n-1,i_lat,i_long]
                    data_convert[1,0,i_n,i_lat,i_long] = T

                if Composition == False :
                    if z < n_lim :
                        data_convert[0,0,i_n,i_lat,i_long] = P_surf*np.exp(-g0*M/(R_gp*T)*z/(1+z/Rp))
                    else :
                        data_convert[0,0,i_n,i_lat,i_long] = P_top*np.exp(-g0*(1/(1+z_maxi/Rp))**2*M/(R_gp*T)*(z-z_maxi)/(1+(z-z_maxi)/(Rp+z_maxi)))
                    data_convert[2:2+n_species.size,0,i_n,i_lat,i_long] = x_ratio_species
                    data_convert[number-1,0,i_n,i_lat,i_long] = M
                else :
                    res, c_grid, i_grid = interp2olation_uni_multi(data_convert[0,0,i_n,i_lat,i_long],data_convert[1,0,i_n,i_lat,i_long],P_comp,T_comp,comp)
                    data_convert[2:2+n_species.size,0,i_n,i_lat,i_long] = res/np.nansum(res)
                    data_convert[number-1,0,i_n,i_lat,i_long] = np.nansum(M_species*data_convert[2:2+n_species.size,0,i_n,i_lat,i_long])
                    if i_n == 0 :
                        data_convert[0,0,i_n,i_lat,i_long] = P_surf
                    else :
                        g = g0*1/(1+z/Rp)**2
                        if i_n == 1 or i_n == n_layers+1 :
                            delta = delta_z/2.
                        else :
                            delta = delta_z
                        data_convert[0,0,i_n,i_lat,i_long] = data_convert[0,0,i_n-1,i_lat,i_long]*np.exp(-g*data_convert[number-1,0,i_n,i_lat,i_long]/(R_gp*T)*delta)
        bar.animate(i_lat+1)

    print data_convert[0,0,:,0,0]
    print data_convert[1,0,:,0,0]

    if TopPressure == True :
        h = h_top
        reso_alt = int(h/1000)
        z_array = np.arange(h/np.float(delta_z)+1)*float(delta_z)
        if LimTop == False :
            lim_alt = h
        save_adress = "/data1/caldas/Pytmosph3R/Tools/%s_real_npy/"%(name_exo)
        if Composition == False :
            save_name_3D = "%s%s_3D_duo_linear_real_%i_%i_%i_%.2f"%(save_adress,name_exo,np.amin(T_iso_array),np.amax(T_iso_array),beta,P_tau/(1.e+5))
        else :
            save_name_3D = "%s%s_3D_duo_linear_real_%i_%i_%i_%.2f_eq"%(save_adress,name_exo,np.amin(T_iso_array),np.amax(T_iso_array),beta,P_tau/(1.e+5))

    np.save("%s%s/%s/%s_data_convert_%i%i%i.npy"%(path,name_file,param_file,name_exo,reso_alt,reso_long,reso_lat),\
                data_convert)

# Telechargement des parametres d'equilibre thermodynamique

########################################################################################################################

if Parameters == True :

########################################################################################################################

    if Cylindre == True :

        p_grid,q_grid,z_grid = cylindric_assymatrix_parameter(Rp,h,alpha_step,delta_step,r_step,theta_step,theta_number,\
                                x_step,z_array,phi_rot,phi_obli,reso_long,reso_lat,Obliquity,Middle,Layers)

        np.save("%s%s/%s/p_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,reso_lat,\
                reso_alt,r_step,phi_rot,phi_obli),p_grid)
        np.save("%s%s/%s/q_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,reso_lat,\
                reso_alt,r_step,phi_rot,phi_obli),q_grid)
        np.save("%s%s/%s/z_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,reso_lat,\
                reso_alt,r_step,phi_rot,phi_obli),z_grid)

        del p_grid,q_grid,z_grid

########################################################################################################################

    if Corr == True :

        p_grid = np.load("%s%s/%s/p_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli))
        q_grid = np.load("%s%s/%s/q_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli))
        z_grid = np.load("%s%s/%s/z_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli))

        if Profil == False :

            data_convert = np.load("%s%s/%s/%s_data_convert_%i%i%i.npy"%(path,name_file,param_file,name_exo,reso_alt,\
                            reso_long,reso_lat))

        dx_grid,dx_grid_opt,order_grid,pdx_grid = dx_correspondance(p_grid,q_grid,z_grid,data_convert,x_step,r_step,\
                    theta_step,Rp,g0,h,t,reso_long,reso_lat,Middle,Integral,Discret,Gravity,Ord)

        np.save("%s%s/%s/dx_grid_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli),dx_grid)
        np.save("%s%s/%s/order_grid_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli),order_grid)

        if Discret == True :
             np.save("%s%s/%s/dx_grid_opt_%i_%i%i%i_%i_%.2f_%.2f.npy"
            %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli),dx_grid_opt)
        if Integral == True :
            np.save("%s%s/%s/pdx_grid_%i_%i%i%i_%i_%.2f_%.2f.npy"
            %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli),pdx_grid)

########################################################################################################################

    if Matrix == True :

        data_convert = np.load("%s%s/%s/%s_data_convert_%i%i%i.npy"%(path,name_file,param_file,name_exo,reso_alt,reso_long,\
                reso_lat))

        order_grid = np.load("%s%s/%s/order_grid_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,\
                reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))

        result = atmospheric_matrix_3D(order_grid,data_convert,t,Rp,c_species,Tracer,Cloudy)

        np.save("%s%s/%s/%s_P_%i%i%i_%i_%i_%.2f_%.2f.npy"%(path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,\
                t_selec,r_step,phi_rot,phi_obli),result[0])
        np.save("%s%s/%s/%s_T_%i%i%i_%i_%i_%.2f_%.2f.npy"%(path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,\
                t_selec,r_step,phi_rot,phi_obli),result[1])

        if Tracer == True :
            np.save("%s%s/%s/%s_Q_%i%i%i_%i_%i_%.2f_%.2f.npy"%\
                    (path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,phi_rot,phi_obli),\
                    result[2])
            np.save("%s%s/%s/%s_Cn_%i%i%i_%i_%i_%.2f_%.2f.npy"%\
                    (path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,phi_rot,phi_obli),\
                    result[3])
            if Cloudy == True :
                np.save("%s%s/%s/%s_gen_%i%i%i_%i_%i_%.2f_%.2f.npy"%\
                    (path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,phi_rot,phi_obli),\
                        result[4])
                np.save("%s%s/%s/%s_compo_%i%i%i_%i_%i_%.2f_%.2f.npy"%\
                        (path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,phi_rot,phi_obli),\
                            result[5])
            else :
                np.save("%s%s/%s/%s_compo_%i%i%i_%i_%i_%.2f_%.2f.npy"%\
                        (path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,phi_rot,phi_obli),\
                            result[4])
        else :
            np.save("%s%s/%s/%s_Cn_%i%i%i_%i_%i_%.2f_%.2f.npy"%\
                    (path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,phi_rot,phi_obli),\
                    result[2])
            if Cloudy == True :
                np.save("%s%s/%s/%s_gen_%i%i%i_%i_%i_%.2f_%.2f.npy"%\
                        (path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,phi_rot,phi_obli),\
                        result[3])
                np.save("%s%s/%s/%s_compo_%i%i%i_%i_%i_%.2f_%.2f.npy"%\
                        (path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,phi_rot,phi_obli),\
                        result[4])
            else :
                np.save("%s%s/%s/%s_compo_%i%i%i_%i_%i_%.2f_%.2f.npy"%\
                        (path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,phi_rot,phi_obli),\
                        result[3])

        del result,order_grid,data_convert

########################################################################################################################

    if Convert == True :

        if Kcorr == True :
            gauss = np.arange(0,dim_gauss,1)
            gauss_val = np.load("%s%s/gauss_sample.npy"%(path,name_source))
            P_sample = np.load("%s%s/P_sample.npy"%(path,name_source))
            T_sample = np.load("%s%s/T_sample.npy"%(path,name_source))
            if Tracer == True :
                Q_sample = np.load("%s%s/Q_sample.npy"%(path,name_source))
            else :
                Q_sample = np.array([])
            bande_sample = np.load("%s%s/bande_sample_%s.npy"%(path,name_source,domain))

            k_corr_data_grid = "%s%s/k_corr_%s_%s.npy"%(path,name_source,name_exo,domain)
        else :
            gauss = np.array([])
            gauss_val = np.array([])
            P_sample = np.load("%s%s/P_sample_%s.npy"%(path,name_source,source))
            T_sample = np.load("%s%s/T_sample_%s.npy"%(path,name_source,source))
            if Tracer == True :
                Q_sample = np.load("%s%s/Q_sample_%s.npy"%(path,name_source,source))
            else :
                Q_sample = np.array([])
            bande_sample = np.load("%s%s/bande_sample_%s.npy"%(path,name_source,source))

            k_corr_data_grid = "%s%s/crossection_%s.npy"%(path,name_source,source)

        # Telechargement des donnees CIA

        if Cont == True :
            k_cont = continuum()
        else :
            k_cont = np.array([])

        # Telechargement des donnees nuages

        if Cl == True :
            bande_cloud = np.load("%s%s/bande_cloud_%s.npy"%(path,name_source,name_exo))
            r_cloud = np.load("%s%s/radius_cloud_%s.npy"%(path,name_source,name_exo))
            cl_name = ''
            for i in range(c_species_name.size) :
                cl_name += '%s_'%(c_species_name[i])
            Q_cloud = "%s%s/Q_%s%s.npy"%(path,name_source,cl_name,name_exo)
            message_clouds = ''
            for i in range(c_species.size) :
                message_clouds += '%s (%.2f microns/%.3f)  '%(c_species[i],r_eff*10**6,rho_p[i]/1000.)
        else :
            bande_cloud = np.array([])
            r_cloud = np.array([])
            Q_cloud = np.array([])

########################################################################################################################

        P = np.load("%s%s/%s/%s_P_%i%i%i_%i_%i_%.2f_%.2f.npy"%(path,name_file,param_file,name_exo,reso_long,reso_lat,\
                reso_alt,t_selec,r_step,phi_rot,phi_obli))
        T = np.load("%s%s/%s/%s_T_%i%i%i_%i_%i_%.2f_%.2f.npy"%(path,name_file,param_file,name_exo,reso_long,reso_lat,\
                reso_alt,t_selec,r_step,phi_rot,phi_obli))
        if Tracer == True :
            Q = np.load("%s%s/%s/%s_Q_%i%i%i_%i_%i_%.2f_%.2f.npy"
                %(path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,phi_rot,phi_obli))
        else :
             Q = np.array([])
        if Cloudy == True :
            gen = np.load("%s%s/%s/%s_gen_%i%i%i_%i_%i_%.2f_%.2f.npy"
                %(path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,phi_rot,phi_obli))
        else :
            gen = np.array([])
        comp = np.load("%s%s/%s/%s_compo_%i%i%i_%i_%i_%.2f_%.2f.npy"
                %(path,name_file,param_file,name_exo,reso_long,reso_lat,reso_alt,t_selec,r_step,phi_rot,phi_obli))
        data_convert = np.load("%s%s/%s/%s_data_convert_%i%i%i.npy"%(path,name_file,param_file,name_exo,reso_alt,reso_long,\
                reso_lat))

########################################################################################################################

        direc = "%s/%s"%(name_file,opac_file)

        convertator (P,T,gen,c_species,Q,comp,ind_active,ind_cross,k_corr_data_grid,k_cont,\
                     Q_cloud,P_sample,T_sample,Q_sample,bande_sample,bande_cloud,x_step,r_eff,r_cloud,rho_p,direc,\
                     t,phi_rot,phi_obli,n_species,domain,ratio_HeH2,path,name_exo,reso_long,reso_lat,\
                     Tracer,Molecular,Cont,Cl,Scatt,Kcorr,Optimal)


########################################################################################################################





########################################################################################################################
########################################################################################################################
##########################################      TRANSFERT 3D      ######################################################
########################################################################################################################
########################################################################################################################

if Cylindric_transfert_3D == True :

    print 'Download of opacities data'

    if Isolated == False :
        if Kcorr == True :
            k_rmd = np.load("%s%s/%s/k_corr_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
            %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain))
            gauss_val = np.load("%s%s/gauss_sample.npy"%(path,name_source))
        else :
            if Optimal == True :
                k_rmd = np.load("%s%s/%s/k_cross_opt_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
            else :
                k_rmd = np.load("%s%s/%s/k_cross_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
            gauss_val = np.array([])
    else :
        if Kcorr == True :
            k_rmd = np.load("%s%s/%s/k_corr_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
            %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain))
            k_rmd = np.shape(k_rmd)
        else :
            k_rmd = np.load("%s%s/%s/k_cross_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
            %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
            k_rmd = np.shape(k_rmd)
        gauss_val = np.array([])

    if Continuum == True :
        if Kcorr == True :
            k_cont_rmd = np.load("%s%s/%s/k_cont_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
            %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain))
        else :
            k_cont_rmd = np.load("%s%s/%s/k_cont_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
            %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
    else :
        k_cont_rmd = np.array([])

    if Scattering == True :
        if Kcorr == True :
            k_sca_rmd = np.load("%s%s/%s/k_sca_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
            %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,domain))
        else :
            k_sca_rmd = np.load("%s%s/%s/k_sca_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
            %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
    else :
        k_sca_rmd = np.array([])

    if Clouds == True :
        if Kcorr == True :
            k_cloud_rmd = np.load("%s%s/%s/k_cloud_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%.2f_%s.npy" \
            %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,\
              r_eff*10**6,domain))
        else :
            k_cloud_rmd = np.load("%s%s/%s/k_cloud_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%.2f_%s.npy" \
            %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,r_eff*10**6,domain))
    else :
        k_cloud_rmd = np.array([])

########################################################################################################################

    order_grid = np.load("%s%s/%s/order_grid_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
    if Module == True :
        z_grid = np.load("%s%s/%s/z_grid_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
    else :
        z_grid = np.array([])

    if Discreet == True :
        dx_grid = np.load("%s%s/%s/dx_grid_opt_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
        pdx_grid = np.array([])

    else :

        pdx_grid = np.load("%s%s/%s/pdx_grid_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                       %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
        dx_grid = np.load("%s%s/%s/dx_grid_opt_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                      %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))

    data_convert = np.load("%s%s/%s/%s_data_convert_%i%i%i.npy"%(path,name_file,param_file,name_exo,reso_alt,reso_long,\
                reso_lat))

########################################################################################################################

    print('Download of couples array')

    if Kcorr == True :
        T_rmd = np.load("%s%s/%s/T_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,\
                  domain))
        P_rmd = np.load("%s%s/%s/P_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,\
                  domain))
        if Cl == True :
            gen_rmd = np.load("%s%s/%s/gen_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,\
                  domain))
        else :
            gen_rmd = np.array([])
        if Tracer == True :
            Q_rmd = np.load("%s%s/%s/Q_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,\
                  domain))
        else :
            Q_rmd = np.array([])
        rmind = np.load("%s%s/%s/rmind_%i%i_%s_%i_%i%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,dim_gauss-1,x_step,phi_rot,phi_obli,\
                  domain))
    else :
        T_rmd = np.load("%s%s/%s/T_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
        P_rmd = np.load("%s%s/%s/P_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
        if Cl == True :
            gen_rmd = np.load("%s%s/%s/gen_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
        else :
            gen_rmd = np.array([])
        if Tracer == True :
            Q_rmd = np.load("%s%s/%s/Q_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))
        else :
            Q_rmd = np.array([])
        rmind = np.load("%s%s/%s/rmind_%i%i_%s_%i_%i_%i_rmd_%.2f_%.2f_%s.npy"\
                %(path,name_file,opac_file,reso_long,reso_lat,name_exo,t,dim_bande,x_step,phi_rot,phi_obli,domain))

########################################################################################################################

    Itot = trans2fert3D (k_rmd,k_cont_rmd,k_sca_rmd,k_cloud_rmd,Rp,h,g0,r_step,theta_step,gauss_val,dim_bande,data_convert,\
                  P_rmd,T_rmd,Q_rmd,dx_grid,order_grid,pdx_grid,z_grid,t,\
                  name_file,n_species,Single,rmind,lim_alt,rupt_alt,\
                  Tracer,Continuum,Isolated,Scattering,Clouds,Kcorr,Rupt,Module,Integration,TimeSel)

    np.save(save_name_3D,Itot)

########################################################################################################################

if View == True :

    if Cylindric_transfert_3D == False :
        Itot = np.load('%s.npy'%(save_name_3D))
    if Kcorr == True :
        bande_sample = np.load("%s%s/bande_sample_%s.npy"%(path,name_source,domain))
    else :
        bande_sample = np.load("%s%s/bande_sample_%s.npy"%(path,name_source,source))

    R_eff_bar,R_eff,ratio_bar,ratR_bar,bande_bar,flux_bar,flux = atmospectre(Itot,bande_sample,Rs,Rp,r_step,0,\
                                                                                False,Kcorr,Middle)

    if Radius == True :
        plt.figure(1)
        plt.semilogx()
        plt.grid(True)
        plt.plot(1/(100.*bande_sample)*10**6,R_eff,'g',linewidth = 2,label='3D spectrum')
        plt.ylabel('Effective radius (m)')
        plt.xlabel('Wavelenght (micron)')
        plt.legend(loc=4)
        plt.draw()

    if Flux == True :
        plt.figure(2)
        plt.semilogx()
        plt.grid(True)
        plt.plot(1/(100.*bande_sample)*10**6,flux,'r',linewidth = 2,label='3D spectrum')
        plt.ylabel('Flux (Rp/Rs)2')
        plt.xlabel('Wavelenght (micron)')
        plt.legend(loc=4)
        plt.draw()

########################################################################################################################

print 'Pytmosph3R process finished with success'

########################################################################################################################

if Script == True :

    I = np.load('%s.npy'%(save_name_3D))
    save_adress = "/data1/caldas/Pytmosph3R/Tools/%s_real/"%(name_exo)
    if Composition == False :
        save_name_3D = "%s%s_3D_duo_linear_real_%i_%i_%i_%.2f"%(save_adress,name_exo,np.amin(T_iso_array),np.amax(T_iso_array),beta,P_tau/(1.e+5))
    else :
        save_name_3D = "%s%s_3D_duo_linear_real_%i_%i_%i_%.2f_eq"%(save_adress,name_exo,np.amin(T_iso_array),np.amax(T_iso_array),beta,P_tau/(1.e+5))
    if ErrOr == True :
        class star :
            def __init__(self):
                self.radius = Rs
                self.temperature = Ts
                self.distance = d_al
        bande_sample = np.load("%s%s/bande_sample_%s.npy"%(path,name_source,source))
        bande_sample = np.delete(bande_sample,[0])
        int_lambda = np.zeros((2,bande_sample.size))
        bande_sample = np.sort(bande_sample)

        for i_bande in range(bande_sample.size) :
            if i_bande == 0 :
                int_lambda[0,i_bande] = bande_sample[0]
                int_lambda[1,i_bande] = (bande_sample[i_bande+1]+bande_sample[i_bande])/2.
            elif i_bande == bande_sample.size - 1 :
                int_lambda[0,i_bande] = (bande_sample[i_bande-1]+bande_sample[i_bande])/2.
                int_lambda[1,i_bande] = bande_sample[bande_sample.size-1]
            else :
                int_lambda[0,i_bande] = (bande_sample[i_bande-1]+bande_sample[i_bande])/2.
                int_lambda[1,i_bande] = (bande_sample[i_bande+1]+bande_sample[i_bande])/2.
        int_lambda = np.sort(10000./int_lambda[::-1])
        noise = stellar_noise(star(),detection,int_lambda)
        noise = noise[::-1]
    flux_script(path,name_source,source,save_name_3D,I,noise,Rs,Rp,r_step,Kcorr,Middle,Noise)

########################################################################################################################
