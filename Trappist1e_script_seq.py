import sys
sys.path.append('/Users/caldas/Desktop/Pytmosph3R/PyRouts/')

from pytransfert import *

########################################################################################################################
########################################################################################################################

# Informations diverses sur l'etude

path = "/Users/caldas/Desktop/Pytmosph3R/"
name_file = "Files"
name_source = "Source"
name_exo = "Trappiste"
opac_file, param_file, stitch_file = 'Opacity', 'Parameters', 'Stitch'
version = 6.2

########################################################################################################################
########################################################################################################################

# Donnees de base

data_base, diag_file = "/Users/caldas/Bureautique/Simu/N2_CO2_H2O/", 'diagfi_0.1barN2_376ppmCO2'
reso_long, reso_lat = 64, 48
t, t_selec, phi_rot, phi_obli, inclinaison = 0, 94, 0.00, 0.00, 0.00
if inclinaison != 0. :
    phi_obli = np.abs(phi_obli+inclinaison-np.pi/2.)
Record = True

########################################################################################################################

# Proprietes de l'exoplanete

Rp = 0.0842213671668*R_J
Mp = 0.00197606301125*M_J
g0 = G*Mp/(Rp**2)

# Proprietes de l'etoile hote

Rs = 0.114*R_S
Ts = 2550.

# Proprietes en cas de lecture d'un diagfi

if data_base != '' :
    Rp, g0, reso_long, reso_lat = diag('%s%s'%(data_base,diag_file))
alpha_step, delta_step = 2*np.pi/np.float(reso_long), np.pi/np.float(reso_lat)

# Proprietes de l'atmosphere

n_species = np.array(['H2','He','H2O','N2','CO2'])
n_species_active = np.array(['H2O','CO2'])

# Proprietes de l'atmosphere isotherme

T_iso, P_surf = 500., 1.e+4
x_ratio_species_active = np.array([0.01,0.01,0.01,0.01,0.01])
M_species, M, x_ratio_species = ratio(n_species,x_ratio_species_active,IsoComp=False)

# Proprietes des nuages

c_species = np.array(['h2o_ice'])
c_species_name = np.array(['H2O'])
c_species_file = np.array(['icevis_n50','iceir_n50'])
rho_p = np.array([917.])
r_eff = 1.e-6

########################################################################################################################

# Crossection

n_species_cross = np.array(['H2O','CH4','NH3','CO','CO2'])
m_species = np.array(['h2o_vap'])
domain, domainn, source = "HR", "HR", "bin10"
dim_bande, dim_gauss = 3000, 16

# Selection des sections efficaces

ind_cross, ind_active = index_active (n_species,n_species_cross,n_species_active)

# Informations generale sur les donnees continuum

cont_tot = np.array(['H2O_CONT_SELF.dat','H2O_CONT_FOREIGN.dat'])
cont_species = np.array(['H2Os','H2O'])
cont_associations = np.array(['h2oh2o','h2ofor'])

########################################################################################################################

# Proprietes de maille

h, P_h = 5.970e+6, 1.e-3
delta_z, r_step, x_step, theta_number, n_layers = 3.0e+4, 3.0e+4, 3.0e+4, 96, 150
z_array = np.arange(h/np.float(delta_z)+1)*float(delta_z)
theta_step = 2*np.pi/np.float(theta_number)
Upper = "Isotherme"
number = 3 + n_species.size + m_species.size + c_species.size

# Choix dans la section de la maille

lim_alt, rupt_alt = 5.7e+6, 0.e+0
lat, long = 24, 47
z_lim = int(lim_alt/delta_z)
z_reso = int(h/delta_z) + 1

# En cas de modulation de la densite

type = np.array(['',0.00])

########################################################################################################################

# Importation des donnees GCM

if Record == True :

    class composition :
        def __init__(self):
            self.file = '%sSources/corrk_data/'%(data_base)
            self.parameters = np.array(['T','p','Q'])
            self.species = n_species
            self.ratio = ratio_HeH2
            self.renorm = np.array([])
    class aerosol :
        def __init__(self):
            self.number = c_species.size
            self.species = c_species
            self.nspecies = c_species_name
            self.file_name = c_species_file
            self.continuity = np.array([True])
    class continuum :
        def __init__(self) :
            self.number = cont_tot.size
            self.associations = cont_associations
            self.species = cont_species
            self.file_name = cont_tot
    class kcorr :
        def __init__(self) :
            self.resolution = '38x36'
            self.resolution_n = np.array([dim_bande,dim_gauss])
            self.type = np.array(['VI','IR'])
            self.parameters = np.array(['T','p','Q'])
            self.tracer = m_species
            self.exception = np.array([])
            self.jump = False
    class crossection :
        def __init__(self) :
            self.file = '/Users/caldas/Bureautique/Data_taurex/TauREx-develop/Input/xsec/10wno/'
            self.type = source
            self.species = n_species_cross
            self.type_ref = np.array(['xsecarr','wno','p','t'])

    data_record(path,name_source,data_base,name_exo,aerosol(),continuum(),kcorr(),crossection(),composition(),Renorm=False)
else :
    class continuum :
        def __init__(self) :
            self.number = cont_tot.size
            self.associations = cont_associations
            self.species = cont_species
            self.file_name = cont_tot

########################################################################################################################
########################################################################################################################

# Les options choisies lors de l'etude

Tracer = True          ###### S'il y a des traceurs
Cloudy = True          ###### S'il y a des nuages
Middle = True          ###### Construction de la maille sur le milieu des couches

########################################################################################################################

# Parameters

Parameters = True

Profil = True          ###### Reproduit la maille spherique en altitude
Layers = True          ###### Decoupage en nombre de couche
Surf = True            ###### Si des donnees de surface sont accessibles
LogInterp = False       ###### Interpolation de la pression via le logarithme
TopPressure = 'Down'      ###### Si nous voulons fixer le toit de l'atmosphere par rapport a une pression minimale

Cylindre = True        ###### Construit la maille cylindrique
Obliquity = False       ###### Si l'exoplanete est inclinee

Corr = True            ###### Traite les parcours optiques
Gravity = False         ###### Pour travailler a gravite constante
Discret = True         ###### Calcul les distances discretes
Integral = True        ###### Effectue l'integration sur les chemins optiques
Ord = False             ###### Si Discreet == False, Ord permet de calculer les indices avec l'integration

Matrix = True          ###### Transposition de la maille spherique dans la maille cylindrique

Convert = True         ###### Lance la fonction convertator qui assure l'interpolation des sections efficaces
Kcorr = False           ###### Sections efficaces ou k-correles
Molecular = True       ###### Effectue les calculs pour l'absorption moleculaire
Cont = True            ###### Effectue les calculs pour l'absorption par les collisions
Scatt = True           ###### Effectue les calculs pour l'absorption par diffusion Rayleigh
Cl = True              ###### Effectue les calculs pour l'absorption par les nuages
Optimal = False         ###### Interpolation optimal (Waldmann et al.)
TimeSelec = True       ###### Si nous etudions un temps precis de la simulation

########################################################################################################################

# Cylindric transfert

Cylindric_transfert_3D = True

Isolated = True        ###### Ne tiens pas compte de l'absorption moleculaire
Continuum = True       ###### Tiens compte de l'absorption par les collisions
Scattering = True      ###### Tiens compte de l'absorption par la diffusion
Clouds = True          ###### Tiens compte de l'absoprtion par les nuages
Single = "no"           ###### Isole une espece de nuage
Rupt = False            ###### Si l'atmosphere est tronquee
Discreet = True        ###### Calcul discret
Integration = False     ###### Calcul integral
Module = False          ###### Si nous souhaitons moduler la densite de reference

D3Maille = True        ###### Si nous voulons resoudre le transfert dans la maille 3D
TimeSel = True         ###### Si nous etudions un temps precis de la simulation

########################################################################################################################

# Plot

View = True

Radius = True          ###### Spectre rayon effective = f(longueur d'onde)
Flux = True            ###### Spectre flux = f(longueur d'onde)

########################################################################################################################
########################################################################################################################

# Sauvegardes

save_adress = "/Users/caldas/Desktop/Pytmosph3R/I/"
special = ''
stud = stud_type(r_eff,Single,Continuum,Isolated,Scattering,Clouds)
save_name_1D = saving('1D',type,special,save_adress,version,name_exo,reso_long,reso_lat,t,h,dim_bande,dim_gauss,r_step,\
            phi_rot,r_eff,domain,stud,lim_alt,rupt_alt,long,lat,Discreet,Integration,Module,Optimal,Kcorr,False)
save_name_3D = saving('3D',type,special,save_adress,version,name_exo,reso_long,reso_lat,t,h,dim_bande,dim_gauss,r_step,\
            phi_rot,r_eff,domain,stud,lim_alt,rupt_alt,long,lat,Discreet,Integration,Module,Optimal,Kcorr,False)

########################################################################################################################
