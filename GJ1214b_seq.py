from GJ1214b_script_seq import *
sys.path.append('/Users/caldas/Desktop/Pytmosph3R/PyRouts/')
from pytransfert import *

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

# Telechargement des sections efficaces ou des k_correles

if Parameters == True :

# Telechargement des parametres d'equilibre thermodynamique

    T_comp = np.load("%s%s/T_comp_%s.npy"%(path,name_source,name_exo))
    P_comp = np.load("%s%s/P_comp_%s.npy"%(path,name_source,name_exo))
    if Tracer == True :
        Q_comp = np.load("%s%s/Q_comp_%s.npy"%(path,name_source,name_exo))
    else :
        Q_comp = np.array([])
    X_species = np.load("%s%s/x_species_comp_%s.npy"%(path,name_source,name_exo))

########################################################################################################################

    if Profil == True :

        data_path = '%s%s'%(data_base,diag_file)
        if Layers == False :
            data_convert,h_top = Boxes(data_path,delta_z,Rp,h,P_h,t_selec,g0,M,number,T_comp,P_comp,Q_comp,n_species,X_species,\
                M_species,c_species,m_species,ratio_HeH2,Upper,TopPressure,Surf,Tracer,Cloudy,Middle,LogInterp,TimeSelec,MassAtm,NoH2)
        else :
            data_convert,h_top = NBoxes(data_path,n_layers,Rp,h,P_h,t_selec,g0,M,number,T_comp,P_comp,Q_comp,n_species,X_species,\
                M_species,c_species,m_species,ratio_HeH2,Upper,TopPressure,Surf,Tracer,Cloudy,Middle,LogInterp,TimeSelec,MassAtm,NoH2)

        if TopPressure != 'No' :
            h = h_top
            if Layers == True :
                reso_alt, delta_z, r_step, x_step = int(h/1000), h/np.float(n_layers), h/np.float(n_layers), h/np.float(n_layers)
            z_array = np.arange(h/np.float(delta_z)+1)*float(delta_z)
            save_name_3D = saving('3D',type,special,save_adress,version,name_exo,reso_long,reso_lat,t,h,dim_bande,dim_gauss,r_step,\
            phi_rot,r_eff,domain,stud,lim_alt,rupt_alt,long,lat,Discreet,Integration,Module,Optimal,Kcorr,False)

        np.save("%s%s/%s/%s_data_convert_%i%i%i.npy"%(path,name_file,param_file,name_exo,reso_alt,reso_long,reso_lat),\
                data_convert)

########################################################################################################################

    if Cylindre == True :

        p_grid,q_grid,z_grid = cylindric_assymatrix_parameter(Rp,h,alpha_step,delta_step,r_step,theta_step,theta_number,\
                                x_step,z_array,phi_rot,phi_obli,reso_long,reso_lat,Obliquity,Middle)

        np.save("%s%s/%s/p_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,reso_lat,\
                reso_alt,r_step,phi_rot,phi_obli),p_grid)
        np.save("%s%s/%s/q_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,reso_lat,\
                reso_alt,r_step,phi_rot,phi_obli),q_grid)
        np.save("%s%s/%s/z_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,reso_lat,\
                reso_alt,r_step,phi_rot,phi_obli),z_grid)


########################################################################################################################

    if Corr == True :

        if Cylindre == False :

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

        if Cylindre == False and Corr == False :

            p_grid = np.load("%s%s/%s/p_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli))
            q_grid = np.load("%s%s/%s/q_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli))
            z_grid = np.load("%s%s/%s/z_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli))

        if Profil == False :

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
            K_cont = continuum()
        else :
            K_cont = np.array([])

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

        direc = "%s/%s/"%(name_file,opac_file)

        convertator (P,T,gen,c_species,Q,comp,ind_active,ind_cross,k_corr_data_grid,K_cont,\
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
        plt.semilogx()
        plt.grid(True)
        plt.plot(1/(100.*bande_sample)*10**6,R_eff,'g',linewidth = 2,label='3D spectrum')
        plt.ylabel('Effective radius (m)')
        plt.xlabel('Wavelenght (micron)')
        plt.legend(loc=4)
        plt.show()

    if Flux == True :
        plt.semilogx()
        plt.grid(True)
        plt.plot(1/(100.*bande_sample)*10**6,flux,'r',linewidth = 2,label='3D spectrum')
        plt.ylabel('Flux (Rp/Rs)2')
        plt.xlabel('Wavelenght (micron)')
        plt.legend(loc=4)
        plt.show()

########################################################################################################################

print 'Pytmosph3R process finished with success'