from GJ1214b_script_seq import *



########################################################################################################################
########################################################################################################################
#######################################     ISOTHERM GENERATOR     #####################################################
########################################################################################################################
########################################################################################################################

if Isotherm_generator == True :

    message_species = ''
    message_ratio = ''
    for i in range(n_species.size) :
        message_species += '%s '%(n_species[i])
        message_ratio += '%.3f '%(x_ratio_species[i])
    print 'Atmosphere composition : %s'%(message_species)
    print 'Mixing ratio : %s'%(message_ratio)
    print 'Width of layers : %i m'%(delta_z)
    print 'Top of the atmosphere : %i m'%(h)
    print 'Mean radius of the exoplanet : %i m'%(Rp)
    print 'Mean surface gravity : %.2f m/s^2'%(g0)
    print 'Mean molar mass : %.12f kg/mol'%(M)

    z_array = np.arange(reso_alt/(delta_z/1000)+1)*float(delta_z)

    print 'Resolution of the GCM simulation (latitude/longitude) : %i/%i'%(reso_lat,reso_long)

########################################################################################################################

    if Cylind == True :

        p_grid,q_grid,z_grid = cylindric_assymatrix_parameter(Rp,h,alpha_step,delta_step,r_step,theta_step,theta_number,\
                                x_step,z_array,phi_rot,phi_obli,reso_long,reso_lat,Obliq,Middle)

        np.save("%s%s/1D/%s/q_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,reso_lat,\
                    reso_alt,r_step,phi_rot,phi_obli),q_grid)
        np.save("%s%s/1D/%s/p_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,reso_lat,\
                    reso_alt,r_step,phi_rot,phi_obli),p_grid)
        np.save("%s%s/1D/%s/z_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,reso_lat,\
                    reso_alt,r_step,phi_rot,phi_obli),z_grid)

    else :

        p_grid = np.load("%s%s/1D/%s/p_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli))
        z_grid = np.load("%s%s/1D/%s/z_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli))
        q_grid = np.load("%s%s/1D/%s/q_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli))

########################################################################################################################

    if Conv == True :

        dim = int(h/r_step) + 1

        data_convert = PTprofil1D(Rp,g0,M,P_surf,T_iso,n_species,x_ratio_species,r_step,delta_z,dim,number,Middle,Origin,Gravity)

        if Origin == True :
            np.save("%s%s/1D/%s/%s_data_convert_1D_%i%i%i.npy"%(path,name_file,param_file,name_exo,reso_alt,reso_long,\
                    reso_lat),data_convert)
        else :
            np.save("%s%s/1D/%s/%s_data_anal_convert_1D_%i%i%i.npy"%(path,name_file,param_file,name_exo,reso_alt,reso_long,\
                    reso_lat),data_convert)

    else :

        if Origin == True :
            data_convert = np.load("%s%s/1D/%s/%s_data_convert_1D_%i%i%i.npy"%(path,name_file,param_file,name_exo,\
                    reso_alt,reso_long,reso_lat))
        else :
            data_convert = np.load("%s%s/1D/%s/%s_data_anal_convert_1D_%i%i%i.npy"%(path,name_file,param_file,name_exo,\
                    reso_alt,reso_long,reso_lat))

########################################################################################################################

    if Correspondance == True :

        dim = int(h/r_step) + 1
        data = data_convert
        data_convert = np.zeros((number,1,dim,reso_lat+1,reso_long))
        for i in range(reso_lat+1) :
            for j in range(reso_long) :
                data_convert[:,:,:,i,j] = data[:,:,:,0,0]
        del data

        dx_grid,dx_grid_opt,order_grid,pdx_grid = dx_correspondance(p_grid,q_grid,z_grid,data_convert,x_step,r_step,\
                                    theta_step,Rp,g0,h,t,reso_long,reso_lat,Middle,Integ,Discr,Gravity,Order)

        np.save("%s%s/1D/%s/dx_grid_1D_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli),dx_grid)
        np.save("%s%s/1D/%s/order_grid_1D_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli),order_grid)

        if Discreet == True :
            if Origin == True :
                if Gravity == False :
                    np.save("%s%s/1D/%s/dx_grid_opt_1D_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                    %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli),\
                    dx_grid_opt)
                else :
                    np.save("%s%s/1D/%s/dx_grid_opt_g0_1D_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                    %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli),\
                    dx_grid_opt)
            else :
                if Gravity == False :
                    np.save("%s%s/1D/%s/dx_grid_opt_anal_1D_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                    %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli),\
                    dx_grid_opt)
                else :
                    np.save("%s%s/1D/%s/dx_grid_opt_anal_g0_1D_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                    %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli),\
                    dx_grid_opt)
        if Integral == True :
            if Origin == True :
                if Gravity == False :
                    np.save("%s%s/1D/%s/pdx_grid_1D_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                    %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli),\
                    pdx_grid)
                else :
                    np.save("%s%s/1D/%s/pdx_grid_g0_1D_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                    %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli),\
                    pdx_grid)
            else :
                if Gravity == False :
                    np.save("%s%s/1D/%s/pdx_grid_anal_1D_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                    %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli),\
                    pdx_grid)
                else :
                    np.save("%s%s/1D/%s/pdx_grid_anal_g0_1D_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                    %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli),\
                    pdx_grid)

########################################################################################################################





########################################################################################################################
########################################################################################################################
#######################################     1D GENERATOR     #####################################################
########################################################################################################################
########################################################################################################################

if D1_generator == True :

    message_species = ''
    for i in range(n_species.size) :
        message_species += '%s '%(n_species[i])
    print 'Atmosphere composition : %s'%(message_species)
    print 'Width of layers : %i m'%(delta_z)
    print 'Top of the atmosphere : %i m'%(h)
    print 'Mean radius of the exoplanet : %i m'%(Rp)
    print 'Mean surface gravity : %.2f m/s^2'%(g0)
    print 'Mean molar mass : %.12f kg/mol'%(M)

    z_array = np.arange(reso_alt/(delta_z/1000)+1)*float(delta_z)

    print 'Resolution of the GCM simulation (latitude/longitude) : %i/%i'%(reso_lat,reso_long)

########################################################################################################################

    if CylindM == True :

        p_grid,q_grid,z_grid = cylindric_assymatrix_parameter(Rp,h,alpha_step,delta_step,r_step,theta_step,theta_number,\
                                                          x_step,z_array,phi_rot,phi_obli,reso_long,reso_lat,Obliq,Middle)

        np.save("%s%s/1D/%s/q_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,reso_lat,\
                    reso_alt,r_step,phi_rot,phi_obli),q_grid)
        np.save("%s%s/1D/%s/p_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,reso_lat,\
                    reso_alt,r_step,phi_rot,phi_obli),p_grid)
        np.save("%s%s/1D/%s/z_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,reso_lat,\
                    reso_alt,r_step,phi_rot,phi_obli),z_grid)

    else :

        p_grid = np.load("%s%s/1D/%s/p_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli))
        z_grid = np.load("%s%s/1D/%s/z_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli))
        q_grid = np.load("%s%s/1D/%s/q_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                    reso_lat,reso_alt,r_step,phi_rot,phi_obli))


########################################################################################################################

    if CorrespM == True :

        data = np.load("%s%s/%s/%s_data_convert_%i%i%i.npy"%(path,name_file,param_file,name_exo,reso_alt,reso_long,\
                    reso_lat))
        dim = int(h/r_step) + 1
        data_convert = np.zeros((number,1,dim,reso_lat+1,reso_long))
        for i in range(reso_lat+1) :
            for j in range(reso_long) :
                data_convert[:,:,:,i,j] = data[:,:,:,lat,long]
        del data

        dx_grid,dx_grid_opt,order_grid,pdx_grid = dx_correspondance(p_grid,q_grid,z_grid,data_convert,x_step,r_step,\
                    theta_step,Rp,g0,h,t,reso_long,reso_lat,Middle,Integr,Discre,Grav,Ordered)

        np.save("%s%s/1D/%s/dx_grid_1D_%i_%i_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,long,lat,theta_number,\
                    reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli),dx_grid)
        np.save("%s%s/1D/%s/order_grid_1D_%i_%i_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,long,lat,theta_number,\
                    reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli),order_grid)

        if Discre == True :
            np.save("%s%s/1D/%s/dx_grid_opt_1D_%i_%i_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                        %(path,name_file,stitch_file,long,lat,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli),\
                        dx_grid_opt)
        if Integr == True :
            np.save("%s%s/1D/%s/pdx_grid_1D_%i_%i_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                        %(path,name_file,stitch_file,long,lat,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli),\
                        pdx_grid)

########################################################################################################################






########################################################################################################################
########################################################################################################################
##########################################        TRANSFERT 1D       ###################################################
########################################################################################################################
########################################################################################################################

if Cylindric_transfert_1D == True :

    if Kcorr == True :
        gauss = np.arange(0,dim_gauss,1)
        gauss_val = np.load("%s%s/gauss_sample.npy"%(path,name_source))
        P_sample = np.load("%s%s/P_sample.npy"%(path,name_source))
        T_sample = np.load("%s%s/T_sample.npy"%(path,name_source))
        if Marker == True :
            Q_sample = np.load("%s%s/Q_sample.npy"%(path,name_source))
        else :
            Q_sample = np.array([])
        bande_sample = np.load("%s%s/bande_sample_%s.npy"%(path,name_source,domain))

        k_corr_data_grid = np.load("%s%s/k_corr_%s_%s.npy"%(path,name_source,name_exo,domain))
    else :
        gauss = np.array([])
        gauss_val = np.array([])
        P_sample = np.load("%s%s/P_sample_%s.npy"%(path,name_source,source))
        T_sample = np.load("%s%s/T_sample_%s.npy"%(path,name_source,source))
        if Marker == True :
            Q_sample = np.load("%s%s/Q_sample_%s.npy"%(path,name_source,source))
        else :
            Q_sample = np.array([])
        bande_sample = np.load("%s%s/bande_sample_%s.npy"%(path,name_source,source))

        k_cross = np.load("%s%s/crossection_%s.npy"%(path,name_source,source))

    # Telechargement des donnees CIA

    if Continuum == True :
        k_cont_h2h2 = np.load("%s%s/k_cont_h2h2.npy"%(path,name_source))
        k_cont_h2he = np.load("%s%s/k_cont_h2he.npy"%(path,name_source))
        k_cont_nu = np.load("%s%s/K_cont_nu_h2h2.npy"%(path,name_source))
        T_cont = np.load("%s%s/T_cont_h2h2.npy"%(path,name_source))
    else :
        k_cont_h2h2 = np.array([])
        k_cont_h2he = np.array([])
        k_cont_nu = np.array([])
        T_cont = np.array([])

    # Telechargement des donnees nuages

    if Clouds == True :
        bande_cloud = np.load("%s%s/bande_cloud_%s.npy"%(path,name_source,name_exo))
        r_cloud = np.load("%s%s/radius_cloud_%s.npy"%(path,name_source,name_exo))
        cl_name = ''
        for i in range(c_species_name.size) :
            cl_name += '%s_'%(c_species_name[i])
        Q_cloud = np.load("%s%s/Q_%s%s.npy"%(path,name_source,cl_name,name_exo))
        message_clouds = ''
        for i in range(c_species.size) :
            message_clouds += '%s (%.2f microns/%.3f)  '%(c_species[i],r_eff*10**6,rho_p[i]/1000.)
    else :
        bande_cloud = np.array([])
        r_cloud = np.array([])
        Q_cloud = np.array([])

########################################################################################################################

    if Isotherm_generator == True :
        if Origin == True :
            data_convert = np.load("%s%s/1D/%s/%s_data_convert_1D_%i%i%i.npy"%(path,name_file,param_file,name_exo,reso_alt,\
                                                                               reso_long,reso_lat))
        else :
            data_convert = np.load("%s%s/1D/%s/%s_data_anal_convert_1D_%i%i%i.npy"%(path,name_file,param_file,name_exo,\
                                                                                    reso_alt,reso_long,reso_lat))
        order_grid = np.load("%s%s/1D/%s/order_grid_1D_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,theta_number,\
            reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
        if Module == True :
            z_grid = np.load("%s%s/1D/%s/z_grid_%i_%i%i%i_%i_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                                                                        reso_lat,reso_alt,r_step,phi_rot))
        else :
            z_grid = np.array([])
        if Origin == True :
            if Gravity == False :
                dx_grid = np.load("%s%s/1D/%s/dx_grid_opt_1D_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                                  %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
                pdx_grid = np.load("%s%s/1D/%s/pdx_grid_1D_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                                   %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
            else :
                dx_grid = np.load("%s%s/1D/%s/dx_grid_opt_g0_1D_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                                  %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
                pdx_grid = np.load("%s%s/1D/%s/pdx_grid_g0_1D_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                                   %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
        else :
            if Gravity == False :
                dx_grid = np.load("%s%s/1D/%s/dx_grid_opt_anal_1D_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                                  %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
                pdx_grid = np.load("%s%s/pdx_grid_anal_1D_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                                   %(path,name_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
            else :
                dx_grid = np.load("%s%s/1D/%s/dx_grid_opt_anal_g0_1D_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                                  %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
                pdx_grid = np.load("%s%s/1D/%s/pdx_grid_anal_g0_1D_%i_%i%i%i_%i_%.2f_%.2f.npy"\
                                   %(path,name_file,stitch_file,theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))

    if D1_generator == True :
        data = np.load("%s%s/%s/%s_data_convert_%i%i%i.npy"%(path,name_file,param_file,name_exo,reso_alt,reso_long,\
                                                             reso_lat))
        order_grid = np.load("%s%s/1D/%s/order_grid_1D_%i_%i_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,long,lat,\
                        theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
        if Module == True :
            z_grid = np.load("%s%s/1D/%s/z_grid_%i_%i%i%i_%i_%.2f.npy"%(path,name_file,stitch_file,theta_number,reso_long,\
                                                            reso_lat,reso_alt,r_step,phi_rot))
        else :
            z_grid = np.array([])
        dx_grid = np.load("%s%s/1D/%s/dx_grid_1D_%i_%i_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,long,lat,\
                        theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))
        pdx_grid = np.load("%s%s/1D/%s/pdx_grid_1D_%i_%i_%i_%i%i%i_%i_%.2f_%.2f.npy"%(path,name_file,stitch_file,long,lat,\
                         theta_number,reso_long,reso_lat,reso_alt,r_step,phi_rot,phi_obli))



########################################################################################################################
########################################################################################################################

    print 'Mean radius of the exoplanet : %i m'%(Rp)
    print 'Mean surface gravity : %.2f m/s^2'%(g0)
    print 'Roof of the atmosphere : %i m'%(h)
    if Clouds == True :
        print 'Clouds in the atmosphere (grain radius/density) : %s'%(message_clouds)

########################################################################################################################

    taille = np.shape(k_cross)
    #arr = np.zeros((n_species_active.size,taille[1],taille[2],taille[3]))
    #arr_0 = np.array([0])
    #arr_0 = np.append(arr_0,np.arange(5,3000*5,5))
    #for u in range(n_species_active.size) :
    #   for v in range(taille[1]) :
    #       for w in range(taille[2]) :
    #           arr[u,v,w,:] = arr_0
    #k_corr_data_grid = arr*6.42939126655e-26
    #del k_cross
    #k_corr_data_grid = np.ones((n_species_active.size,taille[1],taille[2],taille[3]))*6.42939126655e-26
    k_corr_data_grid = np.zeros((n_species_active.size,taille[1],taille[2],taille[3]))
    k_corr_data_grid[:,:,:,:] = k_cross[ind_cross,:,:,:]

########################################################################################################################

    data = data[:,0,:,:,:]
    if D3Maille == True :
        data_convert = np.zeros((number,h/r_step+1,reso_lat+1,reso_long))
        for i_lat in range(reso_lat+1) :
            for i_long in range(reso_long) :
                data_convert[:,:,i_lat,i_long] = data[:,:,lat,long]
        del data

    if lim_alt != h :

        data_convert = data_convert[:,0:z_lim,:,:]

    P_col, T_col = data_convert[0,:,lat,long], data_convert[1,:,lat,long]

    if Clouds == True :
        gen_col = data_convert[2+m_species.size:2+m_species.size+c_species.size,:,lat,long]
    else :
        gen_col = np.array([])
    if Marker == True :
        h2o_vap_col = data_convert[2,:,lat,long]
    else :
        h2o_vap_col = np.array([])
    compo_col = data_convert[2+m_species.size+c_species.size:number,:,lat,long]

    print 'Global radius of the exoplanet (with atmosphere) %i' %(Rp+lim_alt)

########################################################################################################################

    Itot = trans2fert1D (k_corr_data_grid,k_cont_h2h2,k_cont_h2he,k_cont_nu,T_cont,Q_cloud,Rp,h,g0,r_step,theta_step,\
                      x_step,gauss,gauss_val,dim_bande,data_convert,P_col,T_col,gen_col,h2o_vap_col,compo_col,ind_active,\
                      dx_grid,order_grid,pdx_grid,P_sample,T_sample,Q_sample,bande_sample,save_name_1D,n_species,c_species,\
                      Single,bande_cloud,r_eff,r_cloud,rho_p,t,phi_rot,domain,ratio_HeH2,lim_alt,rupt_alt,path,z_grid,type,\
                      Marker,Continuum,Isolated,Scattering,Clouds,Kcorr,Rupt,Middle,Integration,Module,Optimal,D3Maille)

    np.save(save_name_1D,Itot)

    plt.imshow(np.transpose(Itot[:,:,0], [1,0]),aspect='auto')
    plt.colorbar()
    plt.show()

########################################################################################################################