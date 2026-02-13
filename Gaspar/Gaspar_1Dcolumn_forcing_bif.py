from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import curve_fit
# import pyicon as pyic
# from netCDF4 import Dataset
# import sys
# import xarray as xr

import pyvmix_seperated_forcing as pyvmix
import gsw


# Initialize settings
# -------------------
S = pyvmix.Model()

# Modify default settings
# -----------------------
S.run = __file__.split('/')[-1][:-3]
S.path_data = f'/Users/nbruegge/work/pyvmix/{S.run}/'

S.savefig = False
S.path_fig = '../pics_nac/'
S.nnf=0

S.nz = 10
S.dz = 400*np.ones(S.nz) #4000 m in total

#S.fcor = 2*np.pi/86400 * np.sin(np.pi/180. * 45)
S.fcor = 0
S.conv_adj = True #True #toegevoegd: om convective adjustment aan te zetten

S.c_k = 0.1

S.kvAv = 'tke'

# S.deltaT = 100.
# S.nt = 24000*3600//S.deltaT
# S.lsave = 3600//S.deltaT

# used for forcing
S.deltaT = 5*3600*24
S.nt = 80*365*24*3600//S.deltaT
S.lsave = 20*3600*24//S.deltaT
#used for bifurcation diagram
# S.deltaT = 50*3600*24
# S.nt = 1500000*365*24*3600//S.deltaT
# S.lsave = 50*3600*24//S.deltaT

S.gamma = -0.0
S.fwf = S.gamma * (5/7.6) / (3600*24*365*5) # 25/38, then gamma should be similar to Den Toom; made per time step by scaling by delta T / 5 years.
#S.fwf = S.gamma * 5 * (25/38) / (3600*24*365*5) # 25/38, then gamma should be similar to Den Toom; made per time step by scaling by delta T / 5 years.

# Initialize grid and variables
# -----------------------------
S.initialize()

# Modify initial conditions
# -------------------------
#S.b = 5e-3*np.exp(S.zt/200.)
#S.b = 0.*S.zt
dTdz = 0.5
T_init = 20. + dTdz*S.zt
# plt.plot(T_init, S.zt)
# plt.show()

# rho_init = S.rho0*(1 - S.alpha*(T_init - S.T0))
# S.b = -S.grav*(rho_init - S.rho0)/S.rho0

##INITIAL CONDITIONS USING BOUYANCY
#S.b = S.grav*S.tAlpha*(T_init - S.T0)
#S.b = 0.*T_init

#S.b = 1e-4*S.zt/100.

# Modify forcing
# --------------
# wind forcing
#S.taux0 = 1.5e-4
#Tsurf = -1e-4
Tsurf = 0.
S.Qsurf0 =  S.rho0 * S.cp * S.dz[0]* Tsurf  # W/m2

# S.uair = 3.
# S.vair = 0.

#RELAXATION TOWARDS A PROFILE: IMPORTANT FOR LATERAL FORCING
print(S.zt)
T_relax = 15 - 5*np.cos(2*np.pi*S.zt/(S.nz*S.dz[1]))
#plt.plot(T_relax, S.zt) #klopt
# plt.plot(T_relax, S.zt)

S.temp0 = T_relax
S.lam_T = 1./(3600*24*365*5)

S.S_forcing = S.fwf*np.cos(np.pi*S.zt/(S.nz*S.dz[1]))
# plt.plot(S.S_forcing, S.zt)
# plt.show()

S.sal0 = S.S_forcing



S.temp = T_relax

S_init = 35.0 * np.ones(S.nz)
S.sal = S_init

S.sal_relaxation = False
S.lam_S = 1./(3600*24*365*5)
if S.sal_relaxation:
   S.sal0 = S_init + S.gamma * np.cos(np.pi*S.zt/(S.nz*S.dz[1]))
   S.sal = S_init + S.gamma * np.cos(np.pi*S.zt/(S.nz*S.dz[1]))
#S.ylim_hov = [-25, 0]

# def wind_forcing(S, time):
#   #taux = S.taux0
#   #tauy = 0.
#   usurf = S.uvel[0]
#   vsurf = S.vvel[0]
#   taux = (
#     S.rho_air * S.cdrag / S.rho0
#     * np.sqrt((S.uair-usurf)**2+(S.vair-vsurf)**2)
#     * (S.uair-usurf)
#     )
#   tauy = (
#     S.rho_air * S.cdrag / S.rho0
#     * np.sqrt((S.uair-usurf)**2+(S.vair-vsurf)**2)
#     * (S.vair-vsurf)
#     )
#   return taux, tauy
# S.wind_forcing = wind_forcing

def no_wind(S, time):
    # return zero wind stress
    return 0.0, 0.0

S.wind_forcing = no_wind


######## HEAT FORCING FROM THE SURFACE ##########################
#add constant heat flux at surface
Q0 = -300.0  # W/m^2 constant cooling

#seasonal cycle
Qamp = 200.0  # W/m^2
Qmean = -50.0  # W/m^2
year_seconds = 3600*24*365

#seasonal cycle from Tatmosphere
year_seconds = 3600*24*365
Tatm_mean = 5.0 # °C, mean atmospheric temperature,  set to 0 at the start
Tatm_amp = 0.0  # °C #set to zero for constant atmospheric temperature, can be increased for seasonal cycle (10)
def seasonal_Tatm(time):
    Tatm = Tatm_mean + Tatm_amp * np.cos(2*np.pi*time/year_seconds)
    return Tatm
def bulk_heat_flux(S, Tatm, Tsfc):
    return (
        S.rho_air
        * S.cp_air
        * S.cdrag
        * S.uair
        * (Tatm - Tsfc)
    )

#climate ramp for Tatmosphere
dTatm_2100 = 3.0     # total warming [°C]
ramp_start_year = 2020
ramp_end_year   = 2100
start_year      = 2020   # model start year

def climate_ramp(time):
    """
    Smooth cosine ramp from 0 to 1
    """
    t_years = start_year + time / year_seconds
    tau = (t_years - ramp_start_year) / (ramp_end_year - ramp_start_year)
    tau = np.clip(tau, 0.0, 1.0)

    return 0.5 * (1.0 - np.cos(np.pi * tau))


#stochastic NAO forcing
S.Qnao = 0.0
tau_NAO = 2 * 365 * 24 * 3600  # correlation time scale of NAO in seconds, 2 years
sigma_NAO = 30.0 / np.sqrt(365*24*3600) # standard deviation of NAO forcing in W/m^2
S.Qextra = 0.0 #extra heat flux to explore bifurcation diagram, can be positive or negative

def winter_weight(time):
    doy = (time / 86400.0) % 365
    w = np.cos(2*np.pi*(doy-15)/365)
    return max(0.0, w)

def buoyancy_forcing(S, time):
  #add constant heat flux at surface
  #Qsurf = Q0
  #seasonal cycle
  #Qseasonal = Qmean + Qamp * np.cos(2*np.pi*time/year_seconds)
  #seasonal cycle from Tatmosphere
  Tsfc = S.temp[0]           # surface temperature
  Tatm = seasonal_Tatm(time)
  #add climate ramp
  # Tatm += dTatm_2100 * climate_ramp(time)

  Qseasonal = bulk_heat_flux(S, Tatm, Tsfc)
  # stochastic NAO forcing
  dt = S.deltaT
  dW = np.sqrt(dt) * np.random.randn()
  S.Qnao += (-S.Qnao / tau_NAO) * dt + sigma_NAO * dW
  #winter weighting
  S.Qnao_eff = winter_weight(time) * S.Qnao
#   Qsurf = S.Qsurf0*np.sin(2*np.pi*time/86400.)
#   Qcut = -300
#   if Qsurf<Qcut:
#     Qsurf = Qcut
  Qsurf = Qseasonal + S.Qnao_eff
  #Qsurf = 0. # heat flux uitschakelen voor nu !!!!!!!!!!!!!!!
  #Qsurf = Qseasonal + S.Qextra #voor bif diagram
  return Qsurf #changed to get the constant, should be Qsurf
S.buoyancy_forcing = buoyancy_forcing

######## END OF HEAT FORCING FROM THE SURFACE ##########################


######## FRESHWATER FORCING FROM THE SURFACE ##########################

#constant freshwater flux
Fw_constant = 0.1e-6 #m/s

#seasonal freshwater flux
def seasonal_pulse(time):
    """
    Summer pulse around day 230 (August), Gaussian shape
    """
    doy = (time / 86400.0) % 365
    pulse = np.exp(-0.5 * ((doy - 230)/35)**2)
    return pulse
Fw_ampl = 0.2e-6  # m/s

#climate ramp for freshwater flux
dTfw_2100 = 0.8 #increase by 80 percent by 2100

def freshwater_forcing(S, time):
    #constant freshwater flux at surface
    FwFlux = Fw_constant

    # #seasonal pulse
    # FwFlux += Fw_ampl * seasonal_pulse(time)

    # #add climate ramp
    # FwFlux *= 1.0 + dTfw_2100 * climate_ramp(time)
    #FwFlux = 0. # freshwater flux uitschakelen voor nu !!!!!!!!!!!!!!!
    return FwFlux
S.freshwater_forcing = freshwater_forcing

######## END OF FRESHWATER FORCING FROM THE SURFACE ##########################




#find equilibrium
S.find_equilibrium = False
S.eq_tol = 1e-6


# Run the model
# -------------
# to do a transient noise simulation, set S.find_equilibrium = False
#load the equilibrium data to continue from there

data = np.load("data/Gaspar/flux_surfforc_gamma/bifurcation_equilibria_data.npz")

S.gamma = data["gamma"][39]
S.temp  = data["temp"][39,:]
S.sal   = data["sal"][39,:]
S.measure_convection = data["convection"][39]

#salinity forcing
# S.sal0 = S_init + S.gamma * np.cos(np.pi*S.zt/(S.nz*S.dz[1])) #change salinity relaxation accordingly

#salinity flux
S.fwf = S.gamma * (5/7.6) / (3600*24*365*5) # 25/38, then gamma should be similar to Den Toom; made per time step by scaling by delta T / 5 years.
S.S_forcing = S.fwf*np.cos(np.pi*S.zt/(S.nz*S.dz[1]))
S.sal0 = S.S_forcing

# plt.plot((S.temp - 20)*2e-4 , S.zt)
# plt.plot((S.sal - 35)*7.6e-4, S.zt)
# plt.show()

S.run_model()



# ### TO MAKE BIFURCATION DIAGRAM ################################################
# ################################################################################

#reload previous data to start from there
# data = np.load("data/Gaspar/relax_surfforc_gamma/bifurcation_equilibria_data.npz")

# S.gamma = data["gamma"][14]
# S.temp  = data["temp"][14,:]
# S.sal   = data["sal"][14,:]
# S.measure_convection = data["convection"][14]


# number_of_steps = 50
# dgamma = -0.01 #-0.05
# gamma_values = []
# convection_measures = []
# gamma_values.append(S.gamma)
# convection_measures.append(S.measure_convection)
# temp_eq = []
# sal_eq = []
# temp_eq.append(S.temp.copy())   # copy is important!
# sal_eq.append(S.sal.copy())

# for i in range(number_of_steps):
#     print(f'Running step {i+1} of {number_of_steps}')
#     S.found_solution = False
#     S.gamma = S.gamma + dgamma
#     gamma_values.append(S.gamma)

#     # salinity flux
#     S.fwf = S.gamma * (5/7.6) / (3600*24*365*5) # 25/38, then gamma should be similar to Den Toom; made per time step by scaling by delta T / 5 years.
#     S.S_forcing = S.fwf*np.cos(np.pi*S.zt/(S.nz*S.dz[1]))
#     S.sal0 = S.S_forcing

#     #salinity relaxation
#     # S.sal0 = S_init + S.gamma * np.cos(np.pi*S.zt/(S.nz*S.dz[1]))

#     S.run_model()
#     convection_measures.append(S.measure_convection)
#     print("hi")
#     temp_eq.append(S.temp.copy())   # copy is important!
#     sal_eq.append(S.sal.copy())

#     if S.found_solution == False:
#         print(f'No solution found for gamma = {S.gamma}')
#         break
    
#     print("hello, plotting")
#     plt.figure()
#     plt.plot((S.temp - 20)*2e-4, S.zt/4000, label='T equilibrium')
#     plt.plot((S.sal-35)*7.6e-4, S.zt/4000, label='S equilibrium')
#     rho = S.rho0 * (1 - S.b/S.grav)
#     plt.plot((rho - S.rho0)/S.rho0, S.zt/4000, label='rho equilibrium')
#     # plt.plot(S.b_T*(1+S.g_2*S.pressure)*(S.temp - S.T0nl), S.zt/4000, label='T contribution')
#     # plt.plot(S.b_S*(S.sal - S.S0nl), S.zt/4000, label='S contribution')
#     # plt.plot((S.b_T2/2)*(S.temp - S.T0nl)**2, S.zt/4000, label='T2 contribution')
#     # plt.plot((-S.b_T*(1+S.g_2*S.pressure)*(S.temp - S.T0nl) - (S.b_T2/2)*(S.temp - S.T0nl)**2 + S.b_S*(S.sal - S.S0nl)), S.zt/4000, label='rho nondimensional')

#     plt.xlabel("Value at equilibrium")
#     plt.ylabel("Depth")
#     plt.title(f"Equilibrium profiles, gamma = {round(S.gamma, 2)}, {i}")
#     plt.ylim(-1,0)
#     #plt.xlim(-max(abs(S.sal-35)*7.6/5),max(abs(S.sal-35)*7.6/5))
#     plt.legend()
#     plt.savefig(f"data/Gaspar/flux_surfforc_gamma/Gaspar_equilibrium_profiles_gamma{round(S.gamma, 2)}.png")
#     print("plotted")

# plt.figure()
# plt.plot(gamma_values, convection_measures)
# plt.savefig('data/Gaspar/flux_surfforc_gamma/bif_Gaspar.png', dpi=200)


# np.savez(
#     "data/Gaspar/flux_surfforc_gamma/bifurcation_equilibria_data.npz",
#     gamma=np.array(gamma_values),
#     temp=np.array(temp_eq),
#     sal=np.array(sal_eq),
#     convection=np.array(convection_measures),
#     zt=S.zt
# )


# BIFURCATION OVER S.Qextra

# number_of_steps = 30
# dQextra = -1.
# Qextra_values = []
# convection_measures = []
# Qextra_values.append(0.)
# convection_measures.append(S.measure_convection)
# temp_eq = []
# sal_eq = []
# temp_eq.append(S.temp.copy())   # copy is important!
# sal_eq.append(S.sal.copy())


# for i in range(number_of_steps):
#     print(f'Running step {i+1} of {number_of_steps}')
#     S.found_solution = False
#     S.Qextra = S.Qextra + dQextra
#     Qextra_values.append(S.Qextra)

#     # salinity flux
#     S.fwf = S.gamma * (5/7.6) / (3600*24*365*5) # 25/38, then gamma should be similar to Den Toom; made per time step by scaling by delta T / 5 years.
#     S.S_forcing = S.fwf*np.cos(np.pi*S.zt/(S.nz*S.dz[1]))
#     S.sal0 = S.S_forcing

#     #salinity relaxation
#     #S.sal0 = S_init + S.gamma * np.cos(np.pi*S.zt/(S.nz*S.dz[1]))

#     S.run_model()
#     convection_measures.append(S.measure_convection)
#     print("hi")
#     temp_eq.append(S.temp.copy())   # copy is important!
#     sal_eq.append(S.sal.copy())

#     if S.found_solution == False:
#         print(f'No solution found for gamma = {S.gamma}')
#         break
    
#     print("hello, plotting")
#     plt.figure()
#     plt.plot((S.temp - 20)*2e-4, S.zt/4000, label='T equilibrium')
#     plt.plot((S.sal-35)*7.6e-4, S.zt/4000, label='S equilibrium')
#     rho = S.rho0 * (1 - S.b/S.grav)
#     plt.plot((rho - S.rho0)/S.rho0, S.zt/4000, label='rho equilibrium')
#     # plt.plot(S.b_T*(1+S.g_2*S.pressure)*(S.temp - S.T0nl), S.zt/4000, label='T contribution')
#     # plt.plot(S.b_S*(S.sal - S.S0nl), S.zt/4000, label='S contribution')
#     # plt.plot((S.b_T2/2)*(S.temp - S.T0nl)**2, S.zt/4000, label='T2 contribution')
#     # plt.plot((-S.b_T*(1+S.g_2*S.pressure)*(S.temp - S.T0nl) - (S.b_T2/2)*(S.temp - S.T0nl)**2 + S.b_S*(S.sal - S.S0nl)), S.zt/4000, label='rho nondimensional')

#     plt.xlabel("Value at equilibrium")
#     plt.ylabel("Depth")
#     plt.title(f"Equilibrium profiles, Qextra = {round(S.Qextra, 2)}")
#     plt.ylim(-1,0)
#     #plt.xlim(-max(abs(S.sal-35)*7.6/5),max(abs(S.sal-35)*7.6/5))
#     plt.legend()
#     plt.savefig(f"data/Gaspar/flux_surfforc_gamma/Gaspar_equilibrium_profiles_Qextra{round(S.Qextra, 2)}.png")
#     print("plotted")

# plt.figure()
# plt.plot(Qextra_values, convection_measures)
# plt.savefig('data/Gaspar/flux_surfforc_gamma/bif_Gaspar_Qextra_neg.png', dpi=200)

# np.savez(
#     "data/Gaspar/flux_surfforc_gamma/bifurcation_equilibria_data_Qextra_neg.npz",
#     Qextra=np.array(Qextra_values),
#     temp=np.array(temp_eq),
#     sal=np.array(sal_eq),
#     convection=np.array(convection_measures),
#     zt=S.zt
# )

################################################################################
################################################################################

# for i in range(len(gamma_values)):
#     print(i, gamma_values[i], convection_measures[i])


# print(S.gamma, S.measure_convection)
# S.gamma = -0.00000001
# S.fwf = S.gamma * (25/38) * S.deltaT / (3600*24*365*5) # 25/38, then gamma should be similar to Den Toom; made per time step by scaling by delta T / 5 years.
# S.S_forcing = S.fwf*np.cos(np.pi*S.zt/(S.zt[-1]-S.zt[0]))
# S.sal0 = S.S_forcing
# S.run_model()
# print(S.gamma, S.measure_convection)


# WHEN ON MAC I CAN PROBABLY USE THIS TO PLOT DIRECTLY
# # Visualize results
# # -----------------
# plt.close('all')

# exec(open('./extract_parameters/plot_defaults.py').read())

# plt.show()

#PLOT FOR NOW

# print(dir(S))

# # print(S.kv_s[20])
# # print(S.b)

#Time corresponding to snapshots (assuming each snapshot = deltaT * lsave)
time_save = (np.arange(S.b_s.shape[0]) * S.deltaT * S.lsave / 3600) + (S.deltaT * S.lsave /3600)  # in hours #one step shifted, ic not saved
time_sec = time_save * 3600

# T_s = S.temp_s
# S_s = S.sal_s

T_s = (S.temp_s - 20)*2e-4 # to nondimensionalize
S_s = (S.sal_s - 35)*7.6e-4 # to nondimensionalize

# #temperature to temperature as in Den Toom
# Tref_DT = 15
# Sref_DT = 35

# T_sDT = (T_s - Tref_DT) / 5 
# T_sG = (T_s - S.T0) / 5
# S_sDT = (S_s - Sref_DT) * 7.6 / 5
# rho_sDT = -T_sDT + S_sDT #rho DT
# rho_sG = -2*T_sG + S_sDT #rho Gaspar, because alpha_T is 2*


# plt.plot(time_save, S_s[:,0])
# print(S_s[-1,0])
# idx = np.where(S_s[:,5] >= 15)[0][0]
# time_at_15 = time_save[idx]

# print("time at time_at_15 =", time_at_15)



# plt.xlabel = "Time (h)"
# plt.ylabel = "Temperature at mid-depth [°C]"
# plt.title("Temperature relaxation at mid-depth")
# plt.savefig("extract_parameters/Gaspar_temperaturerelaxation_middepth.png")
# plt.xlabel = "Time (h)"
# plt.ylabel = "Salinity at top [psu]"
# plt.title("Salinity flux at top")
# plt.savefig("extract_parameters/Gaspar_salinityflux_top.png")

#T_s = S.b_s  #directly temperature

# # Fit surface temperature cooling rate
# def m_sst(x, a):
#     return 20.0 - a * np.sqrt(x)
# popt, pcov = curve_fit(m_sst, time_sec, T_s[:,0], p0=[0.01], bounds=(0, np.inf))
# a_sst = popt[0]
# sst_fit = m_sst(time_sec, a_sst)

# # Plot surface temperature over time
# plt.plot(time_save, T_s[:, 0])  # surface is the first index
# plt.plot(time_save, sst_fit, '--', label='fit')
# plt.xlabel('Time [hours]')
# plt.ylabel('Surface temperature [°C]')
# plt.savefig("extract_parameters/Gaspar_sst.png", dpi=200)

# print(f'Fitted cooling rate a_sst = {a_sst:.5f}')




# #MLD calculation
# MLD = []

# for u in S.b_s:
#     rho = S.rho0 - u / (S.grav * S.tAlpha)  # convert buoyancy back to density
#     depth = 0.0
#     for k in range(S.nz - 1):
#         if rho[k] < rho[k + 1]:  # density decreases downward. As soon as it increases, MLD is reached
#             depth = abs(S.zt[k] - S.zt[0])
#             break
#     MLD.append(depth)

# MLD = np.array(MLD)

# def m_mld(x, a):
#     return a* np.sqrt(x)

# popt, pcov = curve_fit(m_mld, time_sec, MLD, p0=[0.01], bounds=(0, np.inf))
# a_mld = popt[0]
# mld_fit = m_mld(time_sec, a_mld)

# plt.figure()
# plt.plot(time_save, MLD, label='MLD data')
# plt.plot(time_save, mld_fit, ls='--', label='fit')
# plt.xlabel('Time [hours]')
# plt.ylabel('Mixed Layer Depth [m]')
# plt.legend()
# plt.savefig("extract_parameters/gaspar_mld_fit.png", dpi=200)

# print(f"MLD fit: a1 = {a_mld}")



#PRINT ALL SURFACE FLUXES AND DENSITY PROFILES
plt.close()
# Optional: calculate time array for snapshots
time_save = np.arange(T_s.shape[0]) * S.deltaT * S.lsave / 3600  # in hours

# --- Create the figure ---
fig, ax = plt.subplots()
# line_T_DT, = ax.plot([], [], color='green', label='T DT')
# line_S_DT, = ax.plot([], [], color='orange', label='S DT')
# line_rho_G, = ax.plot([], [], color='blue', ls='--', label='rho G')
# line_rho_DT, = ax.plot([], [], color='blue', label='rho DT')
# line_T_G, = ax.plot([], [], color='green', ls='--', label='T G')
line_T, = ax.plot([], [], color='purple', label='T')
line_S, = ax.plot([], [], color='purple', label='S')

ax.set_xlim(-0.1, 1.0)
ax.set_ylim(S.zt[-1], S.zt[0])  # surface at top
ax.set_xlabel('Salinity [psu]')
ax.set_ylabel('Depth [m]')

def init():
    # line_T_DT = ax.plot([], [], color='green', label='T DT')
    # line_S_DT = ax.plot([], [], color='orange', label='S DT')
    # line_rho_G = ax.plot([], [], color='blue', label='rho G')
    # line_rho_DT = ax.plot([], [], color='blue', ls='--', label='rho DT')
    # line_T_G = ax.plot([], [], color='green', ls='--', label='T G')
    line_T = ax.plot([], [], color='purple', label='T')
    line_S = ax.plot([], [], color='purple', label='S')
    return (
            # line_T_DT, 
            # line_S_DT, 
            # line_rho_G, 
            # line_rho_DT, 
            # line_T_G,
            line_T,
            line_S
            )

def update(frame):
    # line_T_DT.set_data(T_sDT[frame], S.zt)
    # line_S_DT.set_data(S_sDT[frame], S.zt)
    # line_rho_G.set_data(rho_sG[frame], S.zt)
    # line_rho_DT.set_data(rho_sDT[frame], S.zt)
    # line_T_G.set_data(T_sG[frame], S.zt)
    line_T.set_data(T_s[frame], S.zt)
    line_S.set_data(S_s[frame], S.zt)

    ax.set_title(f'Time = {time_save[frame]:.1f} h')
    return (
        # line_T_DT,
        # line_S_DT,
        # line_rho_G,
        # line_rho_DT,
        # line_T_G,
        line_T,
        line_S
    )

# --- Create animation ---
ax.legend(loc='best', frameon=True)
anim = FuncAnimation(fig, update, frames=T_s.shape[0],
                     init_func=init, blit=False, interval=200)

anim.save(f'data/Gaspar/flux_surfforc_gamma/NAOforcing/Gaspar_cooling_seasonal_gamma{S.gamma}.gif')
plt.show()

# plt.plot(time_save, S.Qsurf_ts)
# plt.xlabel('Time [hours]')
# plt.ylabel('Surface buoyancy flux [W/m^2]') 
# plt.savefig(f'data/Gaspar/forcing/Gaspar_surface_buoyancy_flux_seasonal_stochastic_winter_gamma{S.gamma}.png')

# plt.figure()
# plt.plot(time_save, S.Qnao_ts)
# plt.xlabel('Time [hours]')
# plt.ylabel('Surface buoyancy flux [W/m^2]') 
# plt.savefig(f'data/Gaspar/forcing/Gaspar_surface_buoyancy_flux_stochastic_gamma{S.gamma}.png')

# plt.figure()
# plt.plot(time_save, S.Qnaoeff_ts)
# plt.xlabel('Time [hours]')
# plt.ylabel('Surface buoyancy flux [W/m^2]') 
# plt.savefig(f'data/Gaspar/relax_surfforc_gamma/NAOforcing/Gaspar_NAO.png')


##############################TRANSIENT PLOT NAO EFFECTIVE HEAT FLUX AND CONVECTIVE ADJUSTMENT MEASURE ########################
fig, ax1 = plt.subplots()

# --- First axis: NAO-effective heat flux ---
ax1.plot(time_save, S.Qnaoeff_ts, color='tab:blue', label='NAO-effective heat flux')
ax1.set_xlabel('Time [hours]')
ax1.set_ylabel('Surface buoyancy flux [W/m$^2$]', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# --- Second axis: convective adjustment measure ---
ax2 = ax1.twinx()
ax2.plot(time_save, S.measure_convection_ts, color='tab:red', label='Convective adjustment')
ax2.set_ylabel('Convective adjustment measure', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# --- Combined legend ---
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')

plt.title('NAO forcing and convective adjustment')
plt.tight_layout()
plt.savefig(
    f'data/Gaspar/flux_surfforc_gamma/NAOforcing/Gaspar_NAO_conv_0.png',
    dpi=200
)
# plt.show()

# Tclim = dTatm_2100 * np.array([climate_ramp(t) for t in time_sec])
# plt.figure()
# plt.plot(time_sec / year_seconds, Tclim)
# plt.xlabel("Year")
# plt.ylabel("ΔTatm_climate [°C]")
# plt.title("Climate ramp contribution to atmospheric temperature")
# plt.savefig(f'data/Gaspar/forcing/Gaspar_climate_ramp_gamma{S.gamma}.png')

# plt.figure()
# plt.plot(time_save, S.FwFlux_ts)
# plt.xlabel('Time [hours]')
# plt.ylabel('Surface freshwater flux [m/s]') 
# plt.savefig(f'data/Gaspar/forcing/Gaspar_surface_fw_flux_constant_gamma{S.gamma}.png')


####PLOT EQUILIBRIUM PROFILES ##############################
###############################IN DT STANDARDS##########################

#plot single equilibrium profile to see if there is convergence to equilibrium
# plt.figure()
# plt.plot(time_save, S.temp_s[:,-1], label='T equilibrium', color='purple')
# plt.savefig(f"Gaspar/Gaspar_evolution_bottom_temperature_gamma{S.gamma}.png")
# plt.figure()
# plt.plot(time_save, S.sal_s[:,-1], label='S equilibrium', color='blue')
# plt.savefig(f"Gaspar/Gaspar_evolution_bottom_salinity_gamma{S.gamma}.png")  

# if S.found_solution == False:
#     print("No equilibrium found, so no equilibrium profiles to plot.")
#     exit()
# plt.figure()
# pressure = gsw.p_from_z(S.zt, lat=57)
# plt.plot(S.b_T*(1+S.g_2*pressure)*(S.temp - S.T0nl), S.zt/4000, label='T contribution')
# plt.plot(S.b_S*(S.sal - S.S0nl), S.zt/4000, label='S contribution')
# plt.plot((S.b_T2/2)*(S.temp - S.T0nl)**2, S.zt/4000, label='T2 contribution')
# plt.plot((-S.b_T*(1+S.g_2*pressure)*(S.temp - S.T0nl) - (S.b_T2/2)*(S.temp - S.T0nl)**2 + S.b_S*(S.sal - S.S0nl)), S.zt/4000, label='rho nondimensional')
# plt.xlabel = "Value at equilibrium"
# plt.ylabel = "Depth"
# plt.title(f"Equilibrium profiles, gamma = {S.gamma}")
# plt.ylim(-1,0)
# plt.legend()
# #plt.savefig(f"Gaspar/Gaspar_equilibrium_profiles_gamma{S.gamma}.png")
# plt.show()

# # print('S.nt, S.lsave, nsave, S.time[:5] =', S.nt, S.lsave, S.nsave, getattr(S,'time',None)[:5])




#### CHECK TKE EVOLUTION ##############
# plt.close()

# plt.figure()
# plt.plot(S.Ttke_bpr, label='Buoyancy prod')
# plt.plot(S.Ttke_spr, label='Shear prod')
# plt.plot(S.Ttke_dis, label='Dissipation')
# plt.plot(S.Ttke_vdf, label='Vertical diffusion')
# plt.plot(S.Ttke_tot, '--', label='Total')
# plt.axhline(0, color='k', linewidth=0.5)

# plt.xlabel('Saved timestep')
# plt.ylabel('Vertically integrated TKE tendency')
# plt.legend()
# plt.title('TKE budget')
# plt.tight_layout()
# plt.savefig(f"data/Gaspar/tke/Gaspar_TKE_budget_gamma{S.gamma}_no_convadj.png")


# def plot_time_depth(field, title, ylabel='Vertical level', cmap='viridis',
#                     fname=None, vmin=None, vmax=None):
#     plt.figure()
#     plt.pcolormesh(field.T, shading='auto', cmap=cmap,
#                    vmin=vmin, vmax=vmax)
#     plt.gca().invert_yaxis()
#     plt.colorbar(label=title)
#     plt.xlabel('Saved timestep')
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.tight_layout()
#     plt.legend(loc='upper right') 
#     if fname is not None:
#         plt.savefig(fname)
#     plt.show()

# plot_time_depth(S.tke_s, 'TKE', ylabel='Depth level',
#                 fname=f"data/Gaspar/tke/Gaspar_TKE_evolution_gamma{S.gamma}_no_convadj.png")
# plot_time_depth(S.kv_s, 'Vertical diffusivity kv [m2/s]', ylabel='Depth level',
#                 fname=f"data/Gaspar/tke/Gaspar_kv_evolution_gamma{S.gamma}_no_convadj.png")
# plot_time_depth(S.b_s, 'buoyancy', ylabel='Depth level', 
#                 fname=f"data/Gaspar/tke/Gaspar_buoyancy_evolution_gamma{S.gamma}_no_convadj.png") 
# plot_time_depth(S.Lmix_s, 'Mixing length Lmix [m]', ylabel='Depth level',
#                 fname=f"data/Gaspar/tke/Gaspar_Lmix_evolution_gamma{S.gamma}_no_convadj.png") 
# plot_time_depth(S.N2_s, 'N^2', ylabel='Depth level', fname=f"data/Gaspar/tke/Gaspar_N2_evolution_gamma{S.gamma}_no_convadj.png")

