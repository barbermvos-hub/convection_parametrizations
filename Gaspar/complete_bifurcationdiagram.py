import numpy as np
import matplotlib.pyplot as plt

# #### PLOT THE WHOLE BIFURCATION DIAGRAM together ############## 
# plt.figure()
# data =np.load("data/Gaspar/nz10/bifurcation_equilibria_data.npz")
# for i in range(len(data["gamma"])):
#     print(data["gamma"][i], data["convection"][i])
# plt.plot(data["gamma"], data["convection"])
# data =np.load("data/Gaspar/nz10/bifurcation_equilibria_data_back.npz")
# plt.plot(data["gamma"], data["convection"])
# plt.xlabel('gamma')
# plt.ylabel('measure of convection')
# data = np.load("data/Gaspar/nz10/bifurcation_equilibria_data_back_level4.npz")
# plt.plot(data["gamma"], data["convection"])
# data = np.load("data/Gaspar/nz10/bifurcation_equilibria_data_back_level5.npz")
# plt.plot(data["gamma"], data["convection"])
# data = np.load("data/Gaspar/nz10/bifurcation_equilibria_data_level2.npz")
# plt.plot(data["gamma"], data["convection"])
# data = np.load("data/Gaspar/nz10/bifurcation_equilibria_data_level1.npz")
# plt.plot(data["gamma"], data["convection"])
# data = np.load("data/Gaspar/nz10/bifurcation_equilibria_data_level2_extra.npz")
# plt.plot(data["gamma"], data["convection"])
# data = np.load("data/Gaspar/nz10/bifurcation_equilibria_data_level3a.npz")
# plt.plot(data["gamma"], data["convection"])
# plt.savefig(f"Gaspar/Gaspar_bifurcation_diagram.png")

# #Salt relaxation
# plt.figure()
# data = np.load("data/Gaspar/sal_relax/bifurcation_equilibria_data.npz")
# plt.plot(data["gamma"], data["convection"])
# data = np.load("data/Gaspar/sal_relax/bifurcation_equilibria_data_back1.npz")
# plt.plot(data["gamma"], data["convection"])
# plt.savefig(f"Gaspar/Gaspar_bifurcation_diagram.png")


# #Relaxation and surface fluxes, gamma and Qextra
from mpl_toolkits.mplot3d import Axes3D  # required for 3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# --- Qextra > 0 ---
data = np.load("data/Gaspar/flux_surfforc_gamma/bifurcation_equilibria_data_Qextra.npz")
ax.plot(
    data["Qextra"],
    -0.39 * np.ones(len(data["convection"])),
    data["convection"],
    label="Qextra > 0"
)

# --- Qextra < 0 ---
data = np.load("data/Gaspar/flux_surfforc_gamma/bifurcation_equilibria_data_Qextra_neg.npz")
ax.plot(
    data["Qextra"],
    -0.39 * np.ones(len(data["convection"])),
    data["convection"],
    label="Qextra < 0"
)

# --- gamma bifurcation ---
data = np.load("data/Gaspar/flux_surfforc_gamma/bifurcation_equilibria_data.npz")
ax.plot(
    np.zeros(len(data["gamma"])),
    data["gamma"],
    data["convection"],
    label="gamma"
)

# Labels
ax.set_xlabel('Qextra')
ax.set_ylabel('gamma')
ax.set_zlabel('measure of convection')

ax.legend()
plt.tight_layout()
plt.savefig("Gaspar/Gaspar_bifurcationdiagram_flux_gamma_Qextra.png", dpi=300)
plt.show()


#Salt relaxation
# plt.figure()
# data = np.load("data/Gaspar/relax_surfforc_gamma/bifurcation_equilibria_data.npz")
# plt.plot(data["gamma"], data["convection"])
# plt.show()

# zt = -1. * np.arange(len(data["temp"][13,:]))

# gamma =  data["gamma"][13]
# temp  = data["temp"][13,:]
# sal   = data["sal"][13,:]
# measure_convection = data["convection"][13]

# plt.plot((temp - 20)*2e-4 , zt)
# plt.plot((sal - 35)*7.6e-4, zt)
# plt.show()
# print(gamma, measure_convection)