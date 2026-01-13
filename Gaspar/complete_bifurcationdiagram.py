import numpy as np
import matplotlib.pyplot as plt

##### PLOT THE WHOLE BIFURCATION DIAGRAM together ############## 
plt.figure()
data =np.load("Gaspar/nz10/bifurcation_equilibria_data.npz")
for i in range(len(data["gamma"])):
    print(data["gamma"][i], data["convection"][i])
plt.plot(data["gamma"], data["convection"])
data =np.load("Gaspar/nz10/bifurcation_equilibria_data_back.npz")
plt.plot(data["gamma"], data["convection"])
plt.xlabel('gamma')
plt.ylabel('measure of convection')
data = np.load("Gaspar/nz10/bifurcation_equilibria_data_back_level4.npz")
plt.plot(data["gamma"], data["convection"])
data = np.load("Gaspar/nz10/bifurcation_equilibria_data_back_level5.npz")
plt.plot(data["gamma"], data["convection"])
data = np.load("Gaspar/nz10/bifurcation_equilibria_data_level2.npz")
plt.plot(data["gamma"], data["convection"])
data = np.load("Gaspar/nz10/bifurcation_equilibria_data_level1.npz")
plt.plot(data["gamma"], data["convection"])
data = np.load("Gaspar/nz10/bifurcation_equilibria_data_level2_extra.npz")
plt.plot(data["gamma"], data["convection"])
data = np.load("Gaspar/nz10/bifurcation_equilibria_data_level3a.npz")
plt.plot(data["gamma"], data["convection"])
plt.savefig(f"Gaspar/Gaspar_bifurcation_diagram.png")