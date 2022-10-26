import numpy as np
import matplotlib.pyplot as plt
from organo_simulator.simulator import FastOverdampedSimulator

"""
conclusion
    rmin=10
    alpha=10
    eps=
"""

# def f(x):

#     return x*(x<1)+2*(x>=1)

#     # if x<1:
#     #     return x
#     # else:
#     #     return 2

# print(f(np.arange(5)))

# # def E(x, eps, alpha, rmin):
# #     return eps * (6/(alpha-6) * np.exp(alpha * (1-x/rmin)) - alpha/(alpha-6) * (rmin/x)**6)

# # def potential_strength(r, eps, alpha, rmin):
# #     return eps * (-6/(alpha-6) * alpha/rmin * np.exp(alpha * (1-r/rmin)) + 6/rmin * alpha/(alpha-6) * (rmin/r)**7)


# def E(r, eps, rmin):
#     attraction =  (rmin/r)**6
#     return 4 * eps * (attraction**2 - attraction)

# def potential_strength(r, eps, rmin):
#     attraction =  (rmin/r)**6
#     return 4 * eps * (-12*attraction**2/r + 6*attraction/r)


# X = np.linspace(0.1, 20, 1000)

# # for e in np.logspace(-1,np.log10(10),5):
# #     Y = E(X, eps=-e, alpha=10, rmin=10)
# #     plt.plot(X,Y, label=f'e={e}')

# # plt.ylim(-5, 5)
# # plt.legend()

# # plt.figure()

# # for e in np.logspace(-1,np.log10(10),5):
# #     Y = -np.gradient(E(X, eps=-e, alpha=10, rmin=10), X)
# #     plt.plot(X,Y, label=f'e={e}')
# # plt.ylim(-5, 5)
# # plt.legend()


# Y = E(X, eps=10, rmin=10)
# Y2 = E(X, eps=20, rmin=10)
# S = potential_strength(X, 10,10)

# plt.plot(X,Y)
# plt.plot(X,np.gradient(Y,X))
# plt.plot(X, S, 'k--')
# # plt.plot(X,Y2)
# # plt.plot(X,np.gradient(Y2,X))
# plt.plot(X,[0]*len(X),'k')
# plt.ylim(-30, 30)
# plt.show()

N_part = 100#100
d=2
L = 50
Nx = 200

# simulator = FastOverdampedSimulator(
#     L=L,
#     Nx=Nx,
#     d=d,
#     typical_size=5,
#     viscosity=100,
#     D=1000,
#     debug_dist=False,
#     persistence_time=None,
#     energy_potential=1
# )


# R = np.linspace(5, 20, 1000)

# forces = simulator._FastOverdampedSimulator__potential_force(R, size=5, eps=1)

# plt.plot(R, forces)
# plt.plot(R, (np.cumsum(forces)-np.cumsum(forces)[-1])*np.gradient(R)[0])
# plt.ylim(-2,1)


R = np.linspace(0, 30, 100)

def naive_potential_force(r, nuclei_size, eps, wiggle_room):
        # See "ya||a: GPU-Powered Spheroid Models for Mesenchyme and Epithelium,
        # (2019), Sharpe et al"

        nuclei_diameter = 2 * nuclei_size

        value_1 = np.maximum(nuclei_diameter - r, 0)
        value_2 = np.maximum(r - nuclei_diameter - 2*wiggle_room, 0)

        return - eps * (2 * value_1 - value_2)

def test(r):
    return np.max(0.7 - r, 0) * 2 - np.max(r - 0.8, 0)

    

plt.plot(R,[naive_potential_force(elem, 10, 1,0)+0.1 for elem in R])
plt.plot(R,[naive_potential_force(elem, 10, 1,0.1*10) for elem in R])
# plt.plot(R[1:], 10*np.diff([f(elem) for elem in R]))


plt.show()