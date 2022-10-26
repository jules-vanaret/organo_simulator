import numpy as np
import matplotlib.pyplot as plt


"""
conclusion
    rmin=10
    alpha=10
    eps=
"""


def E(r, eps, alpha, rmin):
    return 4 * eps * ((rmin/r)**12 - (rmin/r)**6)  

def potential_strength(r, eps, alpha, rmin):
    return eps * (-6/(alpha-6) * alpha/rmin * np.exp(alpha * (1-r/rmin)) + 6/rmin * alpha/(alpha-6) * (rmin/r)**7)


X = np.linspace(0.1, 20, 1000)

# for e in np.logspace(-1,np.log10(10),5):
#     Y = E(X, eps=-e, alpha=10, rmin=10)
#     plt.plot(X,Y, label=f'e={e}')

# plt.ylim(-5, 5)
# plt.legend()

# plt.figure()

# for e in np.logspace(-1,np.log10(10),5):
#     Y = -np.gradient(E(X, eps=-e, alpha=10, rmin=10), X)
#     plt.plot(X,Y, label=f'e={e}')
# plt.ylim(-5, 5)
# plt.legend()


Y = E(X, eps=-10, alpha=5, rmin=10)
Y2 = E(X, eps=-20, alpha=5, rmin=10)
S = potential_strength(X, -10,5,10)

plt.plot(X,Y)
plt.plot(X,np.gradient(Y,X))
plt.plot(X, S, 'k--')
plt.plot(X,Y2)
plt.plot(X,np.gradient(Y2,X))
plt.plot(X,[0]*len(X),'k')
plt.ylim(-30, 30)
plt.show()