import numpy as np
import matplotlib.pyplot as plt

T = np.arange(1000000)
np.random.seed(2023)

naive_noise = np.random.normal(size=len(T))
taus = np.arange(10,2010,10)


ornul = naive_noise[0] * np.ones(len(taus))
all_ornuls = np.zeros(shape=(len(T), len(taus)))

all_ornuls[0] = ornul

for i in range(1,len(T)):
    ornul = ornul -ornul/taus + naive_noise[i]
    all_ornuls[i] = ornul


plt.plot(np.sqrt(taus), np.std(all_ornuls, axis=0),'o-')




sigmas = np.arange(10,2010,10)


ornul = naive_noise[0] * np.ones(len(sigmas))
all_ornuls = np.zeros(shape=(len(T), len(sigmas)))

all_ornuls[0] = ornul

for i in range(1,len(T)):
    ornul = ornul -ornul/100 + sigmas*naive_noise[i]
    all_ornuls[i] = ornul

plt.figure()
plt.plot(sigmas, np.std(all_ornuls, axis=0),'o-')

plt.show()