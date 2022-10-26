import numpy as np
import matplotlib.pyplot as plt

T = np.arange(100)
np.random.seed(2023)

naive_noise = np.random.normal(size=len(T))

ornul = naive_noise[0]
ornuls = [ornul]

for i in range(1,len(T)):
    ornul = ornul -ornul/20 + naive_noise[i]
    ornuls.append(ornul)


plt.plot(T,naive_noise, '.-',label='white noise')
plt.plot(T,ornuls, '.-',label='correlated noise')

plt.xlabel('time',fontsize=18)
plt.ylabel('noise', fontsize=18)
plt.legend()

plt.show()