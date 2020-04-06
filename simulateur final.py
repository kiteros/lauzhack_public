
import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.integrate import odeint
import urllib.request, json 
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import Bounds
from scipy.optimize import minimize

deaths_real = []
deaths_simulated = []

jour_actuel=73
echantillon=10000
temps = np.linspace(0,jour_actuel, echantillon)

def beta_(c,t):
    sum_ = 0
    for i in range(len(c)):
        sum_ += c[i]*(t**i)
    return sum_

def F(X,t):

    coef_ = [0,0,0]
    
    S,I,R,coef_[0],coef_[1],coef_[2],lambd,mu = X
    return (-beta_(coef_,t)*I*S,beta_(coef_,t)*I*S - I/lambd-mu*I, I/lambd, 0, 0, 0, 0, 0)


pop=67000000

#plt.plot(temps, pop*sol[:,0])
#plt.plot(temps, pop*sol[:,1])
#plt.plot(temps, pop*sol[:,2])

with urllib.request.urlopen("https://pomber.github.io/covid19/timeseries.json") as url:
    data = json.loads(url.read().decode())

def diff(x):

    deaths_real = []
    deaths_simulated = []
    coef = [0,0,0]

    lambd = x[0]
    mu = x[1]
    coef[0] = x[2]
    coef[1] = x[3]
    coef[2] = x[4]

    sol = odeint(F,[1-1/pop,1/pop,0,coef[0],coef[1],coef[2],lambd,mu], temps)
    mort=1-sol[:,0]-sol[:,1]-sol[:,2]

    #plt.plot(temps, pop*mort)

    aa=0
    for x in range(aa,72):
        deaths_real.append(data["France"][x]["deaths"])
        #plt.plot(x, data["France"][x]["deaths"], 'b+')

    for y in range(len(deaths_real)):
        #print(pop*mort[int((time/jours)*y)])
        deaths_simulated.append(pop*mort[int((echantillon/jour_actuel)*y)])

    sum_ = 0
    for i in range(len(deaths_real)):
        sum_ += (deaths_real[i] - deaths_simulated[i])**2

    return sum_

bounds = Bounds([2.0, 0, 0, 0, 0], [60.0, 0.5, 500.0, 500.0, 500.0])
res = minimize(diff, np.array([20,0.2,0.5,0.5,0.5]), method='trust-constr', options={'verbose': 0}, bounds=bounds)
print(res.x)

def F2(X,t):
    pop=67000000

    coeff = [0,0,0]

    lambd=res.x[0]
    mu=res.x[1]
    coeff[0] = res.x[2]
    coeff[1] = res.x[3]
    coeff[2] = res.x[4]

    S,I,R = X
    return (-beta_(coeff,t)*I*S,beta_(coeff,t)*I*S - I/lambd-mu*I, I/lambd)


pop=67000000
infecte=1-1/pop
sol = odeint(F2,[infecte,1-infecte,0], temps)
mort=1-sol[:,0]-sol[:,1]-sol[:,2]
plus=0
"""plt.plot(temps+plus, pop*sol[:,0])"""
#plt.plot(temps+plus, pop*sol[:,1])
#plt.plot(temps+plus, pop*sol[:,2])
#plt.plot(temps+plus, pop*sol[:,3])
#plt.plot(temps+plus, pop*mort)
plt.plot(temps, beta_([res.x[2],res.x[3],res.x[4]], temps))



decalage=0
for x in range(decalage,73):  
    "72 jours depuis le début"
    print(data["France"][x]["deaths"])
    #plt.plot(x-decalage, data["France"][x]["deaths"], 'b+')
plt.show()

