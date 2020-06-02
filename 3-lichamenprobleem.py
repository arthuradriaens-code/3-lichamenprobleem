"""
3 lichamen probleem met 3D animation. het representeert de evolutie van het Centauri binair systeem en zijn 3de minder zware ster,
onder newtonianaanse zwaartekracht.
"""
# gebaseerd op:
# https://towardsdatascience.com/modelling-the-three-body-problem-in-classical-mechanics-using-python-9dc270ad7767

import scipy as sci
import scipy.integrate
#nu voor 3D weergave matplotlib
from matplotlib.colors import cnames
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

import numpy as np

'''
nu de exacte constanten (referentie)
'''
#definieer universele zwaartekrachtconstante
G=6.67408e-11 #N-m2/kg2
#Referentie grootheden om dimensieloosheid te bekomen (snellere berekeningen)
m_nd=1.989e+30 #kg #massa van de zon
r_nd=5.326e+12 #m #afstand tussen sterren in Alpha Centauri
v_nd=30000 #m/s #relatieve snelheid van de aarde rond de zon
t_nd=79.91*365*24*3600*0.51 #s #orbitaalperiode van Alpha Centauri
#terugreken constantes
K1=1#G*t_nd*m_nd/(r_nd**2*v_nd)
K2=1#v_nd*t_nd/r_nd

'''
gekozen configuratie van het systeem
'''
#massa's
m1=1 #Alpha Centauri A
m2=1 #Alpha Centauri B
m3=1 #derde Ster

#initiële positievectors
r1=[-1,0,0] #in meter
r2=[1,0,0] # "
r3=[0,0,0] # "

#Converteer positievectors naar arrays
r1=np.array(r1,dtype="float64")
r2=np.array(r2,dtype="float64")
r3=np.array(r3,dtype="float64")

#Definieer beginsnelheden
p1 = 0.347111
p2 = 0.532728

v1=[p1,p2,0] #m/s
v2=[p1,p2,0] #m/s
v3=[-2*p1,-2*p2,0] #m/s

#Converteer snelheidsvectorer naar arrays
v1=np.array(v1,dtype="float64")
v2=np.array(v2,dtype="float64")
v3=np.array(v3,dtype="float64")

#vind het massamiddelpunt
r_com=(m1*r1+m2*r2 +m3*r3)/(m1+m2+m3)

#vind de snelheid van het massamiddelpunt
v_com=(m1*v1+m2*v2+m3*v3)/(m1+m2+m3)

'''
Om een ODE op te lossen heb je een paar initiële condities nodig en een tijdsperiode, de odeint oplosser
neemt een array met alle afhankelijke variabelen (hier positie en snelheid) en een array met alle
onafhankelijke variabelen (hier tijd) in die volgorde.
'''
#dit is een functie die de bewegingsvergelijkingen definieert in 2D, w is een allesbevattende lijst.
def TwoBodyEquations(w,t,G,m1,m2):
    r1=w[:3]
    r2=w[3:6]
    v1=w[6:9]
    v2=w[9:12]

    r=sci.linalg.norm(r2-r1) #bereken de grootte/norm van de vector en dus de afstand tussen 1 en 2

    #berekenen van \frac{dv_i}{dt}
    dv1bydt=K1*m2*(r2-r1)/r**3
    dv2bydt=K1*m1*(r1-r2)/r**3

    #berekenen van \frac{dr_i}{dt}
    dr1bydt=K2*v1
    dr2bydt=K2*v2


    r_derivs=np.concatenate((dr1bydt,dr2bydt)) #samenvoegen in 1 array van de posities afgelijd naar de tijd
    derivs=np.concatenate((r_derivs,dv1bydt,dv2bydt)) #opnieuw samenvoegen maar nu met snelheden
    return derivs
#analoog ma dan 3D
def ThreeBodyEquations(w,t,G,m1,m2,m3):
    r1=w[:3]
    r2=w[3:6]
    r3=w[6:9]
    v1=w[9:12]
    v2=w[12:15]
    v3=w[15:18]
    r12=sci.linalg.norm(r2-r1) #bereken de grootte/norm van de vector en dus de afstand tussen 2 en 1
    r13=sci.linalg.norm(r3-r1) # " 3 en 1
    r23=sci.linalg.norm(r3-r2) # " 3 en 2

    #berekenen van \frac{dv_i}{dt}
    dv1bydt=K1*m2*(r2-r1)/r12**3+K1*m3*(r3-r1)/r13**3
    dv2bydt=K1*m1*(r1-r2)/r12**3+K1*m3*(r3-r2)/r23**3
    dv3bydt=K1*m1*(r1-r3)/r13**3+K1*m2*(r2-r3)/r23**3

    #berekenen van \frac{dr_i}{dt}
    dr1bydt=K2*v1
    dr2bydt=K2*v2
    dr3bydt=K2*v3

    r12_derivs=np.concatenate((dr1bydt,dr2bydt))
    r_derivs=np.concatenate((r12_derivs,dr3bydt))
    v12_derivs=np.concatenate((dv1bydt,dv2bydt))
    v_derivs=np.concatenate((v12_derivs,dv3bydt))
    derivs=np.concatenate((r_derivs,v_derivs))
    return derivs

#samenvoegen van de initiële parameters 2D
'''
init_params=np.array([r1,r2,v1,v2]) #array van de initiële parameters
init_params=init_params.flatten() #array 1D maken
time_span=np.linspace(0,8,500) #8 orbitaalperiodes als tijdsperiode en 500 punten
'''
#samenvoegen van de initiële parameters 3D
init_params=np.array([r1,r2,r3,v1,v2,v3]) #array van de initiële parameters
init_params=init_params.flatten() #array 1D maken
time_span=np.linspace(0,20,500) #20 orbitaalperiodes als tijdsperiode en 500 punten (array met 500 waarden tussen 0 en 20)

#nu de ODE solver uitvoeren
#two_body_sol=sci.integrate.odeint(TwoBodyEquations,init_params,time_span,args=(G,m1,m2))
three_body_sol=sci.integrate.odeint(ThreeBodyEquations,init_params,time_span,args=(G,m1,m2,m3))

#nu de posities achterhalen om ze te kunnen plotten 2D:
'''
r1_sol=two_body_sol[:,:3]
r2_sol=two_body_sol[:,3:6]
'''
#nu de posities achterhalen om ze te kunnen plotten 3D:
r1_sol=three_body_sol[:,:3]
r2_sol=three_body_sol[:,3:6]
r3_sol=three_body_sol[:,6:9]

#locatie van het massamiddelpunt vinden
rcom_sol=(m1*r1_sol+m2*r2_sol+m3*r3_sol)/(m1+m2+m3)
#locatie van Alpha Centuari A t.o.v het massamiddelpunt
r1com_sol=r1_sol-rcom_sol
#locatie van Alpha Centuari B t.o.v het massamiddelpunt
r2com_sol=r2_sol-rcom_sol
r3com_sol=r3_sol-rcom_sol

"""
effectief plotten
"""
#maak een window
fig = plt.figure()

#3D stelsel
ax=fig.add_subplot(111,projection='3d')

"""
Statisch gedeelte
"""

"""
#massacentrum:
#Plot de orbitalen (x,y,z)
ax.plot(r1com_sol[:,0],r1com_sol[:,1],r1com_sol[:,2],color="darkblue")
ax.plot(r2com_sol[:,0],r2com_sol[:,1],r2com_sol[:,2],color="tab:red")
ax.plot(r3com_sol[:,0],r3com_sol[:,1],r3com_sol[:,2],color="tab:green")

#Plot de uiteindelijke posities van de sterren
ax.scatter(r1com_sol[-1,0],r1com_sol[-1,1],r1com_sol[-1,2],color="darkblue",marker="o",s=100,label="Alpha Centauri A")
ax.scatter(r2com_sol[-1,0],r2com_sol[-1,1],r2com_sol[-1,2],color="tab:red",marker="o",s=100,label="Alpha Centauri B")
ax.scatter(r3com_sol[-1,0],r3com_sol[-1,1],r3com_sol[-1,2],color="tab:green",marker="o",s=100,label="3de ster")
"""
"""
#random punt:
#Plot de orbitalen (x,y,z)
ax.plot(r1_sol[:,0],r1_sol[:,1],r1_sol[:,2],color="darkblue")
ax.plot(r2_sol[:,0],r2_sol[:,1],r2_sol[:,2],color="tab:red")
ax.plot(r3_sol[:,0],r3_sol[:,1],r3_sol[:,2],color="tab:green")

#Plot de uiteindelijke posities van de sterren
ax.scatter(r1_sol[-1,0],r1_sol[-1,1],r1_sol[-1,2],color="darkblue",marker="o",s=100,label="Alpha Centauri A")
ax.scatter(r2_sol[-1,0],r2_sol[-1,1],r2_sol[-1,2],color="tab:red",marker="o",s=100,label="Alpha Centauri B")
ax.scatter(r3_sol[-1,0],r3_sol[-1,1],r3_sol[-1,2],color="tab:green",marker="o",s=100,label="3de ster")
"""

"""
Vanaf hier het animatiegedeelte
"""


fps = 28 # Frame per seconde aangezien 20 seconden 500 punten
nfr = 500 # Number of frames
scat1, = ax.plot([],[],[],color="darkblue",marker="o",label="Alpha Centauri A")
scat2, = ax.plot([],[],[],color="tab:red",marker="o",label="Alpha Centauri B")
scat3, = ax.plot([],[],[],color="tab:green",marker="o",label="3de ster")

lijn1, = ax.plot([],[],[],color="darkblue")
lijn2, = ax.plot([],[],[],color="tab:red")
lijn3, = ax.plot([],[],[],color="tab:green")

def update(i,coordinates1,coordinates2,coordinates3):

    #momentaan punt:
    xyz1 = coordinates1
    xyz2 = coordinates2
    xyz3 = coordinates3
    scat1.set_data(xyz1[0][i], xyz1[1][i])
    scat1.set_3d_properties(xyz1[2][i])
    scat2.set_data(xyz2[0][i], xyz2[1][i])
    scat2.set_3d_properties(xyz2[2][i])
    scat3.set_data(xyz3[0][i], xyz3[1][i])
    scat3.set_3d_properties(xyz3[2][i])

    #orbitaal:

    lijn1.set_data(xyz1[0][:i], xyz1[1][:i])
    lijn1.set_3d_properties(xyz1[2][:i])
    lijn2.set_data(xyz2[0][:i], xyz2[1][:i])
    lijn2.set_3d_properties(xyz2[2][:i])
    lijn3.set_data(xyz3[0][:i], xyz3[1][:i])
    lijn3.set_3d_properties(xyz3[2][:i])


x1,y1,z1 = r1com_sol[:,0],r1com_sol[:,1],r1com_sol[:,2]
x2,y2,z2 = r2com_sol[:,0],r2com_sol[:,1],r2com_sol[:,2]
x3,y3,z3 = r3com_sol[:,0],r3com_sol[:,1],r3com_sol[:,2]
xyz1,xyz2,xyz3 = (x1,y1,z1),(x2,y2,z2),(x3,y3,z3)
ani = animation.FuncAnimation(fig, update, 10000, fargs=(xyz1,xyz2,xyz3), interval=(1/fps)*1000,blit=False)
#dit kon wss minder brak ma idk hoe


"""
Opslaan van de video

fn = 'plot_3d_scatter_funcanimation'
#ani.save(fn+'.gif',writer='imagemagick',fps=fps)
ani.save(fn+'.mp4',writer='ffmpeg',fps=fps)
"""

"""
Opmaak van het assenstelsel
"""
ax.set_xlabel("x-as",fontsize=14)
ax.set_ylabel("y-as",fontsize=14)
ax.set_zlabel("z-as",fontsize=14)
ax.set_title("3 lichamen systeem\n",fontsize=14)
ax.legend(loc="upper left",fontsize=14)
#ook nog eens werken aan het 'custom assenstelsel' ipv maxima enzo automatische updates
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_zlim(-0.025,0.025)

"""
#custom assenstelsel
xmin=x3.min(); xmax=x3.max()
ymin=y3.min(); ymax=y3.max()
zmin=z3.min(); zmax=z3.max()
ax.set_xlim(xmin-0.1*(xmax-xmin),xmax+0.1*(xmax-xmin))
ax.set_ylim(ymin-0.1*(ymax-ymin),ymax+0.1*(ymax-ymin))
ax.set_zlim(zmin-0.1*(zmax-zmin),zmax+0.1*(zmax-zmin))
"""

plt.show()
