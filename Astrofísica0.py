# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 08:09:34 2025

@author: Danti
"""
import numpy as np
#import astropy
import matplotlib.pyplot as plt

from astropy.constants import G
from astropy import units as u
from astropy.time import Time
from astropy.units import Quantity

#Valores constantes a variar
#Todo se trabaja en t[días] d[radios terrestres] m[masas terrestres]

#print((9658.32210*u.km).to(u.R_earth))

t_cero = Time('2025-03-31 00:00:00', format='iso', scale='utc')
#print('t = 0 =', t_cero.mjd)

print('Se tomará como cuerpo atractor a la tierra')
smy = 1.3*u.R_earth#Quantity(input('Da el valor del semieje mayor con unidades separadas de un espacio: '))
exc = 0.16#float(input('Da el valor de la excentricidad: '))
omg = 15*u.deg #15*u.deg#Quantity(input('Da el valor del argumento del pericentro (deg: grados, rad: radianes): '))
t_ini = Time(input('Ingresa el tiempo de paso por el pericentro en formato AAAA-MM-DD hh:mm:ss (ejemplo: 2025-04-06 14:30:00): '), format='iso', scale='utc')
t_calc = Time(input('Ingresa el tiempo al que quieres calcular la posición con el mismo formato: '), format='iso', scale='utc')
#print((t_calc-t_ini).to(u.hour))
#print(t_ini.mjd-t_cero.mjd)

#---------- Constantes sin dimensiones ------------

R0 = 1              #radio del cuerpo central
e = exc         #Excentricidad
t0 = 0            #tiempo de paso por el pericentro
a = (smy.to(u.R_earth)).value         #semieje mayor
omega = (omg.to(u.rad)).value          #argumento del pericentro
G0 = (G.to(u.R_earth**3/(u.M_earth * u.hour**2))).value #Constante gravitacional
M = 1               #Masa del objeto
mu = G0*M            #parámetro gravitacional
c = np.sqrt(a**3/mu)#sqrt(mu/a**3) constante
t = ((t_calc-t_ini).to(u.hour)).value             #tiempo a medir la posición

Tperiodo = 2*np.pi*c#Periodo

print('Periodo=',Tperiodo)

#print('Volverá a estar en la misma posición en', Tvuelt.iso)

#------------ Iterpolación Cúbica de Hermite --------------------
'''
x :     lista de puntos a interpolar con H3
xi :    lista de 2 coordenadas x a interpolar (se define luego) desde [tcoord]
fi :    lista de 2 coordenadas y a interpolar (se define luego) desde [Ecoord]
dfidx : lista de derivadas [(x,f')], solo se hace con la lista xi
'''
def psi0(z):                    #fución 1
	psi_0 = 2*z**3 - 3*z**2 + 1
	return psi_0
def psi1(z):                    #función 2
	psi_1 = z**3 - 2*z**2 + z
	return psi_1
# Interpolated Polynomial
def H3(x, xi, fi, dfidx): #x en lista -> H(x) en lista
    z = (x - xi[0])/(xi[1] - xi[0])         #def variable z(x)
    h1 = psi0(z) * fi[0]                    #monomio 1
    h2 = psi0(1-z)*fi[1]                    #monomio 2
    h3 = psi1(z)*(xi[1] - xi[0])*dfidx[0]   #monomio 3
    h4 = psi1(1-z)*(xi[1] - xi[0])*dfidx[1] #monomio 4
    H =  h1 + h2 + h3 - h4                  #valor Hermite en el punto x
    return H #nos da valores de E(t) en el rango de t
#------------ Lista de derivadas para Hermite ------------------
def Derivative(x, f):
    '''
    Crea una lista dfdx con dos columnas, la primera son los datos x, la 
    segunda es la derivada de cada punto x según los datos. x es la lista 
    de N datos 'experimentales' HAY QUE CREARLA [coord]
    '''
    # Number of points
    N = len(x)
    dfdx = np.zeros([N, 2])
    dfdx[:,0] = x
    # Derivative at the extreme points
    dfdx[0,1] = (f[1] - f[0])/(x[1] - x[0]) #forward
    dfdx[N-1,1] = (f[N-1] - f[N-2])/(x[N-1] - x[N-2]) #backward
    #Derivative at the middle points
    for i in range(1,N-1):
        h1 = x[i] - x[i-1]
        h2 = x[i+1] - x[i]
        dfdx[i,1] = h1*f[i+1]/(h2*(h1+h2)) - (h1-h2)*f[i]/(h1*h2) -\
                    h2*f[i-1]/(h1*(h1+h2))
    return dfdx

#---------- Tiempo en función de la anomalía excéntrica ------------
def T(E): #para obtener los puntos 'experimentaes'
    return (E-e*np.sin(E))*c+t0

#------------------------------------------------------------------
def f(E): #esta nos da el valor de la anomalía verdadera
    if e != 1:
        return 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
    else:
        return E #hasta el momento no se ha usado

#-------- Ubicación tiempo experimental menor al requerido ------------
def texp(tlist, x): #selecciona el t inmediatamente anterior al indicado
    menores = [num for num in tlist if num < x]  # Valores menores a x
    return len(menores)-1  #posición del máximo de los menores

#------------ Función a evaluar los ceros ----------------------
def Q(x, xi, fi, dfidx):
    q = H3(x, xi, fi, dfidx) - t
    return q

#----------- Raíces - Método Bisección ----------------------------------
def R(root0,root1):
    root2 = (root0 + root1)/2.
    while np.abs(Q(root2, Einterp, tinterp, dtinterpdE)) > 1e-15:
        root2 = (root0 + root1)/2
        if Q(root0, Einterp, tinterp, dtinterpdE)*Q(root2, Einterp, tinterp, dtinterpdE) < 0:
            root1 = root2
        else:
            root0 = root2
    return root2
#------------- Graficador de órbita ----------------------------------
def orbit():
    theta = np.linspace(0, 2*np.pi-1e-2, 100)  # Angle in radians (0 to 2π)
    r = a*(1-e**2)/(1+e*np.cos(theta-omega))  # Radius (example function)
    apocentro = np.linspace(R0,(a-e*a),100)
    atractor = np.full(100,R0)
    angulo_perihelio = np.full(100,omega)
    ax.plot(angulo_perihelio, apocentro, color='black', linestyle=':')
    ax.plot(theta, atractor, color='black')
    return ax.plot(theta, r, color='red', linestyle='--')  # Plot the function

#-------------- Fecha a partir del radio
def date(r0):
    if r0 < (a-e*a) or r0 > (a+e*a):
        return print('Igrese un radio entre ',a-e*a,' y ', a+e*a )
    elif e==1:
        return print('En todo momento se encuentra en esta distancia, pues la órbita es circular')
    else:
        f0 = np.arccos((a*(1-e**2)/r0-1)/e) - omega
        E0r0 = 2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(f0/2))
        T0 = E0r0*c - e*np.sin(E0r0)*c + t0
        return print('El primer tiempo de paso por esta distancia es ', Time((T0+t_ini.mjd),format='mjd', scale='utc').iso)


fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})  # Set polar projection
ax.set_xticklabels([])  # Oculta etiquetas de ángulos
ax.xaxis.set_visible(False)  # Oculta completamente la línea de los ángulos
ax.set_yticklabels([])  # Oculta las etiquetas de los radios
ax.yaxis.set_visible(False)  # Oculta completamente las líneas de los radios
ax.spines['polar'].set_visible(False) #Oculta el encuadre exterior

#---------------- Corrección para t y t0 muy grandes -----------------
#tinicial = t
#while t0 > Tperiodo:
#    t0 -= Tperiodo
#    #print('t0=',t0)
#print('t=',t)
while t > Tperiodo:
    t -= Tperiodo
    #print('t=',(Time(t+t_ini,format='mjd', scale='utc').iso))
#if (t0+t)>Tperiodo:
#    t0 -= Tperiodo

#------- Creación de los puntos y puntos de gráfica polar -----------
N = 20 #número de puntos
Coord = np.zeros([N+1,2])

for i in range(N+1): #Matriz de las coordenadas experimentales (E,t)
    Coord[i,0] = i*2*np.pi/N    #ángulo
    Coord[i,1] = T(Coord[i,0])  #tiempo
Ecoord = Coord[:,0] #Coordenadas experimentales E
tcoord = Coord[:,1] #Coordenadas experimentales t
#print('tcoord',tcoord)
#ax.scatter(tcoord, Ecoord, color='red', s=1) #theta, r, color, tamaño
#ax.scatter(Ecoord, tcoord, color='blue', s=1)
#---------------------------------------------------------------------


#------------ Matriz de derivadas en cada punto ---------------------
#dEdt = Derivative(tcoord,Ecoord)[:,1] #lista de derivadas dE/dt
dtdE = Derivative(Ecoord,tcoord)[:,1] #lista de derivadas dt/dE

#ax.scatter(tcoord, dEdt, color='blue', s=1) #gráfica dE/dt
#ax.scatter(Ecoord, dtdE, color='red', s=1) #gráfica dt/dE

#------- Interpolación única para hallar t(E) con t específico ----------
if t==t0:
    phi = omega
else:
    tinterp = tcoord[texp(tcoord,t):texp(tcoord,t)+2] #rango ti:ti+1 a interpolar
    #print('tinterp',tinterp)
    Einterp = Ecoord[texp(tcoord,t):texp(tcoord,t)+2] #rango E a interpolar
    #print('Einterp',Einterp)
    dtinterpdE = dtdE[texp(tcoord,t):texp(tcoord,t)+2] #derivadas
    E0 = R(Einterp[0],Einterp[1]) #raíz t(E)-t    
    Fv = f(E0) #anomalía verdadera
    
    phi = Fv + omega #coordenada angular en t

d_foco = a*(1-e**2)/(1+e*np.cos(phi))

angulo = np.floor((phi*u.rad).to(u.deg)*100)/100
radiopaso = np.floor((d_foco*u.R_earth)*1000)/1000
print('Las coordenas (phi,r) son (',angulo.value,'°, ',radiopaso,') el ', t_calc, sep='')

orbit()

date(1.5)







