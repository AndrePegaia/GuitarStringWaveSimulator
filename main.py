import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
import numba
from numba import jit
import math

# Na equação, temos:
# y-amplitude, t-tempo, x-posição da corda
# j-parte da corda, m-"snapshot in time" (valor momentâneo)


# Corda de violão possui comprimento de cerca de 0.7m
# L = nx*dx (comprimento da corda é igual ao número de intervalos vezes o espaço do intervalo)
Nx = 101
L = 0.7
dx = L/(Nx-1)

# Para obedecer à restrição da equação (cdt/dx <1), adota-se dt = 5*10^-6s
# Nt*dt = duração da "análise"
duracao = 2.5
dt = 5e-6
Nt = int(math.ceil(duracao/dt))

#f-frequência; c-velocidade da onda;
# frequência de nota de violão - f=c/2L -> c=2L*f
nomeNota = input("Insira o nome da nota (ex: E2): ")
f = int(input("Insira a frequência da nota (ex: 82): "))
c = 2*f*L

#l-comprimento carctterístico; gama-constante de amortecimento (valores obtidos por teste)
l = 2e-6
gamma = 2.6e-5

print("Aguarde enquanto está sendo carregado! Esse processo pode levar alguns minutos...")

#Estado inicial da corda:
ya = np.linspace(0, 0.01, 50)
yb = np.linspace(0.01, 0, 51)
y0 = np.concatenate([ya, yb])

#y0 = np.zeros(101)
#for x in range(101):
#    y0[x] = -0.000004*(x**2 - 100*x +15)

#Criar variável solução que armazenará os valores de x em função do tempo ao final, além de definir os dois 1°s valores
sol = np.zeros((Nt, Nx))
sol[0]= y0
sol[1]= y0

#Cálculo da solução em função do tempo a partir da equação diferencial
@numba.jit("f8[:,:](f8[:,:],i8,i8,f8,f8,f8,f8)", nopython=True, nogil=True)
def compute_d(d, times, lenght, dt, dx, l, gamma):
    lenght = len(d[0])
    for t in range(1, times-1):
        for i in range(2, lenght-2):
            outer_fact = (1/(c**2 * dt**2) + gamma/(2*dt))**(-1)
            p1 = 1/dx**2 * (d[t][i-1] - 2*d[t][i] + d[t][i+1])
            p2 = 1/(c**2 * dt**2) * (d[t-1][i] - 2*d[t][i])
            p3 = gamma/(2*dt) * d[t-1][i]
            p4 = l**2 / dx**4 * (d[t][i+2] - 4*d[t][i+1] + 6*d[t][i] - 4*d[t][i-1] + d[t][i-2])
            d[t+1][i] = outer_fact * (p1 - p2 + p3 - p4)
    return d

#Receber solução
sol = compute_d(sol, Nt, Nx, dt, dx, l, gamma)


#Gerar GIF a partir do gráfico em função do tempo
def animate(i):
    ax.clear()
    ax.plot(sol[i*10])
    ax.set_ylim(-0.01, 0.01)

fig, ax = plt.subplots(1,1)
ax.set_ylim(-0.01, 0.01)
ani = animation.FuncAnimation(fig, animate, frames=500, interval=50)
ani.save('GIFs/string.gif', writer="pillow", fps=30)

#Gerar arquivo WAV a partir da integral das harmônicas da onda
def get_integral(n):
    sin_arr = np.sin(n*np.pi*np.linspace(0,1,101))
    return np.multiply(sol, sin_arr).sum(axis=1)

hms = [get_integral(n) for n in range(10)]

total = sol.sum(axis=1)[::10]
total= total.astype(np.float32)

from scipy.io import wavfile
wavfile.write(f'WAVs/{nomeNota}.wav', 20000, total)
print("Processo concluído!")