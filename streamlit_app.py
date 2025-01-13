import streamlit as st
import numpy as np
from neuron import Neuron

st.image("neurona.jpg", width=300)

st.header("Simulador de neurona")

# Establece el número de entradas y pesos
num = st.slider("Elige el número de entradas/pesos que tendrá la neurona", 1, 10)

# Crea tantas columnas, pesos y entradas como entradas/pesos se hayan seleccionado
st.header("Pesos")
col_x_w = st.columns(num)
w = np.zeros(num)
x = np.zeros(num)

# Muestra los pesos a introducir y el array resultante
for i in range(num):
    col_x_w[i].markdown(f"w<sub>{i}</sub>", unsafe_allow_html=True)
    w[i] = col_x_w[i].number_input("", 0.0)
st.write(f"w = {w}")

# Muestra las entradas a introducir y el array resultante
st.header("Entradas")
for i in range(num):
    x[i] = col_x_w[i].number_input(f"x<sub>{i}</sub>", 0.0, unsafe_allow_html=True)
st.write(f"x = {x}")

col_b_f = st.columns(2)
col_b_f[0].header("Sesgo")
b = col_b_f[0].number_input("Introduce el valor del sesgo")
col_b_f[1].header("Función de activación")
f = col_b_f[1].selectbox("Elige la función de activación", ["Sigmoide", "ReLu", "Tangente hiperbólica"])

# Crea una instancia de la clase Neuron y calcula el resultado
n1 = Neuron(weights=w, bias=b, func=f)
y = n1.run(x)
print(f"La salida de la neurona es {y}")

