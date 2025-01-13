import streamlit as st
import numpy as np
from neuron import Neuron

st.image("neurona.jpg", width=300)

st.header("Simulador de neurona")

# Establece el número de entradas y pesos
num = st.slider("Elige el número de entradas/pesos que tendrá la neurona", 1, 10)

# Crea tantas columnas como pesos se hayan seleccionado
st.markdown("<h3>Pesos</h3>", unsafe_allow_html=True)
col_w = st.columns(num)
w = np.zeros(num)

# Muestra los pesos a introducir y el array resultante
for i in range(num):
    col_w[i].markdown(f"w<sub>{i}</sub>", unsafe_allow_html=True)
    w[i] = col_w[i].number_input("", 0.0, key=f"w{i}")
st.write(f"w = {w}")

# Crea tantas columnas como entradas se hayan seleccionado
st.markdown("<h3>Entradas></h3>", unsafe_allow_html=True)
col_x = st.columns(num)
x = np.zeros(num)

# Muestra las entradas a introducir y el array resultante
for i in range(num):
    col_x[i].markdown(f"x<sub>{i}</sub>", unsafe_allow_html=True)
    x[i] = col_x[i].number_input("", 0.0, key=f"x{i}")
st.write(f"x = {x}")

col_b_f = st.columns(2)
col_b_f[0].markdown("<h3>Sesgo</h3>", unsafe_allow_html=True)
b = col_b_f[0].number_input("Introduce el valor del sesgo")
col_b_f[1].markdown("<h3>Función de activación</h3>", unsafe_allow_html=True)
f = col_b_f[1].selectbox("Elige la función de activación", ["Sigmoide", "ReLu", "Tangente hiperbólica"])

# Crea una instancia de la clase Neuron y calcula el resultado
n1 = Neuron(weights=w, bias=b, func=f)
y = n1.run(x)
print(f"La salida de la neurona es {y}")

