import numpy as np
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from numba import jit, njit
import scipy.signal
from plotly import graph_objects as go


@njit
def _morlet_wavelet_function(x):
    return np.exp(-(x ** 2) / 2) * np.cos(5 * x)


@njit
def _gauss_wavelet_function(x):
    return np.exp(-np.abs(x))


@njit
def _dog_wavelet_function(x):
    return -np.exp(-np.square(x)) * np.sin(x)


class Utils:
    def __init__(self):
        self.data = self.load_data()

    @st.cache
    def load_data(self):
        data_normal = np.loadtxt("normal2.dat")
        data_murmur = np.loadtxt("murmur3.dat")
        data_ecg = pd.read_csv("ecg 100.dat", skiprows=[1], delimiter="\t")
        return {
            "Time Normal": data_normal[:, 0].flatten(),
            "Normal": data_normal[:, 1].flatten(),
            "Time Murmur": data_murmur[:, 0].flatten(),
            "Murmur": data_murmur[:, 1].flatten(),
            "Time ECG": data_ecg["'Elapsed time'"]
            .to_numpy()
            .flatten()[int(2.381 * 360) : int(3.131 * 360)],
            "ECG": data_ecg["'V5'"]
            .to_numpy()
            .flatten()[int(2.381 * 360) : int(3.131 * 360)],
        }

    def plot_one(self, df):
        px_fig = px.line(df, x="Time", y="Output")
        px_fig.update_layout(
            yaxis_visible=False,
            yaxis_showticklabels=False,
            xaxis_showticklabels=False,
            xaxis_visible=False,
        )
        st.plotly_chart(px_fig)

    def plot_heatmap(self, data):
        fig = go.Figure(
            data=go.Surface(
                x=data["Time"].flatten(), y=data["Scales"].flatten(), z=data["CWT"]
            )
        )
        fig_2 = go.Figure(
            data=go.Heatmap(z=data["CWT"], connectgaps=True, zsmooth="best")
        )
        fig.update_layout(xaxis_title="Time", yaxis_title="Scales")
        st.plotly_chart(fig)
        st.plotly_chart(fig_2)


class CWT(Utils):
    def __init___(self):
        super().__init__()

    def get_func(self, state):
        if state["Mother Wavelet"] == "Morlet":
            return _morlet_wavelet_function
        elif state["Mother Wavelet"] == "Gaussian":
            return _gauss_wavelet_function
        elif state["Mother Wavelet"] == "DoG":
            return _dog_wavelet_function

    def compute_wavelet(self, duration, state):
        function = self.get_func(state)
        translation = state["Translation"]
        scale = state["Scale"]

        resolution = 1000
        t = np.linspace(-duration, duration, resolution)
        x = (t - translation) / scale
        output = function(x)
        output_df = pd.DataFrame({"Time": t, "Output": output})
        return output_df

    def compute_cwt(self, state):
        data = self.data[state["Data"]]
        n_data = len(data)
        data_time = np.arange(n_data)
        function = self.get_func(state)

        time_rev = data_time[::-1] * -1
        time_all = np.concatenate((time_rev[:-1], data_time))

        scales = np.arange(1, state["Scale Resolution"] + 1).reshape(-1, 1)

        wavelet_out = function(time_all / np.flip(scales, axis=0))

        cwt_out = np.array(
            list(map(lambda x: scipy.signal.correlate(data, x, "valid"), wavelet_out))
        )

        cwt_out = cwt_out / np.sqrt(np.flip(scales, axis=0))
        data_frame = pd.DataFrame(
            {
                "Amplitude": data,
                "Time": self.data["Time " + state["Data"]],
            }
        )
        fig = px.line(data_frame, x="Time", y="Amplitude")
        st.plotly_chart(fig)
        return {"CWT": cwt_out, "Time": data_time, "Scales": scales}
