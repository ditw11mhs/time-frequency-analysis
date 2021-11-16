import streamlit as st
from backend_time_frequency_analysis import CWT


class Main(CWT):
    def __init__(self):
        super().__init__()

    def main(self):
        st.title("Time Frequency Analysis")
        self.one()
        self.two()
        self.three()

    def one(self):
        st.markdown(
            """
                    1. For some types mother wavelets: Gaussian, derivative of gaussian (DoG), 
                    and Morlet, develop a computer program to show basic functions of Wavelet 
                    Transform (translation, scale)
                    """
        )
        with st.form("One"):
            one_state = {}
            one_state["Mother Wavelet"] = st.selectbox(
                "Select Mother Wavelet", ["Gaussian", "DoG", "Morlet"],index=2, key="one"
            )
            c1, c2 = st.columns(2)
            one_state["Translation"] = c1.slider("Translation", -100, 100, 1)
            one_state["Scale"] = c2.slider("Scale", 1, 100, 1)
            if st.form_submit_button("Run"):
                data_frame = self.compute_wavelet(50, one_state)
                self.plot_one(data_frame)
            

    def two(self):
        st.markdown(
            """
                     2. From assignment of FFT Algorithm 
                    exploration, you showed that frequency 
                    spectrum of P, QRS, and T waves explored 
                    by using window arranged manually. Design 
                    CWT using the Mother Wavelets above to 
                    explore that MRA can analyze the 
                    frequency/scale of each waves of ECG as 
                    shown in this Figure.

                     """
        )
        with st.form("two"):
            two_state = {}
            c1, c2 ,c3= st.columns(3)
            two_state["Data"] = c1.selectbox("Select Data", ["ECG"])
            two_state["Mother Wavelet"] = c2.selectbox(
                "Select Mother Wavelet", ["Gaussian", "DoG", "Morlet"],index=2, key="two"
            )
            two_state["Scale Resolution"]=c3.number_input("Scale Resolution", value=150, step=1, key="two")
            if st.form_submit_button("Run"):
                cwt_out = self.compute_cwt(two_state)
                self.plot_heatmap(cwt_out)

    def three(self):
        st.markdown(
            """
                       3. Realize Morlet Wavelet to explore 
                        spectral and temporal data of a non stationary signal. Such as a normal 
                        and MR phonocardiac signals can be 
                        recognized by their time-and 
                        frequency features.
                       """
        )
        with st.form("three"):
            three_state = {}
            c1, c2 ,c3= st.columns(3)
            three_state["Data"] = c1.selectbox("Select Data", ["Murmur","Normal"])
            three_state["Mother Wavelet"] = c2.selectbox(
                "Select Mother Wavelet", ["Morlet"], key="three"
            )
            three_state["Scale Resolution"]=c3.number_input("Scale Resolution", value=125, step=1, key="three")
            if st.form_submit_button("Run"):
                cwt_out = self.compute_cwt(three_state)
                self.plot_heatmap(cwt_out)


if __name__ == "__main__":
    st.set_page_config(page_title="Time Frequency Analysis", layout="centered",page_icon="⏲")
    main = Main()
    main.main()
