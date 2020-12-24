import streamlit as st
from pipeline import Pipeline

if __name__ == '__main__':

    st.write("""
    # Cool NLP demos
    """)

    paragraph = st.text_input("Take input", "Write some big paragraph in hindi")

    # paragraph = "सोनिया गांधी के दामाद रॉबर्ट वाड्रा के करीबी दुबई बेस्ड एनआरआई कारोबारी सीसी थंपू को 3 दिन की रिमांड के बाद प्रवर्तन निदेशालय (ईडी) ने उन्हें कोर्ट में पेश किया. जांच और पूछताछ के लिए कोर्ट ने थंपू को फिर से 3 दिन की रिमांड पर भेज दिया है."

    st.text("It is going to take few minutes")

    pl = Pipeline(translator_id="vasudevgupta/mbart-iitb-hin-eng", summarizer_id="facebook/mbart-large-cc25")
    summary = pl(paragraph, min_length=10, max_length=20)

    st.write("""
    Checkout output

    {}
    """.format(summary))
    print(summary)