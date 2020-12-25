import streamlit as st
from pipeline import Pipeline

from transformers import MBartForConditionalGeneration, MBartTokenizer

@st.cache(allow_output_mutation=True)
def get_model(translator_id, summarizer_id):

    print("initializing model from pre-trained weights")
    t_model = MBartForConditionalGeneration.from_pretrained(translator_id)
    t_tokenizer = MBartTokenizer.from_pretrained(translator_id)

    s_model = MBartForConditionalGeneration.from_pretrained(summarizer_id)
    s_tokenizer = MBartTokenizer.from_pretrained(summarizer_id)

    return {"model": t_model, "tokenizer": t_tokenizer}, {"model": s_model, "tokenizer": s_tokenizer}

if __name__ == '__main__':

    translator, summarizer = get_model(translator_id="vasudevgupta/mbart-iitb-hin-eng", summarizer_id="facebook/mbart-large-cc25")

    st.write("""
    # Cool NLP demo's :)
    This is a demo showing some cool use-cases of NLP.
    """)

    paragraph = st.text_area("Write a paragraph in hindi")
    min_length = st.slider("Min. length of paragraph", min_value=8, max_value=16)
    max_length = st.slider("Max. length of paragraph", min_value=16, max_value=256)

    print({
        "paragraph": paragraph,
        "num_tokens": len(paragraph.split(" ")),
        "min_length": min_length,
        "max_length": max_length
    })

    if paragraph:
        print("initiating pipeline")
        pl = Pipeline(translator, summarizer)
        summary, translation = pl(paragraph, min_length=min_length, max_length=max_length)

        st.markdown("""
        ## Summary in English
        {}
        """.format(summary[0]))

        print(translation)
        print(summary)
