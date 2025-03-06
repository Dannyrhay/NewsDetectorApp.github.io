import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px

# Load the model
@st.cache_resource
def load_transformer_model():
    with open("model.pkl", 'rb') as model_file:
        model = pickle.load(model_file)

    return model

# Streamlit app
if __name__ == '__main__':
    st.title('Fake News Classification App (Transformer Model)')
    st.write("A simple fake news classification app utilizing a Transformer model.")

    # Input text box
    st.subheader("Input the News content below")
    sentence = st.text_area("Enter your news content here", "Some news", height=200)

    predict_btt = st.button("Predict")
    model = load_transformer_model()

    if predict_btt and sentence:
        # Predict using the Transformer model - Pass raw text directly to the model
        prediction = model.predict([sentence])

        # Adjust based on the model's output format
        if isinstance(prediction, (list, np.ndarray)) and len(prediction) > 1:
            prediction_class = prediction.argmax(axis=-1)  # If it's an array of probabilities
        else:
            prediction_class = prediction  # If it's already a class label or scalar

        st.header("Prediction using Transformer model")

        if prediction_class == 0:
            st.success('This is not fake news.')
        elif prediction_class == 1:
            st.warning('This is fake news.')

        # If prediction returns probabilities, visualize them
        if isinstance(prediction, (list, np.ndarray)) and len(prediction) > 1:
            class_label = ["Fake", "True"]
            prob_dict = {"True/Fake": class_label, "Probability": prediction[0] * 100}
            df_prob = pd.DataFrame(prob_dict)
            fig = px.bar(df_prob, x='True/Fake', y='Probability')

            fig.update_layout(title_text="Transformer model - Prediction probability comparison between true and fake")
            st.plotly_chart(fig, use_container_width=True)
