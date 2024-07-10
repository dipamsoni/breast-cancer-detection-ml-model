from random_forest import *
from decision_tree_streamlit_EDA import *
import json

## Streamlit app configuration
st.set_page_config(layout="wide")
st.title("Breast Cancer Data EDA and Model Building")

# Sidebar for navigation
# page = st.sidebar.selectbox("Choose a page", ["JSON prediction", "Randomforest Model", "Prediction", "EDA"])
page = st.sidebar.selectbox("Choose a page", ["EDA", "Randomforest Model", "JSON prediction"])

# Run selected page
if page == "EDA":
    run_eda()
elif page == "Randomforest Model":
    model_result = run_model()
    st.header("Model Building")
    st.write(f"Model trained with accuracy: {model_result['accuracy'] * 100:.2f}%")
elif page == "Prediction":
    model_result = run_model()
    run_prediction(model_result)
elif page == "JSON prediction":
    st.header("Breast Cancer Prediction from JSON")
    st.write("Enter the prediction input data in JSON format:")
    json_input = st.text_area("JSON Input", height=300)
    if st.button("Predict"):
        try:
            json_data = json.loads(json_input)
            prediction = run_prediction_from_json(json_data)
            st.header(prediction)
            # st.header(f"The probability of Not having breast cancer is {(1 - prediction[1]) * 100:.2f}%")
        except Exception as e:
            st.error(f"Error occurred: {e}")