import nltk
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from PIL import Image

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize PorterStemmer
ps = PorterStemmer()

# Text preprocessing function
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize text
    
    # Remove non-alphanumeric characters
    text = [word for word in text if word.isalnum()]
    
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    
    # Perform stemming
    text = [ps.stem(word) for word in text]
    
    return " ".join(text)

# Load the pre-trained model and vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

# Streamlit app layout
st.set_page_config(page_title="SMS Spam Detection", page_icon="📩", layout="wide")

# Add a header with an icon
st.title("📩 SMS Spam Detection App")
st.write("**By JAIKEY SINGH**")

# Add an introductory image (optional)
#st.image("spam_detection_banner.png", use_column_width=True, caption="Protect yourself from spam!")

# Sidebar for instructions
st.sidebar.title("Instructions")
st.sidebar.info(
    """
    1. Enter the SMS text in the input box.\n
    2. Click the "Predict" button to check if it's spam or not.\n
    3. The result will be displayed below.
    """
)

# Input field for SMS
st.subheader("Enter the SMS Text Below")
input_sms = st.text_area("Type your message here...", placeholder="e.g., Congratulations! You've won a free gift card!")

# Prediction button
if st.button("Predict"):
    if input_sms.strip():  # Ensure input is not empty
        # Preprocess the input
        transformed_sms = transform_text(input_sms)
        
        # Vectorize the input
        vector_input = vectorizer.transform([transformed_sms])
        
        # Predict using the model
        result = model.predict(vector_input)[0]
        
        # Display result
        if result == 1:
            st.error("🚨 This message is **Spam**!")
        else:
            st.success("✅ This message is **Not Spam**!")
    else:
        st.warning("Please enter some text to analyze.")

# Footer
st.markdown("---")
st.markdown(
    "Developed with ❤️ by JAIKEY SINGH using **Streamlit**, **NLTK**, and **Scikit-learn**. "
    "For more projects, visit our [GitHub](https://github.com/JAIKEYSINGH913)."
)
