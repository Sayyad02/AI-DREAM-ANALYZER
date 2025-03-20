import streamlit as st
import pandas as pd
import nltk
import spacy
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import base64

# --- Download necessary NLTK data (only needs to be done once) ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# --- Load spaCy model ---
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.warning("Downloading spaCy model 'en_core_web_sm'. This may take a moment...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- Text Preprocessing Function ---
def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    stop_words = nltk.corpus.stopwords.words('english')
    filtered_tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    return " ".join(lemmatized_tokens)

# --- Keyword Extraction Function ---
def extract_keywords(text):
    tokens = nltk.word_tokenize(text)
    tagged_tokens = nltk.pos_tag(tokens)
    keywords = [word for word, tag in tagged_tokens if tag.startswith('N') or tag.startswith('V')]
    return keywords

# --- Dream Symbol Dictionary ---
dream_symbols = {
    "water": ["emotions", "the unconscious", "change", "purification"],
    "falling": ["insecurity", "loss of control", "anxiety", "failure"],
    "flying": ["freedom", "escape", "success", "transcendence"],
    "teeth": ["anxiety", "loss of power", "aging", "vulnerability"],
    "chase": ["avoidance", "stress", "fear", "running from something"],
    "house": ["self", "mind", "security", "different aspects of your life"],
    "food": ["nourishment", "knowledge", "satisfaction", "emotional hunger"],
    "money": ["wealth", "success", "power", "self-worth"],
    "death": ["endings", "transformation", "new beginnings", "letting go"],
    "animals": ["instincts", "emotions", "specific animal traits"], # You'd need to sub-categorize animals
    "colors": ["emotions", "moods", "different meanings depending on the color"], # Needs sub-categories
    "numbers": ["patterns", "order", "specific number symbolism"], # Needs sub-categories
    "people": ["relationships", "aspects of yourself", "specific people in your life"],
    "road": ["journey", "life path", "direction", "choices"],
    "vehicle": ["control", "direction in life", "speed of progress"],
    "school": ["learning", "growth", "challenges", "social situations"],
    "work": ["career", "responsibilities", "stress", "achievement"],
     "sky": ["freedom", "limitless potential", "spirituality"],
    "fire": ["passion", "destruction", "transformation", "anger"]
}


st.markdown("""
<style>
/* --- Background Image (Full Page, with Overlay) --- */
body {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100%25' height='100%25' viewBox='0 0 1600 800'%3E%3Cdefs%3E%3ClinearGradient id='a' x1='0' x2='0' y1='1' y2='0'%3E%3Cstop offset='0' stop-color='%2308294a'/%3E%3Cstop offset='1' stop-color='%234a90e2'/%3E%3C/linearGradient%3E%3ClinearGradient id='b' x1='0' x2='0' y1='0' y2='1'%3E%3Cstop offset='0' stop-color='%236a85b6'/%3E%3Cstop offset='1' stop-color='%232a6099'/%3E%3C/linearGradient%3E%3C/defs%3E%3Cg fill='%236a85b6' fill-opacity='0.05'%3E%3Cpath d='M1518 765.7c-67 0-124.3-15.3-171.8-45.8-47.5-30.5-84.7-76.1-111.5-136.9-26.8-60.8-41-134.5-42.5-221 1.7-92 18.3-172.9 50-242.8 31.3-69.6 75.7-129.6 133-179.9 57-50.3 124-89.9 201-118.8 77-29 160.9-43.2 252-42.5 91 .7 174.9 14.9 251.8 42.5 76.9 27.6 143.9 67.2 201 118.8 57.3 50.3 101.7 110.3 133 179.9 31.7 69.9 48.3 150.8 50 242.8 1.5 86.5-12.7 160.2-42.5 221-26.8 60.8-64 106.4-111.5 136.9-47.5 30.5-104.8 45.8-171.8 45.8-58.3 0-116.7-8.9-175-26.8-58.3-17.9-108.2-39.4-150-64.4-41.8-25-72.7-47.5-92.7-67.5-116.7 94.5-256.5 94.5-370 0-20-20-50.8-42.5-92.7-67.5-41.8-25-91.7-46.5-150-64.4-58.3-17.9-116.7-26.8-175-26.8zM42.5 765.7C-15.8 735.2-66 689.6-106.8 628.8-147.7 568-175 494.3-189.2 407.8-203.3 321.3-201 227-184.2 144.2-167.3 61.3-137.7 5.7-94.8-13-52-31.7-17.7-34.5 42.5-9.5c59.8 24.7 94 80.8 146.8 130 52.8 49.2 122.8 92.3 178 144.2 55.2 51.8 98.3 112.6 150 175 43 51 92.8 106.4 110.3 157.5 17.5 51.2-2.7 94.8-38.8 127.7-43.5 39.6-107.3 75.5-157.5 110.3-50.2 34.8-93.3 77.9-144.2 130-60.2 61.7-94.5 117.8-94.5 175.8 0 34.7 7.8 76.5 26.8 97.5 19 21 58.7 34 94.8 34 41.7 0 76.3-13 110.3-34 34-21 64.8-56.2 92.7-67.5 27.8-11.3 77.7-32.8 150-64.4 58.3-25 116.7-34 175-34 .8 0 1.7 0 2.5 0 17.3 30 51 45.5 78.8 68.2 28 22.8 50.8 51.3 68.2 77 34.7 51.7 77.7 87.5 110.3 127.7 38.7 47.5 75 100.7 111.5 136.9 36.5 36.1 78.3 53.7 125.3 71.5 47 17.8 100 26.8 159.2 26.8 59.2 0 112.2-9 159.2-26.8 47-17.8 88.8-35.4 125.3-71.5 36.5-36.1 72.8-89.4 111.5-136.9 32.6-40.2 75.6-76 110.3-127.7 17.5-51.1 61-106.6 110.3-157.8 51.7-62.4 94.8-123.2 150-175 55.2-51.8 125.2-95 178-144.2 52.8-49.2 87-105.3 146.8-130 60.2-25 94.5-22.2 42.5 9.5z'/%3E%3C/g%3E%3C/svg%3E");
    background-attachment: fixed;
    background-size: cover;
    background-position: center;
    /* Semi-transparent overlay for readability */
    background-color: rgba(255, 255, 255, 0.8); /* Increased opacity for better contrast */
}
/* --- Main Container Styling --- */
.main {
  background-color: rgba(255, 255, 255, 0.9); /* Higher opacity for content area */
  border-radius: 10px; /* More rounded corners */
  padding: 20px;  /* More internal spacing */
  margin-top: 20px; /* Some space from the top*/
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Subtle shadow for depth */

}


.title {
    color: #4A90E2; /* Blue */
    font-size: 42px; /* Larger title */
    font-weight: bold;
    margin-bottom: 30px; /* More space below title */
    text-align: center; /* Center the title */
}

.dream-entry {
    background-color: #F8F8F8; /* Light Gray */
    border: 1px solid #E1E1E1;
    border-radius: 8px; /* Slightly more rounded corners */
    padding: 15px;  /* More padding */
    margin-bottom: 15px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05); /* Subtle shadow */
}

.dream-analysis {
    background-color: #FFFFFF; /* White */
    border: 1px solid #DCDCDC;
    border-radius: 8px; /* Slightly more rounded */
    padding: 15px;
    margin-bottom: 15px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

.subheader {
    color: #2A6099; /* Darker Blue */
    font-size: 28px;  /* Larger subheader */
    font-weight: bold;
    margin-top: 20px;
    margin-bottom: 15px; /* More space below subheader */
}

.sentiment {
    font-size: 20px; /* Larger sentiment text */
}
/* Keep positive, negative, neutral as before */
.positive {
    color: #2ECC71; /* Green */
}
.negative {
    color: #E74C3C; /* Red */
}
.neutral {
    color: #3498DB; /* Blue */
}

.keyword {
    background-color: #d1ecf1; /* Lighter blue background */
    color: #0c5460; /* Darker text for contrast */
    border-radius: 15px; /* More rounded keywords */
    padding: 5px 10px; /* More padding */
    margin: 3px;
    display: inline-block;
    font-weight: bold; /* Make keywords bold */
}

.symbol {
    font-size: 18px;
    font-weight: bold;
}

.symbol-meaning {
    font-size: 16px;
    margin-left: 10px; /* Indent symbol meanings */
    color: #555;  /* Slightly darker color for meanings */
}
.dataframe {
  font-size:16px; /*Larger dataframe font*/
}

/* Style for buttons */
.stButton>button {
    color: #fff;
    background-color: #4A90E2;
    border-radius: 20px; /* Rounded buttons */
    padding: 10px 24px;
    font-weight: bold;
    border: none; /* Remove default border */
    transition: background-color 0.3s ease; /* Smooth transition on hover */
}

.stButton>button:hover {
    background-color: #2A6099; /* Darker blue on hover */
}

/* Style for text area */
textarea {
    border-radius: 8px !important; /* Rounded text area */
}

</style>
""", unsafe_allow_html=True)


# --- Streamlit App ---
st.markdown("<h1 class='title'>AI-Powered Dream Journal Analyzer</h1>", unsafe_allow_html=True)

dream_entry = st.text_area("Enter your dream here:", height=200, key="dream_input", placeholder="Describe your dream in as much detail as possible...")

if 'dream_data' not in st.session_state:
    st.session_state.dream_data = pd.DataFrame(columns=['Date', 'Dream', 'Processed Dream', 'Sentiment', 'Keywords', 'Entities', 'Symbols'])

if st.button("Analyze Dream", type="primary"):  # Use a primary button for emphasis
    if dream_entry:
        with st.spinner("Analyzing your dream..."):  # Show a spinner while processing
            processed_dream = preprocess_text(dream_entry)

            # Sentiment Analysis
            blob = TextBlob(processed_dream)
            sentiment_polarity = blob.sentiment.polarity
            sentiment_subjectivity = blob.sentiment.subjectivity

            if sentiment_polarity > 0.1:
                sentiment = "Positive"
                sentiment_class = "positive"
            elif sentiment_polarity < -0.1:
                sentiment = "Negative"
                sentiment_class = "negative"
            else:
                sentiment = "Neutral"
                sentiment_class = "neutral"

            # Theme/Topic Extraction (using NER)
            doc = nlp(dream_entry)
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            # Keyword Extraction
            keywords = extract_keywords(processed_dream)
            keyword_counts = pd.Series(keywords).value_counts().head(5)

            # Dream Symbol Interpretation
            found_symbols = []
            for symbol in dream_symbols:
                if symbol in processed_dream:
                    found_symbols.append(symbol)

            # --- Display Results (Enhanced) ---
            st.markdown("<div class='dream-analysis'>", unsafe_allow_html=True)  # Use the dream-analysis class
            st.markdown("<h2 class='subheader'>Dream Analysis:</h2>", unsafe_allow_html=True)

            st.markdown(f"<p class='sentiment {sentiment_class}'><b>Sentiment:</b> {sentiment} (Polarity: {sentiment_polarity:.2f}, Subjectivity: {sentiment_subjectivity:.2f})</p>", unsafe_allow_html=True)

            st.markdown("<h3 class='subheader'>Key Themes/Entities:</h3>", unsafe_allow_html=True)
            if entities:
                for entity, label in entities:
                    st.markdown(f"<span class='keyword'>{entity} ({label})</span>", unsafe_allow_html=True)
            elif 'keyword_counts' in locals():
                 for keyword, count in keyword_counts.items():
                      st.markdown(f"<span class='keyword'>{keyword} ({count})</span>", unsafe_allow_html=True)
            else:
                 st.write("No Keywords Found")

            st.markdown("<h3 class='subheader'>Possible Symbol Interpretations (Take with a grain of salt!):</h3>", unsafe_allow_html=True)
            if found_symbols:
                for symbol in found_symbols:
                    st.markdown(f"<p class='symbol'> {symbol}: </p> <p class = 'symbol-meaning'>{', '.join(dream_symbols[symbol])}</p>", unsafe_allow_html=True)

            else:
                st.write("No common dream symbols detected.")


            st.markdown("<h3 class='subheader'>Summary:</h3>", unsafe_allow_html=True)
            st.write(f"This dream had a {sentiment} sentiment and frequently mentioned: {', '.join(keyword_counts.index.tolist()) if 'keyword_counts' in locals() else 'No Keywords'}")

            st.markdown("</div>", unsafe_allow_html=True) # Close dream-analysis div


            # --- Data Storage ---
            new_entry = pd.DataFrame([{
                'Date': pd.to_datetime('today'),
                'Dream': dream_entry,
                'Processed Dream': processed_dream,
                'Sentiment': sentiment,
                'Keywords': list(keyword_counts.index) if 'keyword_counts' in locals() else [],
                'Entities': entities,
                'Symbols': found_symbols
            }])
            st.session_state.dream_data = pd.concat([st.session_state.dream_data, new_entry], ignore_index=True)

    else:
        st.warning("Please enter a dream to analyze.")

# --- Display Dream Journal ---
st.markdown("<h2 class='subheader'>Dream Journal Entries</h2>", unsafe_allow_html=True)
if not st.session_state.dream_data.empty:
    st.dataframe(st.session_state.dream_data, use_container_width=True) # Use full width

    # Download Data
    csv = st.session_state.dream_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        "Download Dream Journal",
        csv,
        "dream_journal.csv",
        "text/csv",
        key='download-csv'
    )
else:
    st.write("No dream entries yet.")