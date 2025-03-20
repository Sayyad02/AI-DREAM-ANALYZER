# AI-Powered Dream Journal Analyzer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url-here.streamlit.app)  This Streamlit application provides an interactive platform for analyzing your dreams using Natural Language Processing (NLP) techniques.  It helps you record, analyze, and gain insights into your dream content.

## Features

*   **Dream Entry:**  A text area for recording your dreams in detail.
*   **Automated Analysis:**
    *   **Text Preprocessing:**  Lowercasing, stop word removal, and lemmatization to prepare the dream text for analysis.
    *   **Sentiment Analysis:**  Determines the overall emotional tone of the dream (Positive, Negative, or Neutral) using TextBlob.
    *   **Named Entity Recognition (NER):**  Identifies and categorizes key entities in the dream (people, places, organizations, etc.) using spaCy.
    *   **Keyword Extraction:**  Extracts the most important keywords from the dream text using part-of-speech tagging.
    *   **Dream Symbol Interpretation:**  Provides potential interpretations of common dream symbols based on a built-in dictionary.
*   **Dream Journal:** Stores your dream entries and analysis results in a Pandas DataFrame.
*   **Data Download:** Allows you to download your dream journal data as a CSV file.
*   **Interactive UI:**  A clean and user-friendly interface built with Streamlit.
*   **Visualizations:** (Add details about any visualizations you include, e.g., word clouds, sentiment over time, etc.)

## How to Run (Locally)

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)  # Replace with your repository URL
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Download NLTK Data and spaCy Model (One-Time):**

    ```bash
    python -m nltk.downloader punkt
    python -m nltk.downloader stopwords
    python -m nltk.downloader wordnet
    python -m nltk.downloader averaged_perceptron_tagger
    python -m spacy download en_core_web_sm
    ```

4.  **Run the Streamlit App:**

    ```bash
    streamlit run app.py
    ```

    This will open the app in your web browser.

## How to Use

1.  **Enter Your Dream:** Type or paste your dream into the text area. Be as detailed as possible.
2.  **Analyze:** Click the "Analyze Dream" button.
3.  **View Results:** The app will display the sentiment, key entities, keywords, and potential symbol interpretations.
4.  **Review Your Journal:**  Your dream entries and analyses are stored in a table below the input area.
5.  **Download Data:** Click the "Download Dream Journal" button to save your data as a CSV file.

## Technologies Used

*   **Python:** The primary programming language.
*   **Streamlit:**  Framework for building the interactive web application.
*   **Pandas:**  Used for data storage and manipulation.
*   **NLTK:**  Natural Language Toolkit for text processing (tokenization, stemming, lemmatization, POS tagging).
*   **spaCy:**  Library for advanced Natural Language Processing (Named Entity Recognition).
*   **TextBlob:**  Library for simplified text processing and sentiment analysis.
*   **Matplotlib/Seaborn:** (If you added visualizations) Libraries for creating visualizations.

## Dream Symbol Dictionary

The app includes a basic dictionary of common dream symbols and their potential meanings.  This dictionary is a starting point, and dream interpretation is highly subjective. It's important to consider your own personal associations with the symbols.  (You could also add a link to a more comprehensive dream dictionary here, if you want.)

## Contributing

Contributions are welcome! If you'd like to improve the app, please fork the repository and submit a pull request.  Areas for potential improvement:

*   Expanding the dream symbol dictionary.
*   Adding more sophisticated NLP techniques (e.g., topic modeling, word embeddings).
*   Improving the user interface and visualizations.
*   Adding user accounts and persistent storage.
*   Adding more robust error handling.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. (You'll need to create a LICENSE file - a standard MIT license is a good choice for open-source projects).

## Disclaimer

This application is intended for informational and entertainment purposes only.  It is not a substitute for professional psychological advice or therapy. Dream interpretation is subjective, and the results provided by this app should be considered as potential insights, not definitive answers.
