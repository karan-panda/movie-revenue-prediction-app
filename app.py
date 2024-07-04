import streamlit as st
import pandas as pd
from pycaret.regression import load_model, predict_model
from datetime import date

# Define all possible genres, languages, and countries
all_genres = ['Horror', 'Comedy', 'Mystery', 'Action', 'Western', 'Animation', 'Adventure', 
        'Thriller', 'TV Movie', 'Romance', 'Science Fiction', 'History', 'Drama', 'War', 
        'Fantasy', 'Crime', 'Documentary', 'Music', 'Family'
    ]

all_languages = ['Afrikaans', 'Bahasa indonesia', 'Dansk', 'Deutsch', 'English',
       'Español', 'Français', 'Gaeilge', 'Italiano', 'Kiswahili', 'Latin',
       'Magyar', 'Malti', 'No Language', 'Norsk', 'Polski', 'Português',
       'Pусский', 'Română', 'Tiếng Việt', 'Türkçe', 'svenska', 'Íslenska',
       'ελληνικά', 'Український', 'български език', 'עִבְרִית', 'اردو',
       'العربية', 'فارسی', 'پښتو', 'हिन्दी', 'ภาษาไทย', '广州话 / 廣州話', '日本語',
       '普通话', '한국어/조선말', 'Bahasa melayu', 'Bosanski', 'Català', 'Cymraeg', 'Eesti',
       'Esperanto', 'Fulfulde', 'Hrvatski', 'Lietuvių', 'Nederlands',
       'Slovenčina', 'Somali', 'Srpski', '[]', 'euskera', 'isiZulu', 'shqip',
       'suomi', 'Český', 'বাংলা', 'ਪੰਜਾਬੀ', 'தமிழ்', 'සිංහල', 'ქართული'
    ]

all_countries = ['Albania', 'Aruba', 'Australia', 'Belgium', 'Brazil', 'Bulgaria',
       'Canada', 'Chile', 'China', 'Colombia', 'Cyprus', 'Czech Republic',
       'Denmark', 'Finland', 'France', 'Germany', 'Greece', 'Hong Kong',
       'Hungary', 'India', 'Ireland', 'Israel', 'Italy', 'Japan', 'Jordan',
       'Luxembourg', 'Macao', 'Malaysia', 'Mexico', 'Morocco', 'Namibia',
       'Netherlands', 'New Zealand', 'Nigeria', 'Palestinian Territory',
       'Poland', 'Portugal', 'Puerto Rico', 'Qatar', 'Romania', 'Russia',
       'Serbia', 'Singapore', 'South Africa', 'Spain', 'Sweden', 'Switzerland',
       'Thailand', 'Tunisia', 'Ukraine', 'United Arab Emirates',
       'United Kingdom', 'United States of America', 'Yugoslavia', 'Austria', 'Bahamas',
       'Botswana', 'Dominican Republic', 'Iceland', 'Indonesia', 'Jamaica',
       'Kenya', 'Kuwait', 'Lebanon', 'Libyan Arab Jamahiriya', 'Malta',
       'Montenegro', 'Norway', 'Peru', 'Saudi Arabia', 'Slovakia', 'Slovenia',
       'South Korea', 'Soviet Union', 'Taiwan', 'Turkey', 'Venezuela', 'Zimbabwe',
       'Unnamed: 131', 'Unnamed: 0'
    ]


def create_empty_movie_dict():
    movie_dict = {
        'id': 0,
        'budget': 0,
        'homepage': '',
        'imdb_id': '',
        'original_language': 0,
        'movie_title': '',
        'overview': '',
        'popularity': 0.0,
        'poster_path': '',
        'production_companies': '',
        'runtime': 0,
        'status': 0,
        'tagline': '',
        'title': '',
        'keywords': '',
        'cast': '',
        'crew': '',
        'release_year': date.today().year,
        'release_month': date.today().month,
        'release_day': date.today().day
    }

    for genre in all_genres:
        movie_dict[genre] = 0

    for language in all_languages:
        movie_dict[language] = 0

    for country in all_countries:
        movie_dict[country] = 0

    return movie_dict

# Loading the model
@st.cache(allow_output_mutation=True)
def load_regression_model():
    return load_model('tuned_elastic_net')

movie_dict = create_empty_movie_dict()
st.title("Movie Revenue Prediction App")
st.markdown('<h5 style="text-align: right;">Made by Karan Panda</h5>', unsafe_allow_html=True)

with st.form("movie_form"):
    movie_dict['id'] = st.number_input("ID", min_value=0)
    movie_dict['budget'] = st.number_input("Budget", min_value=0)
    movie_dict['homepage'] = st.text_input("Homepage")
    movie_dict['imdb_id'] = st.text_input("IMDB ID")
    
    # Update original_language to accept string input
    movie_dict['original_language'] = st.selectbox("Original Language", ["en", "Other"])
    
    movie_dict['movie_title'] = st.text_input("Movie Title")
    movie_dict['overview'] = st.text_area("Overview")
    movie_dict['popularity'] = st.number_input("Popularity", min_value=0.0)
    movie_dict['poster_path'] = st.text_input("Poster Path")
    movie_dict['production_companies'] = st.text_input("Production Companies")
    movie_dict['runtime'] = st.number_input("Runtime", min_value=0)
    
    # Update status to accept string input
    movie_dict['status'] = st.selectbox("Status", ["Released", "Not Released"])
    
    movie_dict['tagline'] = st.text_input("Tagline")
    movie_dict['title'] = st.text_input("Title")
    movie_dict['keywords'] = st.text_input("Keywords")
    movie_dict['cast'] = st.text_area("Cast")
    movie_dict['crew'] = st.text_area("Crew")
    movie_dict['revenue'] = None,
    movie_dict['release_year'] = st.number_input("Release Year", min_value=1900, value=date.today().year)
    movie_dict['release_month'] = st.number_input("Release Month", min_value=1, max_value=12, value=date.today().month)
    movie_dict['release_day'] = st.number_input("Release Day", min_value=1, max_value=31, value=date.today().day)

    selected_genres = st.multiselect("Genres", all_genres)
    selected_languages = st.multiselect("Languages", all_languages)
    selected_countries = st.multiselect("Countries", all_countries)

    submit_button = st.form_submit_button(label="Submit")

if submit_button:
    # Update selected genres, languages and countries to 1
    for genre in selected_genres:
        movie_dict[genre] = 1

    for language in selected_languages:
        movie_dict[language] = 1

    for country in selected_countries:
        movie_dict[country] = 1

    # Encode original_language
    movie_dict['original_language'] = 1 if movie_dict['original_language'] == "en" else 0

    # Encode status
    movie_dict['status'] = 1 if movie_dict['status'] == "Released" else 0

    model = load_regression_model()
    data_df = pd.DataFrame([movie_dict])
    print(data_df.shape)
    new_prediction = predict_model(model, data=data_df)

    predicted_revenue = new_prediction['prediction_label'][0]
    st.markdown(f"### **Predicted Revenue:** ${predicted_revenue:,.2f}")