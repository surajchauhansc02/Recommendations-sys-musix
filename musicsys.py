import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="Music Recommendation System", layout="wide", page_icon="🎵")

st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: #ffffff;
    }
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 1px solid #282828;
    }
    .nav-item {
        padding: 10px;
        font-size: 16px;
        font-weight: bold;
        color: #b3b3b3;
    }
    .nav-item-active {
        color: #1DB954;
        background-color: #282828;
        border-radius: 4px;
    }
    .mix-card {
        background: linear-gradient(180deg, #282828 0%, #181818 100%);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    .player-container {
        background-color: #181818;
        border: 1px solid #282828;
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
    }
    .play-btn {
        background-color: #1DB954;
        color: white !important;
        padding: 10px 20px;
        border-radius: 20px;
        text-decoration: none;
        font-weight: bold;
        display: inline-block;
        margin-top: 10px;
        transition: background-color 0.2s;
    }
    .play-btn:hover {
        background-color: #1ed760;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = {
        'Song': [
            'Tum Hi Ho', 'Channa Mereya', 'Kesariya', 'Agar Tum Saath Ho', 'Ae Dil Hai Mushkil',
            'Tajdar-e-Haram', 'Dil Diyan Gallan', 'Tera Hone Laga Hoon', 'Jeena Jeena', 'Pehli Nazar Mein'
        ],
        'Artist': [
            'Arijit Singh', 'Arijit Singh', 'Arijit Singh', 'Arijit Singh', 'Arijit Singh',
            'Atif Aslam', 'Atif Aslam', 'Atif Aslam', 'Atif Aslam', 'Atif Aslam'
        ],
        'Album': [
            'Aashiqui 2', 'Ae Dil Hai Mushkil', 'Brahmastra', 'Tamasha', 'Ae Dil Hai Mushkil',
            'Coke Studio S8', 'Tiger Zinda Hai', 'Ajab Prem Ki Ghazab Kahani', 'Badlapur', 'Race'
        ],
        'Duration': ['4:22', '4:49', '4:28', '5:41', '4:29', '10:02', '4:20', '5:00', '3:49', '5:14'],
        'Genre': [
            'Romantic Pop', 'Sad Romantic', 'Bollywood Melodic', 'Emotional Drama', 'Sufi Rock',
            'Qawwali Sufi', 'Romantic Dance', 'Bollywood Pop', 'Melodic Acoustic', 'Club Romantic'
        ],
        'Embed_ID': [
            'UNBkn68G9c4', '284Ov7ysmfA', 'BddP6PYo2Gs', 'sK7riqg2mr4', '6FURuLYrR_Q',
            'a18py61_F_w', 'SAcpESN_Fk4', 'r6tV1z6YskA', 'zFdi8M1vZ80', 'VzVLeL-Z-tY'
        ],
        
        'Web_URL': [
            'https://music.youtube.com/watch?v=UNBkn68G9c4',
            'https://music.youtube.com/watch?v=284Ov7ysmfA',
            'https://music.youtube.com/watch?v=BddP6PYo2Gs',
            'https://music.youtube.com/watch?v=sK7riqg2mr4',
            'https://music.youtube.com/watch?v=6FURuLYrR_Q',
            'https://music.youtube.com/watch?v=a18py61_F_w',
            'https://music.youtube.com/watch?v=SAcpESN_Fk4',
            'https://music.youtube.com/watch?v=r6tV1z6YskA',
            'https://music.youtube.com/watch?v=zFdi8M1vZ80',
            'https://music.youtube.com/watch?v=VzVLeL-Z-tY'
        ]
    }
    return pd.DataFrame(data)

df = load_data()


def get_recommendations(song_title, df):
    df['features'] = df['Artist'] + " " + df['Genre']
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df['features'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    
    try:
        idx = df[df['Song'] == song_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        song_indices = [i[0] for i in sim_scores if i[0] != idx][:4]
        return df.iloc[song_indices]
    except:
        return df.head(4)

with st.sidebar:
    st.title("🎵 Music Recs")
    st.markdown('<div class="nav-item nav-item-active">🏠 Home</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">🔍 Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">📚 Library</div>', unsafe_allow_html=True)

col1, col2 = st.columns([4, 1])
with col1:
    search_query = st.selectbox("Select a track to listen:", df['Song'].tolist())
with col2:
    st.markdown("🌐 **Aaditya** ▾")

st.header("Hello, Aaditya")

st.subheader("Made for you")
mix_cols = st.columns(5)
mixes = ["Arijit Mix 1", "Atif Mix 1", "Sufi Session", "Bollywood 2000s", "Romantic Melodies"]
for idx, col in enumerate(mix_cols):
    with col:
        st.markdown(f'<div class="mix-card"><h4 style="color:#1DB954;">{mixes[idx]}</h4></div>', unsafe_allow_html=True)

st.write("---")

selected_song_info = df[df['Song'] == search_query].iloc[0]

st.subheader("Recommended For You")
recommended_df = get_recommendations(search_query, df)
for index, row in recommended_df.iterrows():
    col_s, col_a, col_al = st.columns([4, 4, 2])
    with col_s: st.markdown(f"▶️ **{row['Song']}**")
    with col_a: st.markdown(f"<span style='color:#b3b3b3'>{row['Artist']}</span>", unsafe_allow_html=True)
    with col_al: st.markdown(f"<span style='color:#b3b3b3'>{row['Duration']}</span>", unsafe_allow_html=True)

st.write("---")
st.subheader("🎵 Now Playing Console")

embed_html = f"""
<iframe width="100%" height="180" src="https://www.youtube.com/embed/{selected_song_info['Embed_ID']}?rel=0" 
title="YouTube video player" frameborder="0" 
allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" 
allowfullscreen style="border-radius:12px;"></iframe>
"""
st.components.v1.html(embed_html, height=200)

st.markdown(f"""
<div class="player-container">
    <h4>Having trouble with the player box above?</h4>
    <p style="color:#b3b3b3; font-size:14px;">Some mobile browsers aggressively block inline players. Click the button below to stream the track perfectly on YouTube Music instantly.</p>
    <a class="play-btn" href="{selected_song_info['Web_URL']}" target="_blank">🚀 Launch High-Quality Stream</a>
</div>
""", unsafe_allow_html=True)
            
