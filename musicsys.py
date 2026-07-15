import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# 1. Page Configuration & Custom Styling
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
    .player-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: #181818;
        border-top: 1px solid #282828;
        padding: 15px;
        z-index: 9999;
    }
    /* Style the HTML5 audio element globally */
    audio {
        width: 100%;
        border-radius: 30px;
        background: #282828;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Database with Fully Open, Non-Restricted Audio Streams
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
        # Safe public domain tracks used as stable endpoints to bypass API/platform cross-origin security
        'Audio_URL': [
            'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3',
            'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-2.mp3',
            'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-3.mp3',
            'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-4.mp3',
            'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-5.mp3',
            'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-6.mp3',
            'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-7.mp3',
            'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-8.mp3',
            'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-9.mp3',
            'https://www.soundhelix.com/examples/mp3/SoundHelix-Song-10.mp3'
        ]
    }
    return pd.DataFrame(data)

df = load_data()

# 3. Recommendation System logic
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

# --- SIDEBAR UI ---
with st.sidebar:
    st.title("🎵 Music Recs")
    st.markdown('<div class="nav-item nav-item-active">🏠 Home</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">🔍 Search</div>', unsafe_allow_html=True)
    st.markdown('<div class="nav-item">📚 Library</div>', unsafe_allow_html=True)

# --- MAIN PAGE UI ---
col1, col2 = st.columns([4, 1])
with col1:
    search_query = st.selectbox("Select a track to load:", df['Song'].tolist())
with col2:
    st.markdown("🌐 **Suraj** ▾")

st.header("Good Evening, Suraj")

st.subheader("Made for you")
mix_cols = st.columns(5)
mixes = ["Arijit Mix 1", "Atif Mix 1", "Sufi Session", "Bollywood 2000s", "Romantic Melodies"]
for idx, col in enumerate(mix_cols):
    with col:
        st.markdown(f'<div class="mix-card"><h4 style="color:#1DB954;">{mixes[idx]}</h4></div>', unsafe_allow_html=True)

st.write("---")

selected_song_info = df[df['Song'] == search_query].iloc[0]

# Recommendations Grid
st.subheader("Recommended For You")
recommended_df = get_recommendations(search_query, df)
for index, row in recommended_df.iterrows():
    col_s, col_a, col_al = st.columns([4, 4, 2])
    with col_s: st.markdown(f"▶️ **{row['Song']}**")
    with col_a: st.markdown(f"<span style='color:#b3b3b3'>{row['Artist']}</span>", unsafe_allow_html=True)
    with col_al: st.markdown(f"<span style='color:#b3b3b3'>{row['Duration']}</span>", unsafe_allow_html=True)

# --- NATIVE EMBED PLAYER SYSTEM ---
st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)

# Directly leveraging HTML5 video components via st.components to completely bypass player walls
player_html = f"""
<div style="background-color: #181818; padding: 20px; border-radius: 12px; border: 1px solid #282828; font-family: sans-serif;">
    <div style="margin-bottom: 12px;">
        <span style="color: #ffffff; font-weight: bold; font-size: 16px;">Now Streaming: {selected_song_info['Song']}</span><br>
        <span style="color: #1DB954; font-size: 13px;">{selected_song_info['Artist']} — {selected_song_info['Album']}</span>
    </div>
    <audio controls autoplay src="{selected_song_info['Audio_URL']}" style="width: 100%;"></audio>
</div>
"""
st.components.v1.html(player_html, height=150)
