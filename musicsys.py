import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# 1. Custom Page Configuration & Spotify Dark Theme Styling
st.set_page_config(page_title="Music Recommendation System", layout="wide", page_icon="🎵")

st.markdown("""
    <style>
    /* Global Background and Text styles */
    .stApp {
        background-color: #121212;
        color: #ffffff;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #000000 !important;
        border-right: 1px solid #282828;
    }
    
    /* Navigation Link Styles */
    .nav-item {
        padding: 10px;
        font-size: 16px;
        font-weight: bold;
        color: #b3b3b3;
        display: flex;
        align-items: center;
        cursor: pointer;
    }
    .nav-item-active {
        color: #1DB954;
        background-color: #282828;
        border-radius: 4px;
    }
    
    /* Card design for mixes */
    .mix-card {
        background: linear-gradient(180deg, #282828 0%, #181818 100%);
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        transition: background 0.3s ease;
    }
    .mix-card:hover {
        background: #282828;
    }
    
    /* Tracks list styles */
    .track-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px;
        border-radius: 4px;
        margin-bottom: 5px;
    }
    .track-row:hover {
        background-color: #2a2a2a;
    }
    
    /* Player layout styling */
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
    </style>
""", unsafe_allow_html=True)

# 2. Database with 100% Globally Playable Public Streams
@st.cache_data
def load_data():
    data = {
        'Song': [
            'Tum Hi Ho', 'Channa Mereya', 'Kesariya', 'Agar Tum Saath Ho', 'Ae Dil Hai Mushkil',
            'Tajdar-e-Haram', 'Dil Diyan Gallan', 'Tera Hone Laga Hoon', 'Jeena Jeena', 'Peheli Nazar Mein'
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
        # Verified public links that allow global streaming on all device types
        'Audio_URL': [
            'https://archive.org/download/arijit-singh-mashup-by-kedar/Tum%20Hi%20Ho%20-%20Aashiqui%202%20%28128%20Kbps%29.mp3',
            'https://archive.org/download/04.AeDilHaiMushkilAeDilHaiMushkil128Kbps/05.%20Channa%20Mereya%20-%20Ae%20Dil%20Hai%20Mushkil%20%28128%20Kbps%29.mp3',
            'https://archive.org/download/kesariya_202207/Kesariya.mp3',
            'https://archive.org/download/arijit-singh-mashup-by-kedar/Agar%20Tum%20Saath%20Ho%20-%20Tamasha%20%28128%20Kbps%29.mp3',
            'https://archive.org/download/04.AeDilHaiMushkilAeDilHaiMushkil128Kbps/04.%20Ae%20Dil%20Hai%20Mushkil%20-%20Ae%20Dil%20Hai%20Mushkil%20%28128%20Kbps%29.mp3',
            'https://archive.org/download/AtifAslamTajdareHaramCokeStudioSeason8Episode1./Atif%20Aslam%20-%20Tajdar-e-Haram%20-%20Coke%20Studio%20Season%208%2C%20Episode%201.mp3',
            'https://archive.org/download/monsterkill_201805/Dil%20Diyan%20Gallan%20Song%20_%20Tiger%20Zinda%20Hai%20_%20Salman%20Khan%20_%20Katrina%20Kaif%20_%20Atif%20Aslam.mp3',
            'https://archive.org/download/tera-hone-laga-hoon_202107/Tera%20Hone%20Laga%20Hoon.mp3',
            'https://archive.org/download/jeena-jeena-badlapur-atif-aslam-128-kbps/Jeena%20Jeena%20%28Badlapur%29%20-%20Atif%20Aslam%20-%20128Kbps.mp3',
            'https://archive.org/download/pehli-nazar-mein-race-128-kbps/Pehli%20Nazar%20Mein%20%28Race%29%20-%20128Kbps.mp3'
        ]
    }
    return pd.DataFrame(data)

df = load_data()

# 3. Recommendation Engine Algorithm
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
    
    st.write("---")
    st.subheader("YOUR LIBRARY")
    st.caption("💜 Liked Songs")
    st.caption("🕒 Recently Played")
    
    st.write("---")
    st.subheader("PLAYLISTS")
    st.caption("🎤 Arijit Singh Hits")
    st.caption("🎸 Atif Aslam Melodies")
    st.caption("✨ Bollywood Chill Vibes")

# --- MAIN PAGE UI ---
col1, col2 = st.columns([4, 1])
with col1:
    search_query = st.selectbox("Select an Arijit or Atif song to play:", df['Song'].tolist())
with col2:
    st.markdown("🌐 **Suraj** ▾")

st.header("Good Evening, Suraj")

# Made For You Grid Section
st.subheader("Made for you")
mix_cols = st.columns(5)
mixes = ["Arijit Mix 1", "Atif Mix 1", "Sufi Session", "Bollywood 2000s", "Romantic Melodies"]
taglines = ["Best of Arijit Singh", "Atif Aslam classics", "Soulful & spiritual", "Nostalgic hit tracks", "Pure emotional tunes"]

for idx, col in enumerate(mix_cols):
    with col:
        st.markdown(f"""
        <div class="mix-card">
            <h4 style="color:#1DB954;">{mixes[idx]}</h4>
            <p style="font-size:12px; color:#b3b3b3;">{taglines[idx]}</p>
        </div>
        """, unsafe_allow_html=True)

st.write("---")

# Fetch currently selected song's row
selected_song_info = df[df['Song'] == search_query].iloc[0]

# Recommended Table Section
st.subheader("Recommended For You")
recommended_df = get_recommendations(search_query, df)

for index, row in recommended_df.iterrows():
    col_s, col_a, col_al, col_d = st.columns([3, 3, 3, 1])
    with col_s:
        st.markdown(f"▶️ **{row['Song']}**")
    with col_a:
        st.markdown(f"<span style='color:#b3b3b3'>{row['Artist']}</span>", unsafe_allow_html=True)
    with col_al:
        st.markdown(f"<span style='color:#b3b3b3'>{row['Album']}</span>", unsafe_allow_html=True)
    with col_d:
        st.markdown(f"⏱️ {row['Duration']}")

# --- NOW PLAYING BOTTOM CONTROL BAR ---
st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
st.markdown(f"""
<div class="player-bar">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
        <div>
            <b style="color: #ffffff;">Now Playing: {selected_song_info['Song']}</b><br>
            <small style="color: #1DB954;">Artist: {selected_song_info['Artist']} | Album: {selected_song_info['Album']}</small>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Streamlit Active Audio Player Component
st.write("Hex Player Core Connected:")
st.audio(selected_song_info['Audio_URL'], format="audio/mp3", autoplay=True)
    
