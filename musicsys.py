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

# 2. Database with Stable Streaming CDN Tracks
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
        # Using bulletproof, open-access fallback audio feeds to bypass browser safety blocks
        'Audio_URL': [
            'https://actions.google.com/sounds/v1/ambiences/morning_birds.ogg',
            'https://actions.google.com/sounds/v1/ambiences/coffee_shop_atmosphere.ogg',
            'https://actions.google.com/sounds/v1/ambiences/outdoor_market_atmosphere.ogg',
            'https://actions.google.com/sounds/v1/ambiences/rain_heavy_loud.ogg',
            'https://actions.google.com/sounds/v1/water/sea_waves.ogg',
            'https://actions.google.com/sounds/v1/ambiences/fire_crackle.ogg',
            'https://actions.google.com/sounds/v1/ambiences/wind_howling.ogg',
            'https://actions.google.com/sounds/v1/transportation/subway_train.ogg',
            'https://actions.google.com/sounds/v1/sports/gymnasium.ogg',
            'https://actions.google.com/sounds/v1/alarms/digital_watch_alarm_long.ogg'
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
st.write("🔊 **Streaming Stream Active:**")
st.audio(selected_song_info['Audio_URL'], format="audio/ogg", autoplay=True)
        
