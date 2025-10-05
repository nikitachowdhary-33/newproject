import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("🧠 Fake News Detector")
st.write("Enter a news headline and article text below to check if it's **REAL** or **FAKE**.")

# ------------------------
# 🧩 1. Embedded dataset
# ------------------------
data = """title,text,label
ISRO launches new satellite for weather monitoring,The Indian Space Research Organisation successfully launched a satellite to improve weather forecasting and disaster management,REAL
Government announces new policy for electric vehicles,India’s transport ministry introduced incentives to boost adoption of electric vehicles and reduce pollution,REAL
IIT Delhi develops low-cost ventilator prototype,Engineers at IIT Delhi designed an affordable ventilator to help hospitals tackle respiratory illnesses,REAL
Hyderabad Metro expands with new corridor,The Hyderabad Metro Rail inaugurated a new line connecting IT hubs to residential areas,REAL
Indian startup secures funding for agri-tech solution,A Bengaluru-based startup raised funding to develop AI solutions for improving crop yields,REAL
Mumbai coastal road project faces delay,The ambitious coastal road project in Mumbai faces delays due to environmental clearance issues,REAL
Kerala sets record in literacy campaign,Kerala launched a new literacy program aiming to make the entire state fully digitally literate,REAL
Indian Railways introduces hydrogen-powered train,The Indian Railways unveiled its first hydrogen-powered train to promote clean energy in transport,REAL
Supreme Court delivers verdict on data privacy,India’s Supreme Court reaffirmed the right to privacy as a fundamental right in a landmark case,REAL
Delhi government launches free Wi-Fi hotspots,The Delhi government activated 500 new Wi-Fi hotspots across the city to improve internet accessibility,REAL
Taj Mahal to be demolished for shopping mall!,A viral hoax falsely claims that the Indian government plans to demolish the Taj Mahal to build a shopping mall,FAKE
India bans rain to prevent floods!,A fabricated story alleges that the Indian government has developed technology to ban rainfall to stop flooding,FAKE
PM secretly buys palace in Dubai with taxpayer money!,A baseless claim circulates online accusing India’s Prime Minister of secretly purchasing luxury property abroad,FAKE
Bollywood actress reveals aliens run Indian cinema!,A conspiracy blog falsely claims a popular Bollywood actress said extraterrestrials control the film industry,FAKE
COVID vaccine turns people into monkeys in India!,A false article spreads misinformation that India’s COVID vaccine causes bizarre physical changes,FAKE
ISRO faked Chandrayaan-3 landing with Hollywood set!,A conspiracy theory alleges India’s lunar mission was staged in a film studio,FAKE
India to sell Kashmir to China for debt relief!,A fabricated story falsely claims India plans to sell Kashmir to China to repay international loans,FAKE
Ganga river turns into milk overnight!,A viral hoax claims the sacred Ganga river transformed into milk as a “divine sign”,FAKE
RBI printing ₹10 lakh notes secretly,Fake reports allege the Reserve Bank of India is secretly printing ultra-high denomination currency notes,FAKE
Delhi Metro to connect directly to London!,A sensational article falsely claims Delhi Metro is building an underwater tunnel to London,FAKE
"""

# Convert embedded CSV text to DataFrame
from io import StringIO
df = pd.read_csv(StringIO(data))

# ------------------------
# ⚙️ 2. Train Model
# ------------------------
X = df['title'] + " " + df['text']
y = df['label']

vectorizer = TfidfVectorizer(stop_words='english', max_features=300)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# ------------------------
# 💬 3. User Input
# ------------------------
title = st.text_input("📰 News Title")
text = st.text_area("🧾 News Content", height=150)

if st.button("Check News"):
    if title.strip() == "" or text.strip() == "":
        st.warning("⚠️ Please enter both title and content.")
    else:
        input_text = [title + " " + text]
        input_vec = vectorizer.transform(input_text)
        prediction = model.predict(input_vec)[0]

        if prediction == "REAL":
            st.success("✅ This news seems **REAL**.")
        else:
            st.error("🚫 This news seems **FAKE**.")

# ------------------------
# 📊 4. Accuracy Display (optional)
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.caption(f"Model accuracy on sample dataset: **{acc:.2f}**")
