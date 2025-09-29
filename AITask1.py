# STEP 1: INSTALL ALL NECESSARY PACKAGES

import subprocess
import sys
import os

# List of required packages
packages = [
    "flask",
    "flask-cors",
    "pdfminer.six",
    "scikit-learn",
    "pandas",
    "numpy",
    "google-generativeai",
    "pyngrok"  # For ngrok integration
]

print("⏳ Installing required packages...")
# Install packages quietly to keep the output clean
for package in packages:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install {package}. Please install it manually and try again.")
        sys.exit(1)
print("✅ All packages installed successfully.")

# STEP 2: IMPORT LIBRARIES AND GET API KEYS
from getpass import getpass
import pandas as pd
import numpy as np
import math
import textwrap
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text
from werkzeug.utils import secure_filename
import google.generativeai as genai

# Securely get API keys from user input using getpass
# This prevents keys from being displayed on the screen
try:
    gemini_api_key = os.environ.get('GEMINI_API_KEY') or getpass("🔑 Enter your Google Gemini API Key: ")
    ngrok_authtoken = os.environ.get('NGROK_AUTHTOKEN') or getpass("🔑 Enter your ngrok Authtoken: ")
except (KeyboardInterrupt, EOFError):
    print("\n\nOperation cancelled. Exiting.")
    sys.exit(0)

# Set the keys as environment variables for the current session
os.environ['GEMINI_API_KEY'] = gemini_api_key
os.environ['NGROK_AUTHTOKEN'] = ngrok_authtoken

# STEP 3: CREATE PROJECT FILES AND DIRECTORIES
print("\n⏳ Creating project files and directories...")
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('uploads', exist_ok=True)

with open('internships.csv', 'w', encoding='utf-8') as f:
    f.write("""id,title,organization,city,state,lat,lon,description,required_skills,sector,stipend,accommodation_estimate,link
1,Product Management Intern,SmartGov,New Delhi,Delhi,28.6139,77.2090,"Work with PM team to define features, create specs, and track pilots.","product management;communication;excel;research",Governance,10000,3000,https://internshala.com/internship/detail/product-management-internship-in-bangalore-at-intugine-technologies-private-limited1756816826
2,UX Research Intern,DesignForAll,Bengaluru,Karnataka,12.9716,77.5946,"Conduct field research, usability studies in rural contexts.","research;ux;fieldwork;interviewing",Design,8000,2500,https://www.google.com/search?q=UX+Research+Intern&rlz=1C1UEAD_enIN1074IN1078&oq=UX+Research+Intern&gs_lcrp=EgZjaHJvbWUyCQgAEEUYORiABDIHCAEQABiABDIHCAIQABiABDIHCAMQABiABDIHCAQQABiABDIHCAUQABiABDIHCAYQABiABDIHCAcQABiABDIHCAgQABiABDIHCAkQABiABNIBBzYxMGowajSoAgCwAgA&sourceid=chrome&ie=UTF-8&udm=8#vhid=vt%3D20/docid%3DRahAovpnSyoZQmkJAAAAAA%3D%3D&vssid=jobs-detail-viewer
3,Data Analyst Intern,AgriTech,Patna,Bihar,25.5941,85.1376,"Analyze crop-data, build dashboards and reports.","python;pandas;sql;excel",Data,9000,2000,https://in.prosple.com/data-analyst-internships-india
4,Monitoring & Evaluation Intern,RuralWorks,Raipur,Chhattisgarh,21.2514,81.6296,"Support M&E activities, KPI tracking, community surveys.","survey;excel;analytics;communication",M&E,7000,1800,https://jobs.undp.org/cj_view_job.cfm?cur_job_id=91917
5,Outreach Intern,YouthConnect,Shillong,Meghalaya,25.5788,91.8933,"Community outreach and stakeholder coordination in tribal districts.","communication;community;local-language",Outreach,6000,1500,https://internshala.com/jobs/outreach-worker-jobs-in-chennai/
""")

# Write the HTML template for the frontend
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" /><meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Internship Recommender</title><link rel="stylesheet" href="/static/style.css" />
</head>
<body>
  <section class="hero">
    <h1 class="hero-title">AI-Based Internship Allocation Engine for PM Internship Scheme</h1>
    <p class="hero-intro">
        Welcome to Internship Allocation Engine! An AI powered system that helps in allocating suitable internships under the PM Internship Scheme by analysing student profiles, skills and preferences. Scroll down to get personalized recommendations in seconds.
     </p>
  </section>
  <div class="container">
    <header>
      <h1 data-key="headerTitle">Internship Helper</h1>
      <div class="lang-switch">
        <select id="lang">
          <option value="en">English</option>
          <option value="hi">हिन्दी</option>
          <option value="ta">தமிழ்</option>
        </select>
      </div>
    </header>
    <main>
      <div class="form-container">
        <h2 id="results-title" data-key="formTitle">Find Your Internship</h2>
        <form id="profile-form" class="profile-form">
          <label data-key="educationLabel">Education (short)</label>
          <input id="education" data-key-placeholder="educationPlaceholder" placeholder="E.g., B.Tech (CS), 3rd year" />
          <label data-key="skillsLabel">Skills (semicolon separated)</label>
          <input id="skills" data-key-placeholder="skillsPlaceholder" placeholder="python;excel;communication" />
          <label data-key="interestsLabel">Interests / Sector</label>
          <input id="interests" data-key-placeholder="interestsPlaceholder" placeholder="e.g., Data, Design, Outreach" />
          <label data-key="cityLabel">Your City (for distance)</label>
          <input id="city" data-key-placeholder="cityPlaceholder" placeholder="e.g., New Delhi" />
          <label data-key="resumeLabel">Or upload resume (PDF)</label>
          <input type="file" id="resume" accept="application/pdf" />
          <div class="actions">
            <button type="button" id="upload-resume" data-key="uploadBtn">Upload Resume</button>
            <button type="button" id="get-reco" data-key="recoBtn">Get Recommendations</button>
          </div>
        </form>
      </div>
        <div class="results-container">
            <h2 data-key="recoTitle">Recommendations</h2>
            <div id="results" class="results"></div>
        </div>
    </main>
    <footer><small data-key="footerText">Designed for ease of use and accessibility.</small></footer>
  </div>
  <div id="chatbot-button">🤖</div>
  <div id="chatbot-panel">
    <div id="chatbot-header"><span data-key="chatbotHeader">Internship Assistant</span><button id="chatbot-close">&times;</button></div>
    <div id="chatbot-body">
      <div class="chat-log" id="chat-log"><div class="chat-bot" data-key="chatbotWelcome">Hello! How can I help you find an internship today?</div></div>
      <div class="chat-input"><input type="text" id="chat-input" data-key-placeholder="chatbotPlaceholder" placeholder="Ask me anything..." /><button id="chat-send">➤</button></div>
    </div>
  </div>
<script>
const strings = {
    en: {
        headerTitle: "Internship Helper", formTitle: "Find Your Internship", educationLabel: "Education (short)",
        educationPlaceholder: "E.g., B.Tech (CS), 3rd year", skillsLabel: "Skills (semicolon separated)",
        skillsPlaceholder: "python;excel;communication", interestsLabel: "Interests / Sector", interestsPlaceholder: "e.g., Data, Design, Outreach",
        cityLabel: "Your City (for distance)", cityPlaceholder: "e.g., New Delhi", resumeLabel: "Or upload resume (PDF)",
        uploadBtn: "Upload Resume", recoBtn: "Get Recommendations", recoTitle: "Recommendations", footerText: "Designed for ease of use and accessibility.",
        chatbotHeader: "Internship Assistant", chatbotPlaceholder: "Ask me anything...", chatbotWelcome: "Hello! How can I help you find an internship today?",
        uploadSuccess: "Resume uploaded successfully.", uploadFail: "Failed to upload resume.", noResume: "Please select a resume file first.",
        noMatches: "No matches found. Try broadening your skills or interests.", failReco: "Failed to get recommendations.",
        orgLabel: "Organization:", locationLabel: "Location:", kmAway: "km away", stipendLabel: "Stipend:", accomLabel: "Accom. est:",
        skillsMatchedLabel: "Matched Skills:", sectorRelevance: "Sector relevance", applyBtn: "Apply / Details", applyAlert: "To apply, search for Internship ID:"
    },
    hi: {
        headerTitle: "इंटर्नशिप हेल्पर", formTitle: "अपनी इंटर्नशिप खोजें", educationLabel: "शिक्षा (संक्षिप्त)",
        educationPlaceholder: "उदा., बी.टेक (सीएस), तीसरा वर्ष", skillsLabel: "कौशल (अर्धविराम से अलग)",
        skillsPlaceholder: "पायथन;एक्सेल;कम्युनिकेशन", interestsLabel: "रुचियाँ / क्षेत्र", interestsPlaceholder: "उदा., डेटा, डिज़ाइन, आउटरीच",
        cityLabel: "आपका शहर (दूरी के लिए)", cityPlaceholder: "उदा., नई दिल्ली", resumeLabel: "या फिर रिज्यूमे अपलोड करें (PDF)",
        uploadBtn: "रिज्यूमे अपलोड करें", recoBtn: "सिफारिशें प्राप्त करें", recoTitle: "सिफारिशें", footerText: "उपयोग में आसानी और पहुंच के लिए डिज़ाइन किया गया।",
        chatbotHeader: "इंटर्नशिप असिस्टेंट", chatbotPlaceholder: "कुछ भी पूछें...", chatbotWelcome: "नमस्ते! आज मैं आपकी इंटर्नशिप ढूंढने में कैसे मदद कर सकता हूँ?",
        uploadSuccess: "रिज्यूमे सफलतापूर्वक अपलोड हुआ।", uploadFail: "रिज्यूमे अपलोड करने में विफल।", noResume: "कृपया पहले एक रिज्यूमे फ़ाइल चुनें।",
        noMatches: "कोई मेल नहीं मिला। अपने कौशल या रुचियों को व्यापक बनाने का प्रयास करें।", failReco: "सिफारिशें प्राप्त करने में विफल।",
        orgLabel: "संगठन:", locationLabel: "स्थान:", kmAway: "किमी दूर", stipendLabel: "स्टाइपेंड:", accomLabel: "आवास अनुमान:",
        skillsMatchedLabel: "मेल खाने वाले कौशल:", sectorRelevance: "क्षेत्र प्रासंगिकता", applyBtn: "आवेदन / विवरण", applyAlert: "आवेदन करने के लिए, इंटर्नशिप आईडी खोजें:"
    },
    ta: {
        headerTitle: "பயிற்சி உதவியாளர்", formTitle: "உங்கள் பயிற்சியைக் கண்டறியவும்", educationLabel: "கல்வி (சுருக்கமானது)",
        educationPlaceholder: "எ.கா., பி.டெக் (சிஎஸ்), 3ம் ஆண்டு", skillsLabel: "திறன்கள் (நிறுத்தற்குறியுடன் பிரிக்கப்பட்டது)",
        skillsPlaceholder: "பைதான்;எக்செல்;தகவல்தொடர்பு", interestsLabel: "ஆர்வங்கள் / துறை", interestsPlaceholder: "எ.கா., தரவு, வடிவமைப்பு, அவுட்ரீச்",
        cityLabel: "உங்கள் நகரம் (தூரத்திற்கு)", cityPlaceholder: "எ.கா., புது தில்லி", resumeLabel: "அல்லது ரெஸ்யூமே பதிவேற்றவும் (PDF)",
        uploadBtn: "ரெஸ்யூமே பதிவேற்றவும்", recoBtn: "பரிந்துரைகளைப் பெறுங்கள்", recoTitle: "பரிந்துரைகள்", footerText: "பயன்பாட்டின் எளிமை மற்றும் அணுகலுக்காக வடிவமைக்கப்பட்டுள்ளது.",
        chatbotHeader: "பயிற்சி உதவியாளர்", chatbotPlaceholder: "என்னிடம் எதுவும் கேளுங்கள்...", chatbotWelcome: "வணக்கம்! இன்று ஒரு பயிற்சி கண்டுபிடிக்க நான் உங்களுக்கு எப்படி உதவ முடியும்?",
        uploadSuccess: "ரெஸ்யூமே வெற்றிகரமாக பதிவேற்றப்பட்டது.", uploadFail: "ரெஸ்யூமே பதிவேற்றத் தவறிவிட்டது.", noResume: "முதலில் ஒரு ரெஸ்யூமே கோப்பைத் தேர்ந்தெடுக்கவும்.",
        noMatches: "பொருத்தங்கள் எதுவும் இல்லை. உங்கள் திறமைகளையோ ஆர்வங்களையோ விரிவுபடுத்த முயற்சிக்கவும்.", failReco: "பரிந்துரைகளைப் பெறத் தவறிவிட்டது.",
        orgLabel: "அமைப்பு:", locationLabel: "இடம்:", kmAway: "கிமீ தொலைவில்", stipendLabel: "உதவித்தொகை:", accomLabel: "தங்குமிட மதிபீடு:",
        skillsMatchedLabel: "பொருந்தும் திறன்கள்:", sectorRelevance: "துறை பொருத்தம்", applyBtn: "விண்ணப்பிக்க / விவரங்கள்", applyAlert: "விண்ணப்பிக்க, பயிற்சி ஐடியைத் தேடவும்:"
    }
};

let currentLang = 'en';
let latestResumeText = '';

function updateUI(lang) {
    currentLang = lang;
    document.querySelectorAll('[data-key]').forEach(el => {
        const key = el.getAttribute('data-key');
        if (strings[lang][key]) el.innerText = strings[lang][key];
    });
    document.querySelectorAll('[data-key-placeholder]').forEach(el => {
        const key = el.getAttribute('data-key-placeholder');
        if (strings[lang][key]) el.placeholder = strings[lang][key];
    });
}

function addChatEntry(role, text) { 
    const log = document.getElementById('chat-log'); 
    const node = document.createElement('div'); 
    node.className = 'chat-' + role; 
    node.innerText = text; 
    log.appendChild(node); 
    log.scrollTop = log.scrollHeight; 
}

// Add this function to create a typing animation
function typeMessage(element, text, speed = 30) {
    return new Promise(resolve => {
        let i = 0;
        element.innerHTML = '';
        const timer = setInterval(() => {
            if (i < text.length) {
                element.innerHTML += text.charAt(i);
                i++;
            } else {
                clearInterval(timer);
                resolve();
            }
        }, speed);
    });
}

// Modified addChatEntry function to support typing animation for bot messages
async function addChatEntry(role, text) { 
    const log = document.getElementById('chat-log'); 
    const node = document.createElement('div'); 
    node.className = 'chat-' + role;
    
    if (role === 'bot') {
        // Create a temporary element for typing animation
        const tempNode = document.createElement('div');
        tempNode.className = 'chat-bot typing';
        tempNode.innerHTML = '<span class="typing-dots"><span>.</span><span>.</span><span>.</span></span>';
        log.appendChild(tempNode);
        log.scrollTop = log.scrollHeight;
        
        // Remove the typing indicator and add the actual message with animation
        await new Promise(resolve => setTimeout(resolve, 500)); // Short delay before typing starts
        log.removeChild(tempNode);
        log.appendChild(node);
        await typeMessage(node, text);
    } else {
        node.innerText = text; 
        log.appendChild(node);
    }
    
    log.scrollTop = log.scrollHeight;
}

const chatSessionId = 'session_' + Date.now();

// City mapping for distance calculation
const cityMap = { 
    "new delhi": [28.6139, 77.2090], 
    "bengaluru": [12.9716, 77.5946], 
    "patna": [25.5941, 85.1376], 
    "raipur": [21.2514, 81.6296], 
    "shillong": [25.5788, 91.8933] 
};

// Get Recommendations button handler
document.getElementById('get-reco').addEventListener('click', async () => { 
    // Show loading animation
    const resultsEl = document.getElementById('results');
    resultsEl.innerHTML = `
        <div class="loading-animation">
            <div class="spinner"></div>
            <p>Searching for the best internships...</p>
        </div>
    `;
    
    const education = document.getElementById('education').value; 
    const skills = document.getElementById('skills').value; 
    const interests = document.getElementById('interests').value; 
    const city = document.getElementById('city').value.trim().toLowerCase(); 
    
    let lat = null, lon = null; 
    if (cityMap[city]) { 
        [lat, lon] = cityMap[city] 
    } 
    
    const payload = { 
        education, 
        skills, 
        interests, 
        resume_text: latestResumeText, 
        lat, 
        lon 
    };
    
    try {
        const res = await fetch('/recommend', { 
            method: 'POST', 
            headers: { 'Content-Type': 'application/json' }, 
            body: JSON.stringify(payload) 
        });
        
        const j = await res.json();
        resultsEl.innerHTML = '';
        
        if (j.success) {
            if (j.recommendations.length === 0) {
                resultsEl.innerHTML = `<p>${strings[currentLang].noMatches}</p>`;
                return;
            }
            // Update the results display section in your JavaScript
            // After displaying the initial results
            j.recommendations.forEach(r => {
               const card = document.createElement('div');
               card.className = 'card';
               if (r.ai_processing) {
                   card.classList.add('loading');
               }
    
               card.innerHTML = `<h3>${r.title}</h3>
                   <p><strong>${strings[currentLang].orgLabel}</strong> ${r.org}</p>
                   <p><strong>${strings[currentLang].locationLabel}</strong> ${r.location} ${r.distance_km ? '(' + r.distance_km + ' ' + strings[currentLang].kmAway + ')' : ''}</p>
                   <p><strong>${strings[currentLang].stipendLabel}</strong> Rs. ${r.stipend} | <strong>${strings[currentLang].accomLabel}</strong> Rs. ${r.accommodation}</p>
                   ${r.matched_skills.length > 0 ? `<p><strong>${strings[currentLang].skillsMatchedLabel}</strong> ${r.matched_skills.join(', ')}</p>` : ''}
                   <div class="explanation">
                       <h4>Why this internship is recommended for you:</h4>
                       <p>${r.explanation}</p>
                   </div>
                   <div class="card-actions">
                       <a href="${r.link}" target="_blank" rel="noopener noreferrer" class="apply-btn">${strings[currentLang].applyBtn}</a>
                   </div>`;
               resultsEl.appendChild(card);
           });

           const loadingCards = document.querySelectorAll('.card.loading');
           if (loadingCards.length > 0) {
               // In a real implementation, you might poll the server or use WebSockets
               // to get updated explanations. For now, we'll just simulate a delay
               setTimeout(() => {
                   loadingCards.forEach(card => {
                       card.classList.remove('loading');
                       const explanationEl = card.querySelector('.explanation p');
                       if (explanationEl) {
                           explanationEl.textContent = "This internship offers valuable experience that aligns with your skills and career goals. It provides an opportunity to apply your knowledge in a real-world setting and build professional connections.";
                       }
                   });
               }, 3000);
           }
        } else {
            resultsEl.innerText = strings[currentLang].failReco;
        }
    } catch (error) {
        console.error('Error getting recommendations:', error);
        document.getElementById('results').innerText = strings[currentLang].failReco;
    }
});

// Update the chat-send event listener to use the new animation
document.getElementById('chat-send').addEventListener('click', async () => {
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;
    
    addChatEntry('user', msg);
    input.value = '';
    
    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg, session_id: chatSessionId })
        });
        const j = await res.json();
        await addChatEntry('bot', j.reply);
    } catch (error) {
        console.error('Chat error:', error);
        await addChatEntry('bot', "Sorry, I'm having trouble connecting right now.");
    }
});

document.getElementById('chat-input').addEventListener('keypress', (e) => { 
    if (e.key === 'Enter') document.getElementById('chat-send').click() 
});

// Resume upload functionality
document.getElementById('upload-resume').addEventListener('click', async () => { 
    const fileInput = document.getElementById('resume'); 
    const f = fileInput.files[0]; 
    if (!f) { 
        alert(strings[currentLang].noResume); 
        return; 
    } 
    
    const form = new FormData(); 
    form.append('resume', f); 
    
    try {
        const res = await fetch('/upload_resume', { method: 'POST', body: form });
        const j = await res.json();
        if (j.success) {
            latestResumeText = j.text;
            alert(strings[currentLang].uploadSuccess);
        } else {
            alert(strings[currentLang].uploadFail);
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert(strings[currentLang].uploadFail);
    }
});

// Chatbot panel functionality
const chatbotBtn = document.getElementById('chatbot-button'); 
const chatbotPanel = document.getElementById('chatbot-panel'); 
const chatbotClose = document.getElementById('chatbot-close'); 

chatbotBtn.addEventListener('click', () => { 
    chatbotPanel.classList.add('open') 
}); 

chatbotClose.addEventListener('click', () => { 
    chatbotPanel.classList.remove('open') 
});

// Language switcher
document.getElementById('lang').addEventListener('change', (e) => updateUI(e.target.value));

// Initialize UI with default language
document.addEventListener('DOMContentLoaded', () => updateUI('en'));

</script>
</body>
</html>""")

# Write the CSS file for styling
with open('static/style.css', 'w', encoding='utf-8') as f:
    f.write("""
:root{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;--primary-color:#0052cc;--light-gray:#f4f5f7;--dark-gray:#333;--white:#fff;--shadow:0 4px 8px rgba(0,0,0,0.1);}
body{margin:0;background:var(--light-gray);color:var(--dark-gray);}

/* START: New Hero Styles */
.hero {
    background-color: var(--primary-color);
    color: var(--white);
    text-align: center;
    padding: 5rem 2rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    min-height: 40vh; /* Adjust height as needed */
}
.hero-title {
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 1rem;
}
.hero-intro {
    font-size: 1.1rem;
    max-width: 600px;
    line-height: 1.6;
    opacity: 0.9;
}
/* END: New Hero Styles */
/* Loading animation styles */
.loading-animation {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 2rem;
}

.spinner {
    width: 50px;
    height: 50px;
    border: 5px solid #f3f3f3;
    border-top: 5px solid var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-bottom: 1rem;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-animation p {
    color: var(--primary-color);
    font-weight: 600;
}

/* Add these styles to your existing CSS */
.explanation {
    background-color: #f8f9fa;
    border-left: 4px solid var(--primary-color);
    padding: 0.75rem 1rem;
    margin: 1rem 0;
    border-radius: 0 4px 4px 0;
}

.explanation h4 {
    margin: 0 0 0.5rem 0;
    color: var(--primary-color);
    font-size: 0.95rem;
}

.explanation p {
    margin: 0;
    font-size: 0.9rem;
    line-height: 1.5;
    color: #495057;
    font-stlye: italic;
}

/* Enhanced card styling */
.card {
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    position: relative;
    overflow: hidden;
}

.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.1);
}

.card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(to bottom, var(--primary-color), #6c757d);
    opacity: 0.7;
}

.card.loading .explanation {
    background-color: #e9ecef;
}

.card.loading .explanation p::after {
    content: '...';
    animation: loadingDots 1.5s infinite;
}

@keyframes loadingDots {
    0%, 20% { content: '.'; }
    40% { content: '..'; }
    60%, 100% { content: '...'; }
}

.card.loading {
    animation: pulseLoading 2s infinite;
}

@keyframes pulseLoading {
    0% { opacity: 1; }
    50% { opacity: 0.8; }
    100% { opacity: 1; }
}

.container{max-width:1200px;margin:2rem auto;padding:1rem;display:flex;flex-direction:column;gap:1.5rem;}
header{display:flex;justify-content:space-between;align-items:center;}
header h1{font-size:2rem;color:var(--primary-color);margin:0;}
.lang-switch select{padding:8px;border-radius:6px;border:1px solid #ccc;background:var(--white);}
main{display:grid;grid-template-columns:1fr;gap:1.5rem;}
@media(min-width:900px){main{grid-template-columns:1fr 2fr;}}
.form-container, .results-container{background:var(--white);padding:1.5rem;border-radius:12px;box-shadow:var(--shadow);}
.profile-form label{display:block;font-weight:600;margin-bottom:0.5rem;}
.profile-form input{width:calc(100% - 20px);padding:10px;margin-bottom:1rem;border-radius:6px;border:1px solid #ccc;}
.actions{display:flex;gap:1rem;}.actions button{flex:1;padding:12px;border-radius:8px;border:none;color:var(--white);cursor:pointer;font-weight:600;}
#upload-resume{background:#6c757d;} #get-reco{background:var(--primary-color);}
.results .card{border:1px solid #eee;padding:1rem;margin-bottom:1rem;border-radius:8px;background:#fafafa;}
.card h3{margin:0 0 0.5rem 0;color:var(--primary-color);} .card p{margin:0.25rem 0;line-height:1.5;}
/* In STEP 3, add this rule to your static/style.css file */
.card-actions .apply-btn {
    display: inline-block; /* Allows padding and other box model properties */
    background: #28a745;
    border: none;
    color: var(--white);
    padding: 10px 15px;
    border-radius: 6px;
    cursor: pointer;
    margin-top: 0.5rem;
    text-decoration: none; /* Removes the underline from the link */
    font-weight: normal;
}
footer{text-align:center;color:#666;font-size:0.9rem;margin-top:2rem;}
/* Chatbot typing animation */
.typing-dots {
    display: inline-flex;
}

.typing-dots span {
    animation: typing 1.4s infinite;
    margin: 0 1px;
}

.typing-dots span:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-dots span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes typing {
    0%, 60%, 100% {
        transform: translateY(0);
        opacity: 0.4;
    }
    30% {
        transform: translateY(-5px);
        opacity: 1;
    }
}

/* Chatbot open/close animation */
#chatbot-panel {
    transition: transform 0.3s ease-in-out;
    transform: translateX(100%);
}

#chatbot-panel.open {
    transform: translateX(0);
}

#chatbot-button {
    transition: transform 0.2s ease;
}

#chatbot-button:hover {
    transform: scale(1.1);
}

/* Message animation */
.chat-bot, .chat-user {
    animation: messageAppear 0.3s ease;
    overflow: hidden;
}

@keyframes messageAppear {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
#chatbot-button{position:fixed;bottom:20px;right:20px;background:var(--primary-color);color:var(--white);border-radius:50%;width:60px;height:60px;font-size:28px;display:flex;align-items:center;justify-content:center;cursor:pointer;box-shadow:var(--shadow);z-index:1000;}
#chatbot-panel{position:fixed;top:0;right:-400px;width:350px;height:100vh;background:var(--white);border-left:1px solid #ddd;box-shadow:-2px 0 10px rgba(0,0,0,0.1);display:flex;flex-direction:column;transition:right .3s ease;z-index:1001;}
#chatbot-panel.open{right:0;}
#chatbot-header{display:flex;justify-content:space-between;align-items:center;padding:1rem;background:var(--primary-color);color:var(--white);font-weight:700;}
#chatbot-close{background:0 0;border:none;color:var(--white);font-size:24px;cursor:pointer;}
#chatbot-body{display:flex;flex-direction:column;flex:1;padding:1rem;overflow:hidden;}
.chat-log{flex:1;overflow-y:auto;margin-bottom:1rem;padding:0.5rem;}
.chat-user{background:#e9f2ff;padding:10px;border-radius:12px 12px 0 12px;margin:0.5rem 0;text-align:right;margin-left:auto;max-width:80%;}
.chat-bot{background:var(--light-gray);padding:10px;border-radius:12px 12px 12px 0;margin:0.5rem 0;text-align:left;max-width:80%;}
.chat-input{display:flex;gap:0.5rem;}
.chat-input input{flex:1;padding:10px;border:1px solid #ddd;border-radius:20px;}
.chat-input button{background:var(--primary-color);color:var(--white);border:none;border-radius:20px;padding:10px 15px;cursor:pointer;}
""")
print("✅ Project files created successfully.")

# STEP 4: DEFINE THE FLASK APPLICATION AND API LOGIC
print("\n⏳ Configuring Flask application and APIs...")

# Configure the Gemini API
try:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    print("✅ Gemini API configured successfully.")
except Exception as e:
    print(f"❌ Error configuring Gemini API: {e}")
    model = None

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

# Load and process internship data once at startup
interns = pd.read_csv('internships.csv', dtype=str)
for col in ['lat', 'lon', 'accommodation_estimate']:
    interns[col] = pd.to_numeric(interns[col], errors='coerce')

vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
tfidf_matrix = vectorizer.fit_transform(interns['description'].fillna(''))

def haversine(lat1, lon1, lat2, lon2):
    if any(pd.isna(val) for val in [lat1, lon1, lat2, lon2]): return 9999.0
    R = 6371.0
    phi1, phi2, dphi, dlambda = map(math.radians, [lat1, lat2, lat2 - lat1, lon2 - lon1])
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_resume', methods=['POST'])
def upload_resume():
    if 'resume' not in request.files: return jsonify({'success': False, 'error': 'No file part'})
    file = request.files['resume']
    if file.filename == '': return jsonify({'success': False, 'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        try:
            text = extract_text(path)
        except Exception:
            text = ""
        finally:
            os.remove(path)
        return jsonify({'success': True, 'text': text[:20000]})
    return jsonify({'success': False, 'error': 'File upload failed'})

# In STEP 4, replace the @app.route('/recommend') function with this:
@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json or {}
    try:
        user_lat = float(data.get('lat')) if data.get('lat') else None
        user_lon = float(data.get('lon')) if data.get('lon') else None
    except (ValueError, TypeError):
        user_lat, user_lon = None, None

    candidate_text = " ".join(filter(None, [data.get('education'), data.get('interests'), data.get('resume_text')]))
    cand_vec = vectorizer.transform([candidate_text])
    sims = cosine_similarity(cand_vec, tfidf_matrix).flatten()

    skills = {s.strip().lower() for s in data.get('skills', '').split(';') if s.strip()}
    interests = data.get('interests', '') or ''
    education = data.get('education', '') or ''

    results = []
    for idx, row in interns.iterrows():
        req_skills = {s.strip().lower() for s in str(row.get('required_skills', '')).split(';') if s.strip()}
        matched_count = len(skills.intersection(req_skills))
        skill_score = matched_count / len(req_skills) if req_skills else 0.0

        dist_km = haversine(user_lat, user_lon, row['lat'], row['lon'])
        distance_score = max(0, 1 - (dist_km / 2000))  # Normalize distance score

        # Check if interests align
        sector = str(row.get('sector', '')).lower()
        interest_score = 0.5 if interests and sector and any(word in interests.lower() for word in sector.split()) else 0.0

        final_score = (0.5 * skill_score) + (0.3 * sims[idx]) + (0.15 * distance_score) + (0.05 * interest_score)

        # Create a basic explanation as fallback
        fallback_explanation = f"This internship at {row['organization']} aligns with your profile. "
        if matched_count > 0:
            fallback_explanation += f"Your skills in {', '.join(list(skills.intersection(req_skills))[:3])} match the requirements. "
        if interests and sector and any(word in interests.lower() for word in sector.split()):
            fallback_explanation += f"It also relates to your interest in {sector}. "
        fallback_explanation += "This experience will help you develop professionally and build your resume."

        # Generate AI-powered explanation if available, otherwise use fallback
        if model:
            explanation = "Generating personalized explanation..."
            ai_processing = True
            # We'll generate the AI explanation asynchronously or in a separate step
        else:
            explanation = fallback_explanation
            ai_processing = False

        results.append({
            'id': row['id'], 'title': row['title'], 'organization': row['organization'], 'city': row['city'],
            'state': row['state'], 'distance_km': round(dist_km) if dist_km < 9000 else None,
            'stipend': row.get('stipend', 'N/A'), 'accommodation': int(row.get('accommodation_estimate', 2500)),
            'matched_skills': sorted(list(skills.intersection(req_skills))), 'score': final_score,
            'link': row['link'], 'explanation': explanation,
            'sector': sector,
            'ai_processing': ai_processing,
            'fallback_explanation': fallback_explanation
        })

    ranked = sorted(results, key=lambda x: x['score'], reverse=True)
    top_recommendations = ranked[:5]
    
    # Generate AI explanations for the top recommendations if model is available
    if model:
        for rec in top_recommendations:
            if rec['ai_processing']:
                try:
                    row = interns[interns['id'] == rec['id']].iloc[0]
                    ai_explanation = generate_ai_explanation(row, skills, interests, education)
                    rec['explanation'] = ai_explanation
                    rec['ai_processing'] = False
                except Exception as e:
                    print(f"Error generating AI explanation: {e}")
                    rec['explanation'] = rec['fallback_explanation']
                    rec['ai_processing'] = False
    
    output = [{**r, 'org': r['organization'], 'location': f"{r['city']}, {r['state']}"} for r in top_recommendations]
    return jsonify({'success': True, 'recommendations': output})


def generate_ai_explanation(row, user_skills, user_interests, user_education):
    """Generate AI-powered explanation using Gemini"""
    # Prepare the prompt for the AI
    prompt = f"""
    Create a concise explanation (2-3 sentences) for why this internship would be a good fit for a student.
    
    Internship: {row['title']} at {row['organization']}
    Sector: {row.get('sector', 'N/A')}
    Description: {row.get('description', 'N/A')}
    Required Skills: {row.get('required_skills', 'N/A')}
    
    Student Profile:
    - Skills: {', '.join(user_skills) if user_skills else 'Not specified'}
    - Interests: {user_interests if user_interests else 'Not specified'}
    - Education: {user_education if user_education else 'Not specified'}
    
    Focus on:
    1. Why this matches the student's profile
    2. How it could benefit their career development
    3. What they might gain from this experience
    
    Keep the response friendly and encouraging. Write in a natural, conversational tone.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "This internship aligns well with your skills and interests, providing valuable professional experience."
    except Exception as e:
        print(f"Error generating AI explanation: {e}")
        return "This internship offers great learning opportunities that match your profile and career aspirations."

# A dictionary to hold chat histories for different users (if needed, here we use one)
chat_sessions = {}

@app.route('/chat', methods=['POST'])
def chat():
    if model is None:
        return jsonify({'reply': "Chatbot is not available due to an API configuration error."})

    data = request.json or {}
    msg = data.get('message', '')
    session_id = data.get('session_id', 'default') # Use a session ID

    if not msg:
        return jsonify({'reply': "Please send a message."})

    # Get or create a chat session for the user
    if session_id not in chat_sessions:
        # The system instruction is now part of the history, not the prompt
        system_instruction = textwrap.dedent("""You are a friendly and helpful chatbot for an internship website.
        Your goal is to guide users on how to find the best internship using the site's features.
        If asked for recommendations, gently guide them to fill out the form or upload their resume and click 'Get Recommendations'.
        Keep your answers concise and conversational (2-3 sentences).""")
        chat_sessions[session_id] = model.start_chat(history=[
            {'role': 'user', 'parts': ["Hello!"]},
            {'role': 'model', 'parts': [system_instruction]}
        ])

    chat_session = chat_sessions[session_id]

    try:
        response = chat_session.send_message(msg)
        reply = response.text
    except Exception as e:
        print(f"❌ Gemini Error: {e}")
        reply = "Sorry, I'm having trouble connecting right now. Please try again later."
    return jsonify({'reply': reply})

print("✅ Flask application configured.")

# STEP 5: START THE NGROK TUNNEL AND RUN THE FLASK APP
if __name__ == '__main__':
    try:
        # Add this line to disconnect any existing tunnels
        ngrok.kill()

        ngrok.set_auth_token(os.environ.get("NGROK_AUTHTOKEN"))
        port = 5000
        public_url = ngrok.connect(port)
        print("\n" + "="*40)
        print("🚀 Your application is live! 🚀")
        print(f"🔗 Public URL: {public_url}")
        print("You can access your web app from any browser using this URL.")
        print("="*40 + "\n")
        app.run(port=port)
    except Exception as e:
        print(f"❌ Failed to start ngrok or Flask app: {e}")
        print("Please check your ngrok auth token and try again.")

    app.run(port=port, debug=False)