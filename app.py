from flask import Flask, jsonify, request
from flask_cors import CORS
import jwt
import bcrypt
from datetime import datetime, timedelta
from pymongo import MongoClient
from bson import ObjectId
import os
import requests as http_requests
from statistics import mean
import csv
import threading

from dotenv import load_dotenv
load_dotenv()

from chat_history_endpoints import chat_history_bp, init_app as init_chat_history
from pypdf import PdfReader

# ── Load CSV historical groundwater data at startup ────────────────────────────
def load_csv_data():
    """Load data/sih chatbot data.csv into a dict keyed by lowercase district name."""
    data = {}
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'sih chatbot data.csv')
    if not os.path.exists(csv_path):
        print('⚠️  CSV data file not found:', csv_path)
        return data
    try:
        with open(csv_path, newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                district = row.get('District', '').strip()
                if district:
                    data[district.lower()] = row
        print(f'✅ Loaded CSV historical data: {len(data)} districts')
    except Exception as e:
        print(f'❌ CSV load error: {e}')
    return data

CSV_GW_DATA = load_csv_data()

# ── Load 2019-2021 depth-to-water-level data ──────────────────────────────────
def load_depth_data():
    """Load RS_Session_257_AU_896_3.csv keyed by (lowercase district, year)."""
    data = {}
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'RS_Session_257_AU_896_3.csv')
    if not os.path.exists(csv_path):
        print('⚠️  Depth CSV not found:', csv_path)
        return data
    try:
        with open(csv_path, newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                district = row.get('District', '').strip()
                year = row.get('Year', '').strip()
                if district.lower() == 'total' or not year.isdigit():
                    continue
                data[(district.lower(), year)] = row
        print(f'✅ Loaded depth data: {len(data)} records')
    except Exception as e:
        print(f'❌ Depth CSV load error: {e}')
    return data

DEPTH_GW_DATA = load_depth_data()

# ── Load district categorization CSV at startup ──────────────────────────────
def load_categorization_data():
    """Load tn_district_gw_categorization.csv keyed by lowercase district name."""
    data = {}
    csv_path = os.path.join(os.path.dirname(__file__), 'data', 'tn_district_gw_categorization.csv')
    if not os.path.exists(csv_path):
        print('⚠️  Categorization CSV not found:', csv_path)
        return data
    try:
        with open(csv_path, newline='', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                district = row.get('District', '').strip()
                if district:
                    data[district.lower()] = row
        print(f'✅ Loaded categorization data: {len(data)} districts')
    except Exception as e:
        print(f'❌ Categorization CSV load error: {e}')
    return data

CAT_GW_DATA = load_categorization_data()

# ── Load PDF chunks at startup ───────────────────────────────────────
def load_pdf_chunks():
    """Extract text from all PDFs in data/ folder, one chunk per page."""
    chunks = []
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith('.pdf'):
            continue
        try:
            reader = PdfReader(os.path.join(data_dir, fname))
            for i, page in enumerate(reader.pages):
                text = (page.extract_text() or '').strip()
                if text:
                    # encode/decode to strip non-ASCII chars that cause issues
                    text = text.encode('ascii', errors='ignore').decode('ascii')
                    chunks.append({'source': fname, 'page': i + 1, 'text': text})
            print(f'✅ Loaded PDF: {fname} ({len(reader.pages)} pages)')
        except Exception as e:
            print(f'❌ PDF load error {fname}: {e}')
    print(f'✅ Total PDF chunks: {len(chunks)}')
    return chunks

def find_relevant_chunks(query, chunks, top_k=4):
    """Return top_k most relevant chunks using TF-IDF cosine similarity."""
    if not chunks:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    texts = [c['text'] for c in chunks]
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf = vectorizer.fit_transform(texts + [query])
        scores = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
        top_indices = scores.argsort()[-top_k:][::-1]
        return [chunks[i] for i in top_indices if scores[i] > 0.01]
    except Exception as e:
        print(f'TF-IDF error: {e}')
        return []

PDF_CHUNKS = load_pdf_chunks()

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.register_blueprint(chat_history_bp)

# ── Constants ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
GEMINI_URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent'
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions'
WRIS_BASE = 'https://indiawris.gov.in/Dataset/Ground Water Level'
WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
WEATHER_URL = 'http://api.weatherapi.com/v1'

# ── MongoDB ────────────────────────────────────────────────────────────────────
try:
    client = MongoClient(os.getenv('MONGO_URI'), serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client['puvijal']
    users_col = db['users']
    chats_col = db['chat_history']
    weather_log_col = db['weather_log']
    init_chat_history(db, app.config['SECRET_KEY'])
    print('✅ Connected to MongoDB Atlas!')
except Exception as e:
    print(f'❌ MongoDB connection failed: {e}')

# ── Helper Functions ───────────────────────────────────────────────────────────
def get_email_from_token():
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload.get('email')
    except:
        return None

def call_gemini(prompt):
    try:
        res = http_requests.post(
            GEMINI_URL,
            headers={'Content-Type': 'application/json', 'X-goog-api-key': GEMINI_API_KEY},
            json={'contents': [{'parts': [{'text': prompt}]}]},
            timeout=30
        )
        if res.status_code == 200:
            return res.json()['candidates'][0]['content']['parts'][0]['text']
        print(f'Gemini HTTP {res.status_code}: {res.text[:200]}')
    except Exception as e:
        print(f'Gemini error: {e}')
    return None

def call_llm(prompt):
    """Call Groq Llama 3.3 70B. Falls back to Gemini if Groq fails."""
    if GROQ_API_KEY:
        try:
            res = http_requests.post(
                GROQ_URL,
                headers={'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'},
                json={
                    'model': 'llama-3.3-70b-versatile',
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': 0.7,
                },
                timeout=30
            )
            if res.status_code == 200:
                return res.json()['choices'][0]['message']['content']
            print(f'Groq HTTP {res.status_code}: {res.text[:200]}')
        except Exception as e:
            print(f'Groq error: {e}')
    print('Falling back to Gemini...')
    return call_gemini(prompt)

def fetch_live_gw(location, start=None, end=None, size=20):
    today = datetime.utcnow()
    end = end or today.strftime('%Y-%m-%d')
    start = start or (today.replace(year=today.year - 1)).strftime('%Y-%m-%d')
    try:
        session = http_requests.Session()
        req = http_requests.Request(
            'POST', WRIS_BASE,
            params={
                'stateName': 'Tamil Nadu', 'districtName': location,
                'agencyName': 'CGWB', 'startdate': start.strip(), 'enddate': end.strip(),
                'download': 'true', 'page': 0, 'size': size
            },
            headers={
                'accept': 'application/json', 'User-Agent': 'Mozilla/5.0',
                'Origin': 'https://indiawris.gov.in', 'Referer': 'https://indiawris.gov.in/'
            }
        )
        res = session.send(req.prepare(), timeout=15)
        print(f'WRIS GW STATUS: {res.status_code}')
        print(f'WRIS GW BODY: {res.text[:500]}')
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(f'WRIS GW API error: {e}')
    return None

def fetch_weather(location, state='Tamil Nadu'):
    if not WEATHER_API_KEY:
        return None
    try:
        loc = f"{location},{state},India"
        forecast_res = http_requests.get(
            f"{WEATHER_URL}/forecast.json",
            params={'key': WEATHER_API_KEY, 'q': loc, 'days': 7, 'aqi': 'no'},
            timeout=10
        )
        result = {}
        if forecast_res.status_code == 200:
            fd = forecast_res.json()
            current = fd['current']
            result['current'] = {
                'temp_c': current['temp_c'],
                'humidity': current['humidity'],
                'precip_mm': current['precip_mm'],
                'condition': current['condition']['text'],
                'wind_kph': current['wind_kph'],
            }
            result['forecast'] = [
                {
                    'date': day['date'],
                    'rainfall_mm': day['day']['totalprecip_mm'],
                    'humidity': day['day']['avghumidity'],
                    'condition': day['day']['condition']['text'],
                }
                for day in fd['forecast']['forecastday']
            ]
            result['total_forecast_rain_mm'] = round(sum(d['rainfall_mm'] for d in result['forecast']), 2)
        else:
            print(f'WeatherAPI forecast error {forecast_res.status_code}: {forecast_res.text[:200]}')
        return result if result else None
    except Exception as e:
        print(f'WeatherAPI error: {e}')
    return None

def log_weather(district, result):
    """Silently save weather snapshot to MongoDB for future trend analysis."""
    try:
        if not result or not result.get('current'):
            return
        weather_log_col.insert_one({
            'district': district.lower(),
            'logged_at': datetime.utcnow(),
            'date': datetime.utcnow().strftime('%Y-%m-%d'),
            'temp_c': result['current'].get('temp_c'),
            'humidity': result['current'].get('humidity'),
            'precip_mm': result['current'].get('precip_mm'),
            'condition': result['current'].get('condition'),
            'wind_kph': result['current'].get('wind_kph'),
            'forecast_7day_rain_mm': result.get('total_forecast_rain_mm'),
            'forecast': result.get('forecast', []),
        })
    except Exception as e:
        print(f'Weather log error: {e}')


# ── India states → representative districts ────────────────────────────────────
_INDIA_STATES = {
    'andhra pradesh': ['Visakhapatnam','Vijayawada','Guntur','Tirupati','Kurnool','Nellore','Rajahmundry','Kakinada','Anantapur','Kadapa'],
    'arunachal pradesh': ['Itanagar','Tawang','Ziro','Pasighat','Bomdila'],
    'assam': ['Guwahati','Dibrugarh','Jorhat','Silchar','Tezpur','Nagaon','Tinsukia','Bongaigaon'],
    'bihar': ['Patna','Gaya','Bhagalpur','Muzaffarpur','Darbhanga','Purnia','Ara','Begusarai'],
    'chhattisgarh': ['Raipur','Bilaspur','Durg','Korba','Raigarh','Jagdalpur','Ambikapur'],
    'goa': ['Panaji','Margao','Vasco da Gama','Mapusa','Ponda'],
    'gujarat': ['Ahmedabad','Surat','Vadodara','Rajkot','Bhavnagar','Jamnagar','Gandhinagar','Anand','Junagadh','Kutch'],
    'haryana': ['Gurugram','Faridabad','Panipat','Ambala','Hisar','Rohtak','Karnal','Sonipat','Yamunanagar'],
    'himachal pradesh': ['Shimla','Manali','Dharamshala','Kullu','Mandi','Solan','Hamirpur'],
    'jharkhand': ['Ranchi','Jamshedpur','Dhanbad','Bokaro','Deoghar','Hazaribagh','Giridih'],
    'karnataka': ['Bengaluru','Mysuru','Hubballi','Mangaluru','Belagavi','Kalaburagi','Ballari','Vijayapura','Shivamogga','Tumkur'],
    'kerala': ['Thiruvananthapuram','Kochi','Kozhikode','Thrissur','Kollam','Palakkad','Alappuzha','Malappuram','Kannur','Kasaragod'],
    'madhya pradesh': ['Bhopal','Indore','Gwalior','Jabalpur','Ujjain','Sagar','Satna','Rewa','Dewas'],
    'maharashtra': ['Mumbai','Pune','Nagpur','Nashik','Aurangabad','Solapur','Kolhapur','Amravati','Nanded','Thane'],
    'manipur': ['Imphal','Churachandpur','Thoubal','Bishnupur','Senapati'],
    'meghalaya': ['Shillong','Cherrapunji','Tura','Jowai','Nongpoh'],
    'mizoram': ['Aizawl','Lunglei','Champhai','Serchhip'],
    'nagaland': ['Kohima','Dimapur','Mokokchung','Wokha','Tuensang'],
    'odisha': ['Bhubaneswar','Cuttack','Rourkela','Berhampur','Sambalpur','Puri','Balasore','Koraput'],
    'punjab': ['Amritsar','Ludhiana','Jalandhar','Patiala','Bathinda','Mohali','Hoshiarpur','Gurdaspur'],
    'rajasthan': ['Jaipur','Jodhpur','Udaipur','Kota','Ajmer','Bikaner','Alwar','Bharatpur','Sikar','Churu'],
    'sikkim': ['Gangtok','Namchi','Gyalshing','Mangan'],
    'tamil nadu': ['Chennai','Coimbatore','Madurai','Tiruchirappalli','Salem','Tirunelveli','Tiruppur','Vellore','Erode','Theni','Dindigul','Kancheepuram','Cuddalore','Thanjavur','Tiruvannamalai'],
    'telangana': ['Hyderabad','Warangal','Nizamabad','Karimnagar','Khammam','Ramagundam','Mahbubnagar','Nalgonda'],
    'tripura': ['Agartala','Udaipur','Dharmanagar','Kailashahar'],
    'uttar pradesh': ['Lucknow','Kanpur','Agra','Varanasi','Prayagraj','Meerut','Noida','Ghaziabad','Bareilly','Moradabad'],
    'uttarakhand': ['Dehradun','Haridwar','Roorkee','Rishikesh','Nainital','Haldwani','Mussoorie'],
    'west bengal': ['Kolkata','Howrah','Asansol','Siliguri','Durgapur','Bardhaman','Malda','Jalpaiguri'],
    'delhi': ['New Delhi','Dwarka','Rohini','Saket','Lajpat Nagar','Janakpuri','Karol Bagh'],
    'jammu and kashmir': ['Srinagar','Jammu','Leh','Anantnag','Baramulla','Sopore','Pulwama'],
    'ladakh': ['Leh','Kargil'],
}

_WEATHER_KEYWORDS = [
    'rain','rainfall','weather','temperature','humid','flood','drought','sunny','cloud',
    'storm','cyclone','wind','forecast','heat','cold','snow','hail','thunder','climate'
]

def is_weather_query(query):
    q = query.lower()
    return any(kw in q for kw in _WEATHER_KEYWORDS)

def extract_state(query):
    q = query.lower()
    for state in _INDIA_STATES:
        if state in q:
            return state
    return None

def handle_state_weather_query(query, state):
    """Fetch weather for all districts of a state concurrently and pass to Gemini."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    districts = _INDIA_STATES[state]
    state_title = state.title()

    district_weather = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_district = {
            executor.submit(fetch_weather, d, state_title): d for d in districts
        }
        for future in as_completed(future_to_district):
            d = future_to_district[future]
            try:
                result = future.result()
                if result:
                    district_weather[d] = result
            except Exception as e:
                print(f'Weather fetch error for {d}: {e}')

    if not district_weather:
        return f'Could not fetch weather data for {state_title}. Please try again later.'

    # Build summary lines per district
    summary_lines = []
    for d, w in district_weather.items():
        c = w.get('current', {})
        summary_lines.append(
            f"{d}: {c.get('condition','N/A')}, {c.get('temp_c','N/A')}°C, "
            f"Humidity {c.get('humidity','N/A')}%, Rain today {c.get('precip_mm','N/A')} mm, "
            f"Wind {c.get('wind_kph','N/A')} kph, 7-day rain {w.get('total_forecast_rain_mm','N/A')} mm"
        )
    weather_data = '\n'.join(summary_lines)

    return call_llm(
        f"""You are Puvi Jal AI, a weather and water resource assistant for India.
Below is live weather data for districts of {state_title}.
Answer the user's question using ONLY this data. Group districts by the asked weather condition.
Use bullet points, bold district names, emojis. Answer in 8-15 lines. No filler phrases.

Live Weather Data for {state_title}:
{weather_data}

User Question: {query}

Answer:"""
    ) or f'Weather data fetched for {state_title} but AI analysis is unavailable.'

def fetch_live_rainfall(district, start=None, end=None, size=20):
    today = datetime.utcnow()
    end = end or today.strftime('%Y-%m-%d')
    start = start or (today.replace(year=today.year - 1)).strftime('%Y-%m-%d')
    try:
        res = http_requests.post(
            'https://indiawris.gov.in/Dataset/RainFall',
            params={
                'stateName': 'Tamil Nadu', 'districtName': district,
                'agencyName': 'CWC', 'startdate': start.strip(), 'enddate': end.strip(),
                'download': 'true', 'page': 0, 'size': size
            },
            headers={
                'accept': 'application/json', 'User-Agent': 'Mozilla/5.0',
                'Origin': 'https://indiawris.gov.in', 'Referer': 'https://indiawris.gov.in/'
            },
            timeout=15
        )
        print(f'WRIS RAINFALL STATUS: {res.status_code}')
        print(f'WRIS RAINFALL BODY: {res.text[:500]}')
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(f'WRIS Rainfall API error: {e}')
    return None

# ── Chat Endpoint ──────────────────────────────────────────────────────────────
@app.route('/chat/', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query', '').strip()
    answer = generate_answer(query)
    return jsonify({'answer': answer})

# Tamil Nadu districts — used for free regex-based location extraction
_TN_DISTRICTS = [
    'ariyalur','chengalpattu','chennai','coimbatore','cuddalore','dharmapuri',
    'dindigul','erode','kallakurichi','kancheepuram','kanyakumari','karur',
    'krishnagiri','madurai','mayiladuthurai','nagapattinam','namakkal','nilgiris',
    'perambalur','pudukkottai','ramanathapuram','ranipet','salem','sivaganga',
    'tenkasi','thanjavur','theni','thoothukudi','tiruchirappalli','tirunelveli',
    'tirupathur','tiruppur','tiruvallur','tiruvannamalai','tiruvarur','vellore',
    'viluppuram','virudhunagar'
]

def extract_location(query):
    """Regex match first (no Gemini call). Fall back to Gemini only if no match."""
    q_lower = query.lower()
    for district in _TN_DISTRICTS:
        if district in q_lower:
            return district.title()
    # Fallback: Gemini only for unusual spellings or city names
    result = call_llm(
        f"""Extract the district or city name in Tamil Nadu from this query.
Return ONLY the single location name. If none found, return 'none'.
Query: {query}"""
    )
    if result:
        loc = result.strip().strip('.').strip()
        if loc.lower() not in ('none', '', 'tamil nadu'):
            return loc
    return None

def build_gemini_prompt(query, weather, live_gw, live_rainfall, location):
    """Embed ALL live data + CSV historical data into one prompt — single Gemini call."""
    lines = [f'Location: {location}, Tamil Nadu']

    if weather and weather.get('current'):
        c = weather['current']
        lines.append(
            f"Current Weather: {c.get('condition')}, {c.get('temp_c')}degC, "
            f"Humidity {c.get('humidity')}%, Today Rain: {c.get('precip_mm')} mm, Wind: {c.get('wind_kph')} kph"
        )
        if weather.get('forecast'):
            for d in weather['forecast']:
                lines.append(f"  Forecast {d['date']}: {d['rainfall_mm']} mm rain, {d['condition']}")
            lines.append(f"  7-Day Total Forecast Rain: {weather.get('total_forecast_rain_mm', 0)} mm")
    else:
        lines.append('Weather data: unavailable')

    if live_gw:
        levels = [r['dataValue'] for r in live_gw if r.get('dataValue') is not None]
        if levels:
            latest = sorted(live_gw, key=lambda r: (
                r['dataTime']['year'], r['dataTime']['monthValue'], r['dataTime']['dayOfMonth']
            ), reverse=True)[0]
            lines.append(
                f"Live Groundwater (WRIS/CGWB): Station={latest['stationName']}, "
                f"WaterLevel={latest['dataValue']} m, "
                f"Date={latest['dataTime']['dayOfMonth']} {latest['dataTime']['month'].title()} {latest['dataTime']['year']}, "
                f"AvgLevel={round(mean(levels), 2)} m, WellDepth={latest.get('wellDepth', 'N/A')} m"
            )
    else:
        lines.append('Live groundwater (WRIS): unavailable for this location')

    if live_rainfall and isinstance(live_rainfall, list) and len(live_rainfall) > 0:
        rf_vals = [r['dataValue'] for r in live_rainfall if r.get('dataValue') is not None]
        if rf_vals:
            latest_rf = sorted(live_rainfall, key=lambda r: (
                r['dataTime']['year'], r['dataTime']['monthValue'], r['dataTime']['dayOfMonth']
            ), reverse=True)[0]
            lines.append(
                f"Live Rainfall (WRIS/CWC): Station={latest_rf['stationName']}, "
                f"Rainfall={latest_rf['dataValue']} mm, "
                f"Date={latest_rf['dataTime']['dayOfMonth']} {latest_rf['dataTime']['month'].title()} {latest_rf['dataTime']['year']}, "
                f"AvgRainfall={round(mean(rf_vals), 2)} mm"
            )
    else:
        lines.append('Live rainfall (WRIS): unavailable for this location')

    # ── Inject CSV historical groundwater data for this district ──
    csv_row = CSV_GW_DATA.get(location.lower())
    if csv_row:
        yearly_avg = csv_row.get('Yearly Average Ground Water Level (in Meters)', 'N/A')
        monthly_parts = []
        for k, v in csv_row.items():
            if 'Average Ground Water Level in' in k and v.strip():
                inner = k.replace('Average Ground Water Level in', '').replace('(in Meters)', '').strip()
                monthly_parts.append(f"{inner}: {v} m")
        monthly = ', '.join(monthly_parts)
        lines.append(f"Historical Groundwater (2017-18): YearlyAvg={yearly_avg} m | Monthly -> {monthly}")
    else:
        lines.append('Historical groundwater (2017-18): not available for this district')

    # ── Inject 2019-2021 depth-to-water-level trends ──
    depth_lines = []
    for year in ['2019', '2020', '2021']:
        row = DEPTH_GW_DATA.get((location.lower(), year))
        if row:
            depth_lines.append(
                f"  {year}: stations={row.get('No of station','N/A')}, "
                f"0-2m={row.get('Depth to Water Level (mbgl) - 0 - 2 - %','N/A')}%, "
                f"2-5m={row.get('Depth to Water Level (mbgl) - 45048 - %','N/A')}%, "
                f"5-10m={row.get('Depth to Water Level (mbgl) - 45204 - %','N/A')}%, "
                f"10-20m={row.get('Depth to Water Level (mbgl) - 44105 - %','N/A')}%, "
                f"20-40m={row.get('Depth to Water Level (mbgl) - 20 - 40 - %','N/A')}%, "
                f">40m={row.get('Depth to Water Level (mbgl) - > 40 - %','N/A')}%"
            )
    if depth_lines:
        lines.append('Depth-to-Water-Level Distribution (2019-2021):\n' + '\n'.join(depth_lines))
    else:
        lines.append('Depth-to-Water-Level (2019-2021): not available for this district')

    # ── Inject district categorization data ──
    cat_row = CAT_GW_DATA.get(location.lower())
    if cat_row:
        lines.append(
            f"District Category: {cat_row.get('Category','N/A')} | "
            f"GW Extraction Stage: {cat_row.get('Stage_of_GW_Extraction_pct','N/A')}% | "
            f"Over-Exploited Units: {cat_row.get('Over_Exploited_Units','N/A')} | "
            f"Critical Units: {cat_row.get('Critical_Units','N/A')} | "
            f"Safe Units: {cat_row.get('Safe_Units','N/A')} | "
            f"Key Issue: {cat_row.get('Key_Issue','N/A')}"
        )
    else:
        lines.append('District categorization data: not available')

    context = '\n'.join(lines)

    # ── Inject relevant PDF chunks ──
    relevant_chunks = find_relevant_chunks(query, PDF_CHUNKS, top_k=4)
    pdf_section = ''
    if relevant_chunks:
        pdf_parts = [f'[{c["source"]} p.{c["page"]}] {c["text"][:600]}' for c in relevant_chunks]
        pdf_section = '\n\nReference Documents (relevant excerpts):\n' + '\n\n'.join(pdf_parts)

    return f"""You are Puvi Jal AI, a groundwater and water resource assistant for Tamil Nadu, India.
Use ONLY the live data context below to answer. Do not invent numbers.
Answer in 6-10 lines. Use bullet points, bold for key values, emojis where relevant. No filler phrases.

Live Data Context:
{context}{pdf_section}

User Question: {query}

Answer:"""

def generate_answer(query):
    # ── State-level weather query (any Indian state) ──
    if is_weather_query(query):
        state = extract_state(query)
        if state:
            return handle_state_weather_query(query, state)

    location = extract_location(query)

    if location:
        weather = fetch_weather(location)
        live_gw = fetch_live_gw(location)
        live_rainfall = fetch_live_rainfall(location)
        log_weather(location, weather)

        # ── Build structured markdown from live data ──
        weather_md = ''
        if weather and weather.get('current'):
            c = weather['current']
            weather_md = (
                f'\n### 🌤️ Current Weather\n'
                f'- **Condition:** {c.get("condition")} | **Temp:** {c.get("temp_c")}°C | **Humidity:** {c.get("humidity")}%\n'
                f'- **Today Rain:** {c.get("precip_mm", 0)} mm | **7-Day Total:** {weather.get("total_forecast_rain_mm", 0)} mm\n'
            )
            if weather.get('forecast'):
                weather_md += '\n| Date | Rainfall (mm) | Condition |\n|------|--------------|-----------|\n'
                for day in weather['forecast']:
                    weather_md += f'| {day["date"]} | {day["rainfall_mm"]} | {day["condition"]} |\n'

        gw_md = ''
        if live_gw:
            levels = [r['dataValue'] for r in live_gw if r.get('dataValue') is not None]
            if levels:
                latest = sorted(live_gw, key=lambda r: (
                    r['dataTime']['year'], r['dataTime']['monthValue'], r['dataTime']['dayOfMonth']
                ), reverse=True)[0]
                gw_md = (
                    f'\n### 📡 Live Groundwater (WRIS/CGWB)\n'
                    f'- **Station:** {latest["stationName"]}\n'
                    f'- **Water Level:** {latest["dataValue"]} m | **Avg:** {round(mean(levels), 2)} m\n'
                    f'- **Date:** {latest["dataTime"]["dayOfMonth"]} {latest["dataTime"]["month"].title()} {latest["dataTime"]["year"]}\n'
                    f'- **Well Depth:** {latest.get("wellDepth", "N/A")} m\n'
                )

        rf_md = ''
        if live_rainfall and isinstance(live_rainfall, list) and len(live_rainfall) > 0:
            rf_vals = [r['dataValue'] for r in live_rainfall if r.get('dataValue') is not None]
            if rf_vals:
                latest_rf = sorted(live_rainfall, key=lambda r: (
                    r['dataTime']['year'], r['dataTime']['monthValue'], r['dataTime']['dayOfMonth']
                ), reverse=True)[0]
                rf_md = (
                    f'\n### 🌧️ Live Rainfall (WRIS/CWC)\n'
                    f'- **Station:** {latest_rf["stationName"]}\n'
                    f'- **Rainfall:** {latest_rf["dataValue"]} mm | **Avg:** {round(mean(rf_vals), 2)} mm\n'
                    f'- **Date:** {latest_rf["dataTime"]["dayOfMonth"]} {latest_rf["dataTime"]["month"].title()} {latest_rf["dataTime"]["year"]}\n'
                )

        # ── Single Gemini call with all data embedded ──
        llm_analysis = call_llm(build_gemini_prompt(query, weather, live_gw, live_rainfall, location))

        base = f'## 📍 {location.title()}, Tamil Nadu\n{weather_md}{gw_md}{rf_md}\n---\n\n'
        base += f'### 🧠 Analysis\n{llm_analysis}' if llm_analysis else '### 🧠 Analysis\n*(AI analysis unavailable — quota limit reached. Live data above is accurate.)*'
        return base

    # ── No location found — use PDF context + single Gemini call ──
    relevant_chunks = find_relevant_chunks(query, PDF_CHUNKS, top_k=5)
    pdf_section = ''
    if relevant_chunks:
        pdf_parts = [f'[{c["source"]} p.{c["page"]}] {c["text"][:600]}' for c in relevant_chunks]
        pdf_section = '\n\nReference Documents (relevant excerpts):\n' + '\n\n'.join(pdf_parts)

    # ── Inject categorization summary for general queries ──
    cat_summary_lines = []
    for cat in ['Over-Exploited', 'Critical', 'Semi-Critical', 'Safe']:
        districts_in_cat = [row['District'] for row in CAT_GW_DATA.values() if row.get('Category') == cat]
        if districts_in_cat:
            cat_summary_lines.append(f"{cat}: {', '.join(districts_in_cat)}")
    cat_summary = '\n'.join(cat_summary_lines)

    llm_response = call_llm(
        f"""You are Puvi Jal AI, a groundwater and water resource assistant for Tamil Nadu and India.
Use the reference documents below if relevant. Answer factually and in detail.
Answer in 6-10 lines using bullet points, bold for key values, and emojis. No filler phrases.

Tamil Nadu District Groundwater Categories:
{cat_summary}
{pdf_section}

Question: {query}

Answer:"""
    )
    return llm_response or "I couldn't find relevant data. Please ask about a specific location or groundwater topic."

# ── Profile Endpoints ────────────────────────────────────────────────────────
@app.route('/auth/me', methods=['GET'])
def get_me():
    email = get_email_from_token()
    if not email:
        return jsonify({'detail': 'Unauthorized'}), 401
    user = users_col.find_one({'email': email}, {'password': 0, '_id': 0})
    if not user:
        return jsonify({'detail': 'User not found'}), 404
    return jsonify(user)

@app.route('/auth/update-profile', methods=['POST'])
def update_profile():
    email = get_email_from_token()
    if not email:
        return jsonify({'detail': 'Unauthorized'}), 401
    data = request.json
    updates = {}
    if 'name' in data and data['name'].strip():
        updates['name'] = data['name'].strip()
    if 'avatar' in data:
        updates['avatar'] = data['avatar']  # base64 string
    if not updates:
        return jsonify({'detail': 'Nothing to update'}), 400
    users_col.update_one({'email': email}, {'$set': updates})
    user = users_col.find_one({'email': email}, {'password': 0, '_id': 0})
    return jsonify(user)

@app.route('/auth/change-password', methods=['POST'])
def change_password():
    email = get_email_from_token()
    if not email:
        return jsonify({'detail': 'Unauthorized'}), 401
    data = request.json
    old_pw = data.get('old_password', '')
    new_pw = data.get('new_password', '')
    if not old_pw or not new_pw:
        return jsonify({'detail': 'Both old and new password required'}), 400
    user = users_col.find_one({'email': email})
    if not user or not bcrypt.checkpw(old_pw.encode(), user['password']):
        return jsonify({'detail': 'Old password is incorrect'}), 401
    hashed = bcrypt.hashpw(new_pw.encode(), bcrypt.gensalt())
    users_col.update_one({'email': email}, {'$set': {'password': hashed}})
    return jsonify({'status': 'ok'})

# ── Auth Endpoints ─────────────────────────────────────────────────────────────
@app.route('/auth/signup', methods=['POST'])
def signup():
    data = request.json
    email = data.get('email')
    if users_col.find_one({'email': email}):
        return jsonify({'detail': 'User already exists'}), 400
    hashed_pw = bcrypt.hashpw(data.get('password').encode(), bcrypt.gensalt())
    users_col.insert_one({'name': data.get('name'), 'email': email, 'password': hashed_pw})
    token = jwt.encode({'email': email, 'exp': datetime.utcnow() + timedelta(days=7)}, app.config['SECRET_KEY'])
    return jsonify({'token': token, 'user': {'name': data.get('name'), 'email': email}})

@app.route('/auth/signin', methods=['POST'])
def signin():
    data = request.json
    email = data.get('email')
    user = users_col.find_one({'email': email})
    if not user or not bcrypt.checkpw(data.get('password').encode(), user['password']):
        return jsonify({'detail': 'Invalid credentials'}), 401
    token = jwt.encode({'email': email, 'exp': datetime.utcnow() + timedelta(days=7)}, app.config['SECRET_KEY'])
    return jsonify({'token': token, 'user': {'name': user['name'], 'email': email}})

# ── Chat History Endpoints ─────────────────────────────────────────────────────
@app.route('/chats', methods=['GET'])
def get_chats():
    email = get_email_from_token()
    if not email:
        return jsonify({'detail': 'Unauthorized'}), 401
    chats = list(chats_col.find({'email': email}, {'_id': 1, 'name': 1, 'updated_at': 1}))
    for c in chats:
        c['_id'] = str(c['_id'])
    return jsonify(chats)

@app.route('/chats', methods=['POST'])
def create_chat():
    email = get_email_from_token()
    if not email:
        return jsonify({'detail': 'Unauthorized'}), 401
    data = request.json
    result = chats_col.insert_one({
        'email': email, 'name': data.get('name', 'New Chat'),
        'messages': [], 'created_at': datetime.utcnow(), 'updated_at': datetime.utcnow()
    })
    return jsonify({'_id': str(result.inserted_id), 'name': data.get('name', 'New Chat')})

@app.route('/chats/<chat_id>', methods=['GET'])
def get_chat(chat_id):
    email = get_email_from_token()
    if not email:
        return jsonify({'detail': 'Unauthorized'}), 401
    chat = chats_col.find_one({'_id': ObjectId(chat_id), 'email': email})
    if not chat:
        return jsonify({'detail': 'Not found'}), 404
    chat['_id'] = str(chat['_id'])
    return jsonify(chat)

@app.route('/chats/<chat_id>/messages', methods=['POST'])
def add_message(chat_id):
    email = get_email_from_token()
    if not email:
        return jsonify({'detail': 'Unauthorized'}), 401
    data = request.json
    message = {'sender': data.get('sender'), 'text': data.get('text'), 'timestamp': datetime.utcnow().isoformat()}
    chats_col.update_one(
        {'_id': ObjectId(chat_id), 'email': email},
        {'$push': {'messages': message}, '$set': {'updated_at': datetime.utcnow()}}
    )
    return jsonify({'status': 'ok'})

@app.route('/chats/<chat_id>', methods=['DELETE'])
def delete_chat(chat_id):
    email = get_email_from_token()
    if not email:
        return jsonify({'detail': 'Unauthorized'}), 401
    chats_col.delete_one({'_id': ObjectId(chat_id), 'email': email})
    return jsonify({'status': 'deleted'})

# ── Other Endpoints ────────────────────────────────────────────────────────────
@app.route('/api/groundwater-live', methods=['GET'])
def groundwater_live():
    district = request.args.get('district', 'Chennai')
    start = request.args.get('start', '2025-06-01')
    end = request.args.get('end', '2026-06-30')
    data = fetch_live_gw(district, start, end)
    if not data:
        return jsonify({'error': 'No data available'}), 502
    levels = [r['dataValue'] for r in data if r.get('dataValue') is not None]
    stations = list({r['stationName'] for r in data})
    latest = sorted(data, key=lambda r: (r['dataTime']['year'], r['dataTime']['monthValue'], r['dataTime']['dayOfMonth']), reverse=True)[0]
    return jsonify({
        'district': district, 'station_count': len(stations), 'stations': stations,
        'avg_water_level_m': round(mean(levels), 2) if levels else None,
        'latest': {
            'station': latest['stationName'], 'value_m': latest['dataValue'],
            'date': f"{latest['dataTime']['dayOfMonth']} {latest['dataTime']['month'].title()} {latest['dataTime']['year']}",
            'well_depth_m': latest.get('wellDepth'),
        },
        'total_records': len(data)
    })

@app.route('/api/rainfall-live', methods=['GET'])
def rainfall_live():
    district = request.args.get('district', 'Chennai')
    start = request.args.get('start', '2026-01-01')
    end = request.args.get('end', '2026-06-30')
    data = fetch_live_rainfall(district, start, end)
    if not data:
        return jsonify({'error': 'No rainfall data available'}), 502
    return jsonify({'district': district, 'data': data})

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
        import numpy as np

        csv_path = os.path.join(os.path.dirname(__file__), '..', 'SIH Files', 'tamilnadu_groundwater.csv')
        rows = []
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        districts = [r['District'].title() for r in rows]
        le = LabelEncoder()
        le.fit(districts)

        def safe_float(v):
            try: return float(v)
            except: return 0.0

        X = [[
            le.transform([r['District'].title()])[0],
            safe_float(r['Rainfall_mm']),
            safe_float(r['Annual_GW_Recharge_ham']),
            safe_float(r['Annual_Extractable_GW_ham']),
            safe_float(r['GW_Extraction_Total_ham'])
        ] for r in rows]
        y = [safe_float(r['Stage_of_GW_Extraction_pct']) for r in rows]

        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X, y)

        results = []
        for i, r in enumerate(rows):
            current = rf.predict([X[i]])[0]
            X_future = X[i].copy()
            X_future[4] *= 1.05
            future = rf.predict([X_future])[0]
            trend = 'Declining' if future > current else 'Stable'
            status = 'Over-Exploited' if current > 100 else ('Critical' if current > 90 else ('Semi-Critical' if current > 70 else 'Safe'))
            results.append({
                'region': r['District'].title(),
                'description': f"Groundwater extraction stage: {current:.1f}%",
                'trend': trend,
                'confidence': round(min(95, max(70, 100 - abs(future - current))), 1),
                'period': '2024-2025',
                'current_pct': round(current, 1),
                'forecast_pct': round(future, 1),
                'status': status,
                'rainfall_mm': round(safe_float(r['Rainfall_mm']), 1),
            })
        results.sort(key=lambda x: x['current_pct'], reverse=True)
        return jsonify(results)
    except Exception as e:
        print(f'Predictions error: {e}')
        return jsonify({'error': str(e)}), 500

@app.route('/analysis/', methods=['POST'])
def analysis():
    data = request.json
    return jsonify({
        'period': data.get('period'), 'year': data.get('year'),
        'summary': f'Analysis for {data.get("period")} {data.get("year")}',
        'metrics': {'water_level': 45.2, 'change': -5.3, 'status': 'Declining'}
    })

# ── Background Weather Logger for all 38 TN Districts ──────────────────────
_ALL_TN_DISTRICTS = [
    'Ariyalur','Chengalpattu','Chennai','Coimbatore','Cuddalore','Dharmapuri',
    'Dindigul','Erode','Kallakurichi','Kancheepuram','Kanyakumari','Karur',
    'Krishnagiri','Madurai','Mayiladuthurai','Nagapattinam','Namakkal','Nilgiris',
    'Perambalur','Pudukkottai','Ramanathapuram','Ranipet','Salem','Sivaganga',
    'Tenkasi','Thanjavur','Theni','Thoothukudi','Tiruchirappalli','Tirunelveli',
    'Tirupathur','Tiruppur','Tiruvallur','Tiruvannamalai','Tiruvarur','Vellore',
    'Viluppuram','Virudhunagar'
]

def auto_log_all_districts():
    """Fetch and log weather for all 38 TN districts. Runs every 24 hours."""
    while True:
        print(f'\n⏰ Auto weather logging started at {datetime.utcnow().strftime("%Y-%m-%d %H:%M")} UTC')
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_weather, d): d for d in _ALL_TN_DISTRICTS}
            success = 0
            for future, district in futures.items():
                try:
                    result = future.result()
                    log_weather(district, result)
                    if result:
                        success += 1
                except Exception as e:
                    print(f'Auto log error for {district}: {e}')
        print(f'✅ Auto logged weather for {success}/38 districts')
        threading.Event().wait(86400)  # wait 24 hours

def start_background_logger():
    t = threading.Thread(target=auto_log_all_districts, daemon=True)
    t.start()
    print('✅ Background weather logger started (every 24h for all 38 TN districts)')

if __name__ == '__main__':
    start_background_logger()
    app.run(port=8000, debug=True)
