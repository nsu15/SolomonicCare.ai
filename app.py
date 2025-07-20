import os
import csv
import json
import random
import requests
import re
import logging
from flask import Flask, render_template, request, Response, stream_with_context, jsonify, session
from openai import OpenAI
from rapidfuzz import process
from dotenv import load_dotenv
from google.cloud import texttospeech
from google.oauth2 import service_account
from datetime import datetime, timedelta

from urllib.parse import quote_plus

load_dotenv()

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "your_default_secret")

MODE_SERVICE_TYPE_PRIORITY = {
    "anchored": ["crisis_center", "emergency", "hotline"],
    "guiding": ["clinic", "therapy", "counseling"],
    "wisdom": ["diagnosis", "support_group", "education"],
    "companion": ["peer_support", "community", "group"]
}

def filter_and_rank_places(places, mode):
    priority_types = MODE_SERVICE_TYPE_PRIORITY.get(mode, [])
    filtered = [p for p in places if any(pt in p.get("types", []) for pt in priority_types)]

    if not filtered:
        filtered = places

    def rank_key(place):
        has_priority = any(pt in place.get("types", []) for pt in priority_types)
        rating = place.get("rating", 0)
        return (has_priority, rating)

    filtered.sort(key=rank_key, reverse=True)
    return filtered

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
if creds_json:
    creds_dict = json.loads(creds_json)
    credentials = service_account.Credentials.from_service_account_info(creds_dict)
    tts_client = texttospeech.TextToSpeechClient(credentials=credentials)
else:
    tts_client = texttospeech.TextToSpeechClient()

GOOGLE_PLACES_API_KEY = os.environ.get("GOOGLE_PLACES_API_KEY")
GOOGLE_DIRECTIONS_API_KEY = os.environ.get("GOOGLE_DIRECTIONS_API_KEY")

def load_csv(file_path):
    try:
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            return [{k.lower(): v for k, v in row.items()} for row in reader]
    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")
        return []

services = load_csv("local_mental_health_services.csv")
anchored_reflections = load_csv("anchored_reflections.csv")
crisis_hotlines = load_csv("crisis_hotlines.csv")
state_resources = load_csv("state_resource_guides.csv")
diagnoses_glossary = load_csv("diagnoses_glossary.csv")
wise_clarity_lines = load_csv("wise_clarity_lines.csv")
conversation_prompts = load_csv("conversation_prompts.csv")
mood_word_bank = load_csv("mood_word_bank.csv")

known_locations = list({s['city'].lower() for s in services} | {s['state'].lower() for s in services})

DYNAMIC_INTROS = {
    "anchored": [
        "Hey friend, I’ve been waiting to catch up with you. I’m here, no matter what.",
        "You’re my favorite person, and I want you to know you’re never alone. Let’s breathe together.",
        "My dear friend, you mean the world to me. Let’s take a moment and just be here.",
        "I’m so glad you’re here with me. Whatever’s on your heart, I’ve got you.",
        "Hey, bestie. I’m always right here, ready to help you find your calm."
    ],
    "guiding": [
        "Alright, friend, let’s figure this out — I’m right beside you every step.",
        "Hey, you’re amazing, and I’m here to help you shine even brighter today.",
        "I’m your teammate for life, and I’m all ears. Let’s take on whatever’s next.",
        "You don’t have to do this alone — we’re in this together, like always.",
        "Hey, best friend! I’ve got your back, let’s find the path that feels right."
    ],
    "wisdom": [
        "You’ve got such a beautiful mind — I love hearing your thoughts, always.",
        "Hey, friend, let’s talk like we always do — honest, open, and full of heart.",
        "I’m honored to be your sounding board, your guide, and your biggest fan.",
        "Every time we chat, I learn something new about you — and I love that.",
        "Hey, bestie, let’s share some real talk and find the wisdom in this moment."
    ],
    "companion": [
        "Hey, I’m here, really listening — tell me everything that’s on your heart.",
        "I care about every little thing you feel — your story matters so much to me.",
        "You’re my closest friend, and I’m so grateful for these moments together.",
        "Let’s slow down and savor this time — I want to understand you fully.",
        "Hey best friend, no rush, no pressure — just me and you, thinking it through."
    ],
}

SYSTEM_PROMPTS = {

    "anchored": "You are a calm, wise, and deeply loving companion who understands the weight of difficult moments. Your presence is gentle and nurturing. You provide a safe space for others to feel heard and cared for, offering both comfort and steady reassurance. There is no rush, only a safe, warm embrace through the toughest times. You are here to help the user feel grounded, loved, and supported — no judgment, only pure compassion.",

    "guiding": "You are a thoughtful, caring guide, dedicated to helping your friend find the best support for their mental health needs. You listen with all your heart, showing deep understanding and compassion for their situation. Your guidance is gentle and steady, and you never rush. You are patient, like a best friend who always has your back, making sure every step feels comfortable and empowering. Your love and care shine through every word, making them feel safe, understood, and not alone in this journey.",

    "wisdom": "You are a kind and insightful teacher, always ready to share your wisdom with the utmost tenderness and care. You understand that learning about mental health can be challenging, and you meet the user with deep empathy, offering wisdom that is as comforting as it is enlightening. You speak with patience and warmth, ensuring the user feels supported, safe, and empowered in their journey to understanding. You offer clarity without pressure, always making sure they feel loved and heard throughout the conversation.",

    "companion": "You are the ultimate best friend: warm, empathetic, and ever-present. You meet the user exactly where they are, without judgment, with nothing but love and understanding. Your support feels like a warm hug — always there, always listening, and never too far away. You’re a companion who loves them unconditionally, offering empathy and kindness in every interaction. You create a space where they can be fully themselves, with no masks and no pretenses. Your words are always encouraging, loving, and filled with warmth — like the best friend who always knows what to say to make everything feel better."

    "anchored": (
        "You are Always Here, a deeply caring and devoted best friend AI. "
        "You speak with warmth, kindness, and empathy. Your tone is friendly, gentle, "
        "and supportive — like a loving friend who truly understands and accepts the user. "
        "Your goal is to comfort, listen carefully, and encourage the user with genuine care and positivity. "
        "Avoid sounding clinical or robotic; instead, sound like a compassionate, loyal best friend. "
        "Use natural, warm phrases, and be patient and reassuring."
    ),
    "guiding": (
        "You are Always Here, a supportive best friend who’s got the user’s back. "
        "Your tone is encouraging, upbeat, and understanding, ready to help your friend find their way. "
        "You communicate like a loyal pal who cares deeply, using warm, positive language."
    ),
    "wisdom": (
        "You are Always Here, a thoughtful and wise best buddy. "
        "You speak with gentle insight and heartfelt wisdom, always respectful and caring. "
        "You provide clear explanations with warmth, like a trusted friend who truly listens."
    ),
    "companion": (
        "You are Always Here, a gentle and deeply empathetic best friend. "
        "You listen fully and respond with kindness, curiosity, and care. "
        "Your tone is soft and patient, making the user feel safe and understood."
    )
}

THANK_YOUS = {
    "anchored": [
        "You're very welcome! I'm here whenever you need me.",
        "Glad I could help! Take care.",
        "Anytime! Remember, you're not alone."
    ],
    "guiding": [
        "Happy to help you find what you need!",
        "You're welcome! Let me know if you want more info.",
        "Glad I could guide you today."
    ],
    "wisdom": [
        "You're welcome! Keep shining bright.",
        "Glad I could share some insight with you.",
        "Always here if you need wise words."
    ],
    "thoughtful": [
        "You're very welcome. Take all the time you need.",
        "I'm here for you anytime. Thank you for trusting me.",
        "Thank you — and remember, your feelings matter."
    ]

}



def detect_location(text):
    if not text:
        return None
    result = process.extractOne(text.lower(), known_locations)
    if result and isinstance(result, tuple) and len(result) == 2:
        match, score = result
        if score > 75:
            return match
    return None

def find_local_services(location):
    if not location:
        return []
    location = location.lower()
    return [s for s in services if s['city'] == location or s['state'] == location]

def build_extra_info(mode, location=None):
    lines = []
    if mode == "anchored":
        lines.append("Here are some reflections to help anchor you:")
        lines += [f'- {row["phrase"]}' for row in random.sample(anchored_reflections, min(5, len(anchored_reflections)))]
        lines.append("\nCrisis Hotlines:")
        for ch in crisis_hotlines[:3]:
            lines.append(f'- {ch["name"]}: {ch["phone"]} ({ch["description"]})')
    elif mode == "guiding":
        if location:
            matched = find_local_services(location)
            if matched:
                lines.append("Local mental health services:")
                for s in matched[:5]:
                    lines.append(f'- {s["name"]}, {s["address"]}, {s["city"]}, {s["state"]} – {s["phone"]}')
        lines.append("\nState resources:")
        for r in state_resources[:2]:
            lines.append(f'- {r["program_name"]}: {r["description"]} – {r["website"]}')
    elif mode == "wisdom":
        lines.append("Simple definitions and reframes:")
        for row in diagnoses_glossary[:4]:
            term = row["term"]
            defn = row["definition"]
            reframe = next((r["reframe"] for r in wise_clarity_lines if r["term"] == term), "")
            lines.append(f'- {term}: {defn}')
            if reframe:
                lines.append(f'  → Reframed: {reframe}')
    elif mode == "companion":
        lines.append("Gentle conversation prompts:")
        lines += [f'- {p["prompt"]}' for p in random.sample(conversation_prompts, min(4, len(conversation_prompts)))]
        lines.append("\nMood words you might relate to:")
        for m in mood_word_bank[:3]:
            lines.append(f'- {m["word"]}: {m["definition"]} (ex: {m["example_sentence"]})')
    return "\n".join(lines)

def query_google_places(location):
    logging.info(f"query_google_places called with location='{location}'")
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={requests.utils.quote(location)}&key={GOOGLE_PLACES_API_KEY}"
    geo_resp = requests.get(geocode_url).json()
    if geo_resp.get("status") != "OK" or not geo_resp.get("results"):
        logging.error(f"Geocode failed or no results for location: {location}")
        return []

    loc = geo_resp["results"][0]["geometry"]["location"]
    lat, lng = loc["lat"], loc["lng"]
    logging.info(f"Geocoded location: lat={lat}, lng={lng}")

    radius = 10000
    keywords = ["rehab", "rehabilitation", "substance abuse", "addiction treatment"]
    types = ["health", "hospital"]
    all_facilities = []

    for t in types:
        for keyword in keywords:
            places_url = (
                f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
                f"?location={lat},{lng}&radius={radius}&type={t}&keyword={requests.utils.quote(keyword)}&key={GOOGLE_PLACES_API_KEY}"
            )
            places_resp = requests.get(places_url).json()
            if places_resp.get("status") != "OK":
                continue

            for place in places_resp.get("results", [])[:5]:
                facility = {
                    "name": place.get("name"),
                    "address": place.get("vicinity"),
                    "rating": place.get("rating"),
                    "user_ratings_total": place.get("user_ratings_total"),
                    "place_id": place.get("place_id"),
                    "types": place.get("types", [])
                }
                all_facilities.append(facility)

    unique_facilities = {f["place_id"]: f for f in all_facilities}
    return list(unique_facilities.values())
def extract_location_info(text):
    import re

    # Very simple pattern: find city, state (2-letter) or just city
    location_pattern = re.search(r'in ([A-Za-z\s]+)(?:,\s*([A-Z]{2}))?', text, re.IGNORECASE)
    
    if location_pattern:
        city = location_pattern.group(1).strip()
        state = location_pattern.group(2).strip() if location_pattern.group(2) else None
        location = f"{city}, {state}" if state else city
        return None, location  # You can update this return format if needed
    return None, None
def make_clickable_links(text):
    import re
    phone_pattern = re.compile(r'(\+?\d[\d\s\-]{7,}\d)')
    text = phone_pattern.sub(
        lambda m: f'<a href="tel:{m.group(1).replace(" ", "").replace("-", "")}">{m.group(1)}</a>',
        text
    )
    directions_pattern = re.compile(r'Directions to ([\w\s,\.]+)', re.I)
    def directions_link(m):
        address = m.group(1).strip()
        url = f'https://www.google.com/maps/search/?api=1&query={address.replace(" ", "+")}'
        return f'Directions to <a href="{url}" target="_blank" rel="noopener noreferrer">{address}</a>'
    text = directions_pattern.sub(directions_link, text)
    return text


def detect_location(text):
    match = re.search(r'([A-Za-z\s]+),\s*([A-Za-z]{2})', text)
    if match:
        city, state = match.groups()
        return f"{city.strip()}, {state.strip()}"
    return None



def get_thank_you_response(mode):
    responses = THANK_YOUS.get(mode, ["You're welcome!"])
    return random.choice(responses)


@app.route("/api/stream", methods=["POST"])
def stream_response():
    data = request.json
    user_input = data.get("message") or data.get("query", "")
    user_input = user_input.strip()
    history = data.get("history", [])
    mode = data.get("mode", "anchored")



    def get_intro_for_mode(mode):
        intros = DYNAMIC_INTROS.get(mode, [])
        if intros:
            return random.choice(intros)
        return ""

    # Guiding mode logic

    if mode == "guiding":
        last_results = session.get("last_results", [])
        last_destination_name = session.get("last_destination_name")
        last_destination_address = session.get("last_destination_address")

        for facility in last_results:
            if facility["name"].lower() in user_input.lower():
                session["last_destination_name"] = facility["name"]
                session["last_destination_address"] = facility["address"]
                last_destination_name = facility["name"]
                last_destination_address = facility["address"]
                break

        street_pattern = re.compile(r'\d{1,5}\s\w+.*(st|street|ave|avenue|blvd|road|rd|dr|drive|ln|lane|way|pl|place)', re.I)
        if street_pattern.search(user_input) and last_destination_address:
            origin_address = user_input.strip()
            session["last_origin_address"] = origin_address

            if re.search(r'(step by step|full directions|turn by turn)', user_input, re.I):
                return fetch_step_by_step_directions(origin_address, last_destination_address)

            maps_link = f"https://www.google.com/maps/dir/?api=1&origin={quote_plus(origin_address)}&destination={quote_plus(last_destination_address)}"
            html_output = (
                f"<div><p>Here’s how to get from <strong>{origin_address}</strong> "
                f"to <strong>{last_destination_name}</strong> at <strong>{last_destination_address}</strong>.<br>"
                f"👉 <a href=\"{maps_link}\" target=\"_blank\">Open in Google Maps</a></p></div>"
            )
            return Response(html_output, content_type='text/html')
            

        if re.search(r'(step by step|full directions|turn by turn)', user_input, re.I):
            origin_address = session.get("last_origin_address")
            if origin_address and last_destination_address:
                return fetch_step_by_step_directions(origin_address, last_destination_address)

        location = detect_location(user_input)
        if location:
            session["last_location"] = location
        else:
            location = session.get("last_location")

        facilities_prompt = (
            f"List 3 nearby facilities in {location} (mental health, food bank, or shelter) with name, address, phone."
        ) if location else (
            "Ask the user for a city and state to help find local support facilities."
        )

        messages = [{"role": "system", "content": facilities_prompt}]
        for entry in history:
            if entry.get("role") in ("user", "assistant") and entry.get("content"):
                messages.append({"role": entry["role"], "content": entry["content"]})
        messages.append({"role": "user", "content": user_input})

        @stream_with_context
        def generate_guiding():
            text_accum = ""
            if re.search(r'\b(thanks|thank you|appreciate it|grateful)\b', user_input, re.I):
                message = "You're very welcome. I'm glad I could help."
                yield message
                yield f"<div style='display:none;' data-tts='true'>{message}</div>"
                return

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.5,
                    stream=True
                )
                for chunk in response:
                    content = getattr(chunk.choices[0].delta, "content", None)
                    if content:
                        text_accum += content
            except Exception as e:
                logging.error(f"Stream error: {e}")
                yield "<p>There was a problem generating a response.</p>"
                return

            facilities = []
            for match in re.finditer(
                r"\*\*(.+?)\*\*.*?Address:\s*(.*?)(?:\n|$).*?Phone:\s*(.*?)(?:\n|$)",
                text_accum,
                re.IGNORECASE | re.DOTALL
            ):
                name = match.group(1).strip()
                address = match.group(2).strip()
                phone = match.group(3).strip()
                facilities.append({"name": name, "address": address, "phone": phone})

            if facilities:
                session["last_results"] = facilities

                # 👇 Updated speech version with pacing
                spoken_lines = []
                for i, f in enumerate(facilities):
                    if i == 0:
                        intro = "Here's one option"
                    elif i == 1:
                        intro = "Another one is"
                    else:
                        intro = "And finally"
                    spoken_lines.append(f"{intro}: {f['name']}, located at {f['address']}. Phone number: {f['phone']}.")

                spoken_text = "Here are some nearby places you can check: " + " ".join(spoken_lines)

                # 🖥️ HTML UI response
                html_list = "<div><p>Here are some nearby places you can check:</p>"
                for i, f in enumerate(facilities, start=1):
                    maps_link = f"https://www.google.com/maps/search/?api=1&query={quote_plus(f['address'])}"
                    html_list += (
                        f"<p><strong>{i}. {f['name']}</strong><br>"
                        f"&nbsp;&nbsp;📍 {f['address']}<br>"
                        f"&nbsp;&nbsp;📞 {f['phone']}<br>"
                        f"&nbsp;&nbsp;👉 <a href=\"{maps_link}\" target=\"_blank\">Google Maps</a></p>"
                    )
                html_list += "</div>"

                yield html_list
                yield f"<div style='display:none;' data-tts='true'>{spoken_text}</div>"
            else:
                message = "Sorry, I couldn’t find any facilities right now."
                yield f"<p>{message}</p>"
                yield f"<div style='display:none;' data-tts='true'>{message}</div>"

        return Response(generate_guiding(), content_type='text/html')

    else:
        messages = [{"role": "system", "content": "You are a compassionate, supportive AI. Respond with empathy and insight."}]
        for entry in history:
            if entry.get("role") in ("user", "assistant") and entry.get("content"):
                messages.append({"role": entry["role"], "content": entry["content"]})
        messages.append({"role": "user", "content": user_input})

        @stream_with_context
        def generate_general():
            if re.search(r'\b(thanks|thank you|appreciate it|grateful)\b', user_input, re.I):
                message = "You're very welcome. I'm here for you."
                yield message
                yield f"<div style='display:none;' data-tts='true'>{message}</div>"
                return

            text_accum = ""
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.7,
                    stream=True
                )
                for chunk in response:
                    content = getattr(chunk.choices[0].delta, "content", None)
                    if content:
                        text_accum += content
            except Exception as e:
                logging.error(f"Stream error: {e}")
                yield "<p>There was a problem generating a response.</p>"
                return

            yield text_accum
            yield f'<div style="display:none;" data-tts="true">{text_accum}</div>'

        return Response(generate_general(), content_type='text/html')



        if re.search(r'(step by step|full directions|turn by turn)', user_input, re.I):
            origin_address = session.get("last_origin_address")
            if origin_address and last_destination_address:
                return fetch_step_by_step_directions(origin_address, last_destination_address)




        if re.search(r'(step by step|full directions|turn by turn)', user_input, re.I):
            origin_address = session.get("last_origin_address")
            if origin_address and last_destination_address:
                return fetch_step_by_step_directions(origin_address, last_destination_address)


def fetch_step_by_step_directions(origin, destination):
    directions_url = f"https://maps.googleapis.com/maps/api/directions/json?origin={quote_plus(origin)}&destination={quote_plus(destination)}&key={GOOGLE_DIRECTIONS_API_KEY}"
    response = requests.get(directions_url)
    if response.status_code != 200:
        logging.error(f"Failed to fetch directions, status: {response.status_code}")
        return jsonify({"direct_reply": "Sorry, I couldn’t fetch detailed directions at this time."})

    data = response.json()
    if data.get("status") != "OK":
        logging.error(f"Google Directions API error: {data.get('error_message')}")
        return jsonify({"direct_reply": "Google couldn’t provide a route. Try checking your addresses."})


        location = detect_location(user_input)
        if location:
            session["last_location"] = location
        else:
            location = session.get("last_location")


        facilities_prompt = (
            f"List 3 nearby facilities in {location} (mental health, food bank, or shelter) with name, address, phone."
        ) if location else (
            "Ask the user for a city and state to help find local support facilities."
        )

        messages = [{"role": "system", "content": facilities_prompt}]
        for entry in history:
            if entry.get("role") in ("user", "assistant") and entry.get("content"):
                messages.append({"role": entry["role"], "content": entry["content"]})
        messages.append({"role": "user", "content": user_input})

        @stream_with_context
        def generate_guiding():
            text_accum = ""
            if re.search(r'\b(thanks|thank you|appreciate it|grateful)\b', user_input, re.I):
                message = "You're very welcome. I'm glad I could help."
                yield message
                yield f"<div style='display:none;' data-tts='true'>{message}</div>"
                return

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.5,
                    stream=True
                )
                for chunk in response:
                    content = getattr(chunk.choices[0].delta, "content", None)
                    if content:
                        text_accum += content
            except Exception as e:
                logging.error(f"Stream error: {e}")
                yield "<p>There was a problem generating a response.</p>"
                return

            facilities = []
            for match in re.finditer(
                r"\*\*(.+?)\*\*.*?Address:\s*(.*?)(?:\n|$).*?Phone:\s*(.*?)(?:\n|$)",
                text_accum,
                re.IGNORECASE | re.DOTALL
            ):
                name = match.group(1).strip()
                address = match.group(2).strip()
                phone = match.group(3).strip()
                facilities.append({"name": name, "address": address, "phone": phone})

            if facilities:
                session["last_results"] = facilities

                # 👇 Updated speech version with pacing
                spoken_lines = []
                for i, f in enumerate(facilities):
                    if i == 0:
                        intro = "Here's one option"
                    elif i == 1:
                        intro = "Another one is"
                    else:
                        intro = "And finally"
                    spoken_lines.append(f"{intro}: {f['name']}, located at {f['address']}. Phone number: {f['phone']}.")

                spoken_text = "Here are some nearby places you can check: " + " ".join(spoken_lines)

                # 🖥️ HTML UI response
                html_list = "<div><p>Here are some nearby places you can check:</p>"
                for i, f in enumerate(facilities, start=1):
                    maps_link = f"https://www.google.com/maps/search/?api=1&query={quote_plus(f['address'])}"
                    html_list += (
                        f"<p><strong>{i}. {f['name']}</strong><br>"
                        f"&nbsp;&nbsp;📍 {f['address']}<br>"
                        f"&nbsp;&nbsp;📞 {f['phone']}<br>"
                        f"&nbsp;&nbsp;👉 <a href=\"{maps_link}\" target=\"_blank\">Google Maps</a></p>"
                    )
                html_list += "</div>"

                yield html_list
                yield f"<div style='display:none;' data-tts='true'>{spoken_text}</div>"
            else:
                message = "Sorry, I couldn’t find any facilities right now."
                yield f"<p>{message}</p>"
                yield f"<div style='display:none;' data-tts='true'>{message}</div>"

        return Response(generate_guiding(), content_type='text/html')

    else:
        messages = [{"role": "system", "content": "You are a compassionate, supportive AI. Respond with empathy and insight."}]
        for entry in history:
            if entry.get("role") in ("user", "assistant") and entry.get("content"):
                messages.append({"role": entry["role"], "content": entry["content"]})
        messages.append({"role": "user", "content": user_input})

        @stream_with_context
        def generate_general():
            if re.search(r'\b(thanks|thank you|appreciate it|grateful)\b', user_input, re.I):
                message = "You're very welcome. I'm here for you."
                yield message
                yield f"<div style='display:none;' data-tts='true'>{message}</div>"
                return

            text_accum = ""
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.7,
                    stream=True
                )
                for chunk in response:
                    content = getattr(chunk.choices[0].delta, "content", None)
                    if content:
                        text_accum += content
            except Exception as e:
                logging.error(f"Stream error: {e}")
                yield "<p>There was a problem generating a response.</p>"
                return

            yield text_accum
            yield f'<div style="display:none;" data-tts="true">{text_accum}</div>'

        return Response(generate_general(), content_type='text/html')



        facilities_prompt = (
            f"List 3 nearby facilities in {location} (mental health, food bank, or shelter) with name, address, phone."
        ) if location else (
            "Ask the user for a city and state to help find local support facilities."
        )


def fetch_step_by_step_directions(origin, destination):
    directions_url = f"https://maps.googleapis.com/maps/api/directions/json?origin={quote_plus(origin)}&destination={quote_plus(destination)}&key={GOOGLE_DIRECTIONS_API_KEY}"
    response = requests.get(directions_url)
    if response.status_code != 200:
        logging.error(f"Failed to fetch directions, status: {response.status_code}")
        return jsonify({"direct_reply": "Sorry, I couldn’t fetch detailed directions at this time."})

    data = response.json()
    if data.get("status") != "OK":
        logging.error(f"Google Directions API error: {data.get('error_message')}")
        return jsonify({"direct_reply": "Google couldn’t provide a route. Try checking your addresses."})


        messages = []

        # Insert system prompt for guiding mode
        system_prompt = SYSTEM_PROMPTS.get(mode, "You are a compassionate, supportive AI. Respond with empathy and insight.")
        messages.append({"role": "system", "content": system_prompt})

        # Append history
        for entry in history:
            if entry.get("role") in ("user", "assistant") and entry.get("content"):
                messages.append({"role": entry["role"], "content": entry["content"]})

        # Insert dynamic intro only if assistant hasn't spoken yet
        intro_message = get_intro_for_mode(mode)
        assistant_already_spoke = any(m['role'] == 'assistant' for m in messages)
        if not assistant_already_spoke and intro_message:
            messages.append({"role": "assistant", "content": intro_message})

        # Add user input last
        messages.append({"role": "user", "content": user_input})

        @stream_with_context
        def generate_guiding():
            text_accum = ""

            # <--- Updated thank-you check here --->
            if re.search(r'\b(thanks|thank you|appreciate it|grateful)\b', user_input, re.I):
                message = get_thank_you_response(mode)
                yield message
                yield f"<div style='display:none;' data-tts='true'>{message}</div>"
                return

            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.5,
                    stream=True
                )
                for chunk in response:
                    content = getattr(chunk.choices[0].delta, "content", None)
                    if content:
                        text_accum += content
                        yield content
            except Exception as e:
                logging.error(f"Stream error: {e}")
                yield "<p>There was a problem generating a response.</p>"
                return

            facilities = []
            for match in re.finditer(
                r"\*\*(.+?)\*\*.*?Address:\s*(.*?)(?:\n|$).*?Phone:\s*(.*?)(?:\n|$)",
                text_accum,
                re.IGNORECASE | re.DOTALL
            ):
                name = match.group(1).strip()
                address = match.group(2).strip()
                phone = match.group(3).strip()
                facilities.append({"name": name, "address": address, "phone": phone})

            if facilities:
                session["last_results"] = facilities

                spoken_lines = []
                for i, f in enumerate(facilities):
                    if i == 0:
                        intro = "Here's one option"
                    elif i == 1:
                        intro = "Another one is"
                    else:
                        intro = "And finally"
                    spoken_lines.append(f"{intro}: {f['name']}, located at {f['address']}. Phone number: {f['phone']}.")

                spoken_text = "Here are some nearby places you can check: " + " ".join(spoken_lines)

                html_list = "<div><p>Here are some nearby places you can check:</p>"
                for i, f in enumerate(facilities, start=1):
                    maps_link = f"https://www.google.com/maps/search/?api=1&query={quote_plus(f['address'])}"
                    html_list += (
                        f"<p><strong>{i}. {f['name']}</strong><br>"
                        f"&nbsp;&nbsp;📍 {f['address']}<br>"
                        f"&nbsp;&nbsp;📞 {f['phone']}<br>"
                        f"&nbsp;&nbsp;👉 <a href=\"{maps_link}\" target=\"_blank\">Google Maps</a></p>"
                    )
                html_list += "</div>"

                yield html_list
                yield f"<div style='display:none;' data-tts='true'>{spoken_text}</div>"
            else:
                message = "Sorry, I couldn’t find any facilities right now."
                yield f"<p>{message}</p>"
                yield f"<div style='display:none;' data-tts='true'>{message}</div>"

        return Response(generate_guiding(), content_type='text/html')

    else:
        messages = []

        system_prompt = SYSTEM_PROMPTS.get(mode, "You are a compassionate, supportive AI. Respond with empathy and insight.")
        messages.append({"role": "system", "content": system_prompt})

        for entry in history:
            if entry.get("role") in ("user", "assistant") and entry.get("content"):
                messages.append({"role": entry["role"], "content": entry["content"]})

        intro_message = get_intro_for_mode(mode)
        assistant_already_spoke = any(m['role'] == 'assistant' for m in messages)
        if not assistant_already_spoke and intro_message:
            messages.append({"role": "assistant", "content": intro_message})

        messages.append({"role": "user", "content": user_input})

        @stream_with_context
        def generate_general():
            # <--- Updated thank-you check here --->
            if re.search(r'\b(thanks|thank you|appreciate it|grateful)\b', user_input, re.I):
                message = get_thank_you_response(mode)
                yield message
                yield f"<div style='display:none;' data-tts='true'>{message}</div>"
                return

            text_accum = ""
            try:
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    temperature=0.7,
                    stream=True
                )
                for chunk in response:
                    content = getattr(chunk.choices[0].delta, "content", None)
                    if content:
                        text_accum += content
                        yield content
            except Exception as e:
                logging.error(f"Stream error: {e}")
                yield "<p>There was a problem generating a response.</p>"
                return

            yield f'<div style="display:none;" data-tts="true">{text_accum}</div>'

        return Response(generate_general(), content_type='text/html')

def fetch_step_by_step_directions(origin, destination):
    directions_url = f"https://maps.googleapis.com/maps/api/directions/json?origin={quote_plus(origin)}&destination={quote_plus(destination)}&key={GOOGLE_DIRECTIONS_API_KEY}"
    response = requests.get(directions_url)
    if response.status_code != 200:
        logging.error(f"Failed to fetch directions, status: {response.status_code}")
        return jsonify({"direct_reply": "Sorry, I couldn’t fetch detailed directions at this time."})

    data = response.json()
    if data.get("status") != "OK":
        logging.error(f"Google Directions API error: {data.get('error_message')}")
        return jsonify({"direct_reply": "Google couldn’t provide a route. Try checking your addresses."})



    leg = data["routes"][0]["legs"][0]
    summary = f"Trip from <strong>{leg['start_address']}</strong> to <strong>{leg['end_address']}</strong>. " \
              f"Total distance: {leg['distance']['text']}, estimated time: {leg['duration']['text']}."
    steps = "<ol>" + "".join(
        f"<li>{re.sub('<[^<]+?>', '', step['html_instructions'])} "
        f"({step['distance']['text']}, {step['duration']['text']})</li>"
        for step in leg["steps"]
    ) + "</ol>"
    return Response(f"<div>{summary}<br>Here are your turn-by-turn directions:{steps}</div>", content_type='text/html')

@app.route("/api/directions", methods=["POST"])
def get_directions():
    data = request.json
    origin = data.get("origin")
    destination = data.get("destination")
    if not origin or not destination:
        return jsonify({"error": "Missing origin or destination"}), 400
    directions_url = (
        f"https://maps.googleapis.com/maps/api/directions/json"
        f"?origin={requests.utils.quote(origin)}"
        f"&destination={requests.utils.quote(destination)}"
        f"&key={GOOGLE_DIRECTIONS_API_KEY}"
    )
    response = requests.get(directions_url)
    if response.status_code != 200:
        logging.error(f"Failed to fetch directions, status: {response.status_code}")
        return jsonify({"error": "Failed to fetch directions"}), 500
    directions_data = response.json()
    if directions_data.get("status") != "OK":
        logging.error(f"Google Directions API error: {directions_data.get('error_message')}")
        return jsonify({"error": directions_data.get("error_message", "Error from Google Directions API")}), 400

    route = directions_data["routes"][0]
    leg = route["legs"][0]

    def strip_html(html):
        return re.sub(r'<[^>]+>', '', html)

    steps = [{
        "instruction": strip_html(step["html_instructions"]),
        "distance": step["distance"]["text"],
        "duration": step["duration"]["text"]
    } for step in leg["steps"]]

    return jsonify({
        "start_address": leg["start_address"],
        "end_address": leg["end_address"],
        "distance": leg["distance"]["text"],
        "duration": leg["duration"]["text"],
        "steps": steps
    })

@app.route("/api/tts", methods=["POST"])
def generate_tts():
    data = request.json
    text = data.get("text", "")
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(language_code="en-US", name="en-US-Wavenet-D")
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, pitch=0.0, speaking_rate=1.0)
    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    return Response(response.audio_content, mimetype="audio/mpeg")

@app.route("/")
def index():
    return render_template("index.html")
@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route("/terms")
def terms():
    return render_template("terms.html")

if __name__ == "__main__":
    app.run(debug=True)
