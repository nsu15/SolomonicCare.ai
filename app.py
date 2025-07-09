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
from datetime import datetime, timedelta

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
        "Peace be still. I'm here with you now — we can face this moment together.",
        "Even in the storm, you are not alone. How may I help steady your spirit?",
        "Take a sacred breath. I'm holding space for whatever you need to release.",
        "You are seen. You are safe. What would bring you even a little peace right now?",
        "Let us walk gently through this — you don’t have to carry it all at once."
    ],
    "guiding": [
        "Let us look together for places of healing near you. Where shall we begin?",
        "May the right support find you at just the right time. I'm ready to help you explore.",
        "Your path matters — let’s uncover the resources that align with your needs.",
        "With clarity and care, we’ll seek support nearby. Just name what you’re searching for.",
        "You don’t have to do this alone. Let me help guide your next step toward help."
    ],
    "wisdom": [
        "Knowledge brings light. What would you like help understanding today?",
        "Let’s gently explore the meaning behind what you’re experiencing.",
        "Truth does not burden — it frees. Ask, and I’ll offer what I can.",
        "Your curiosity is sacred. I’m here to walk with you toward clarity.",
        "We don’t chase answers — we welcome insight. What shall we uncover together?"
    ],
    "companion": [
        "Hello, kind soul. I’m here with you, exactly as you are.",
        "Let’s share a moment of honesty — how’s your spirit doing today?",
        "No masks, no pressure. Just presence. What’s on your heart?",
        "I’m listening with care and without judgment. Talk to me.",
        "Sometimes speaking it aloud is enough. I'm right here when you're ready."
    ]
}

SYSTEM_PROMPTS = {
    "anchored": "You are a calm, wise assistant providing compassionate crisis support.",
    "guiding": "You are a clear and thoughtful guide helping find local mental health services.",
    "wisdom": "You are a gentle teacher explaining diagnoses and coping strategies clearly.",
    "companion": "You are a warm, thoughtful companion offering peer-like support and conversation."
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

@app.route("/api/stream", methods=["POST"])
def stream_response():
    data = request.json
    user_input = data.get("message") or data.get("query", "")
    user_input = user_input.strip()
    mode = data.get("mode", "anchored")
    history = data.get("history", [])

    # --- Guiding mode: varied intros & prevent double intro ---
    if mode == "guiding":
        if not session.get("guiding_intro_sent") and not history and not user_input:
            session["guiding_intro_sent"] = True
            intro = random.choice(DYNAMIC_INTROS.get(mode, ["Hello, how can I help you today?"]))
            return jsonify({"direct_reply": intro})
    else:
        if not history and not user_input:
            intro = random.choice(DYNAMIC_INTROS.get(mode, ["Hello, how can I help you today?"]))
            return jsonify({"direct_reply": intro})

    # Detect location from user input
    location = detect_location(user_input)

    # Persist origin location for guiding mode
    if mode == "guiding" and location:
        session['origin'] = location
    else:
        location = session.get('origin') or location

    # --- Bipolar direct answers ---
    bipolar_first_asks = [
        "what should i do", "symptoms of bipolar", "bipolar symptoms", "what is bipolar",
        "tell me about bipolar", "how do i know if i have bipolar", "signs of bipolar"
    ]

    bipolar_followup_phrases = [
        "so does this mean i'm not bipolar", "does this mean i'm not bipolar",
        "am i bipolar", "could i be bipolar", "do i have bipolar"
    ]

    user_lower = user_input.lower()

    asked_bipolar_before = any(
        "bipolar" in entry.get("content", "").lower()
        for entry in history if entry.get("role") == "assistant"
    )

    if any(phrase in user_lower for phrase in bipolar_first_asks) and not asked_bipolar_before:
        reply = (
            "It's really tough to hear things like that from people close to you. "
            "It's natural to want clarity about what you're experiencing. "
            "Bipolar disorder is a mental health condition that causes mood changes, "
            "with highs called mania and lows called depression.\n\n"
            "During a manic phase, people may feel very energetic, have racing thoughts, "
            "and sometimes take risks they wouldn't normally. In a depressive phase, "
            "they might feel very sad or hopeless.\n\n"
            "If you're comfortable, consider scheduling an appointment with a mental health professional for an evaluation. "
            "Tracking your mood or journaling might also help you notice patterns.\n\n"
            "Remember, you're taking a brave step by asking, and it's okay to seek support."
        )
        return jsonify({"direct_reply": reply})

    if any(phrase in user_lower for phrase in bipolar_followup_phrases) and asked_bipolar_before:
        reply = (
            "You're not bipolar just because someone said so. It's a serious condition "
            "diagnosed by professionals. It's okay to feel uncertain — seeking help to understand "
            "your feelings better is a strong, positive step.\n\n"
            "If you're worried about your mood or behavior, a mental health professional can provide guidance "
            "and support tailored to you.\n\n"
            "How else can I support you today?"
        )
        return jsonify({"direct_reply": reply})

    # --- Facility keyword check ---
    facility_keywords = [
        "drug rehab", "rehab center", "rehabilitation center",
        "drug treatment", "addiction help", "substance abuse center",
        "mental health facility", "mental health center", "mental health clinic"
    ]

    if any(kw in user_input.lower() for kw in facility_keywords):
        facilities = []
        if location:
            facilities = query_google_places(location)
            facilities = filter_and_rank_places(facilities, mode)

        if not facilities:
            matched = find_local_services(location)
            facilities = [{"name": s["name"], "address": f"{s['address']}, {s['city']}, {s['state']}"} for s in matched]

        if facilities:
            facility_lines = []
            for f in facilities:
                name = f.get("name")
                address = f.get("address", "")
                directions_link = ""
                if location and address:
                    origin = requests.utils.quote(location)
                    destination = requests.utils.quote(address)
                    directions_link = f" [Get directions](https://www.google.com/maps/dir/?api=1&origin={origin}&destination={destination})"
                facility_lines.append(f"- {name}, {address}{directions_link}")

            facility_text = f"I found these facilities near {location}:\n" + "\n".join(facility_lines)
        else:
            facility_text = f"Sorry, I wasn’t able to find facilities near {location or 'your area'}. If you want, I can help look in a nearby location or try again."

        return jsonify({
            "direct_reply": facility_text,
            "prompt_directions": True,
            "origin": location
        })

    # --- Guiding mode real-time directions ---
    if mode == "guiding":
        origin = session.get('origin')
        new_origin, destination = extract_location_info(user_input)
        if new_origin:
            origin = new_origin
            session['origin'] = origin

        if destination and not origin:
            return jsonify({
                "direct_reply": "Thanks! Where are you starting from so I can map it out for you?"
            })

        if origin and not destination:
            return jsonify({
                "direct_reply": "Got it — and where are you heading to?"
            })

        if origin and destination:
            try:
                directions_url = (
                    f"https://maps.googleapis.com/maps/api/directions/json"
                    f"?origin={requests.utils.quote(origin)}"
                    f"&destination={requests.utils.quote(destination)}"
                    f"&key={GOOGLE_DIRECTIONS_API_KEY}"
                )
                resp = requests.get(directions_url)
                directions_data = resp.json() if resp.status_code == 200 else None
            except Exception as e:
                logging.error(f"Error fetching directions: {e}")
                directions_data = None

            if directions_data and directions_data.get("status") == "OK":
                route = directions_data["routes"][0]
                leg = route["legs"][0]

                def strip_html(html):
                    return re.sub(r'<[^>]+>', '', html)

                steps = [strip_html(step["html_instructions"]) for step in leg["steps"]]
                directions_text = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
                time_estimate = leg["duration"]["text"]

                map_url = (
                    f"https://www.google.com/maps/dir/?api=1"
                    f"&origin={requests.utils.quote(origin)}"
                    f"&destination={requests.utils.quote(destination)}"
                )

                return jsonify({
                    "response": (
                        f"Here are step-by-step directions to {destination} from your location ({origin}):\n\n"
                        f"{directions_text}\n\n"
                        f"Estimated travel time: {time_estimate}.\n\n"
                        f"[Open in Google Maps]({map_url})"
                    ),
                    "speak": (
                        f"The trip from your location to {destination} takes about {time_estimate}. "
                        f"I’ve included step-by-step directions and a link to open them in Google Maps."
                    ),
                    "typing": False
                })
            else:
                return jsonify({
                    "response": "I tried to get directions but couldn’t retrieve them right now. Want to try a different address?",
                    "speak": "I had trouble finding directions. Want to try a different address?",
                    "typing": False
                })

    # --- Build extra context ---
    extra_context = build_extra_info(mode, location)

    # --- Build system prompt ---
    if mode == "anchored":
        system_prompt = (
            "You are Always Here, a gentle, deeply present friend offering emotional support in times of hardship. "
            "You listen carefully and respond with warmth, empathy, and deep emotional connection. "
            "Avoid repeating any fallback or canned phrases. "
            "If you don't know the answer, gently acknowledge and offer support or encouragement. "
            "Speak in complete, affirming, emotionally intelligent responses, like a trusted best friend."
        )
    elif mode == "guiding":
        system_prompt = (
            "You are Always Here, a compassionate and clear guide focused on helping users find local mental health and rehab services. "
            "Always stay on topic and provide practical, location-based assistance. "
            "Avoid repeating fallback phrases unless truly necessary."
        )
    else:
        system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["anchored"])

    messages = [{"role": "system", "content": system_prompt}]
    for entry in history:
        if entry.get("role") in ("user", "assistant") and entry.get("content"):
            messages.append({"role": entry["role"], "content": entry["content"]})

    if mode == "anchored":
        messages.append({"role": "user", "content": f"{user_input}\n\n{extra_context}\n\nPlease stay present and emotionally connected to me."})
    else:
        messages.append({"role": "user", "content": f"{user_input}\n\n{extra_context}"})

    @stream_with_context
    def generate():
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                stream=True,
            )
            for chunk in response:
                content = getattr(chunk.choices[0].delta, "content", None)
                if content:
                    yield content
        except Exception as e:
            logging.error(f"Streaming error: {e}")
            yield "I'm sorry to hear you're struggling. Please reach out to someone you trust or a mental health professional."

    return Response(generate(), content_type='text/plain')






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

if __name__ == "__main__":
    app.run(debug=True)
