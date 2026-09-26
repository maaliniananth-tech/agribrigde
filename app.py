# ============================================================
# AGRIBRIDGE AI - STREAMLIT APPLICATION
# ============================================================

import os
import json
from datetime import date, timedelta

import requests
import numpy as np
import streamlit as st

from PIL import Image
from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AgriBridge AI",
    page_icon="🌱",
    layout="wide"
)


# ============================================================
# MULTILINGUAL / I18N SUPPORT
# ============================================================

# The internal crop/soil/disease keys remain in English so the
# ML model and business logic are unchanged. Only user-facing
# strings are translated.
LANGUAGES = {
    "English": "en",
    "தமிழ்": "ta",
    "हिन्दी": "hi",
    "తెలుగు": "te",
    "ಕನ್ನಡ": "kn",
    "മലയാളം": "ml",
}

TRANSLATIONS = {
    "en": {
        "app_subtitle": "AI-powered agriculture assistant",
        "navigation": "Navigation",
        "dashboard": "🏠 Dashboard",
        "crop_setup": "🌾 Crop Setup",
        "disease_scanner": "🦠 Disease Scanner",
        "weather": "🌦️ Weather",
        "smart_irrigation": "💧 Smart Irrigation",
        "recommendations": "🤖 Crop Recommendations",
        "market": "💰 Market Information",
        "smart_agriculture": "Smart Agriculture Assistant",
        "dashboard_desc": "Use AI-assisted crop disease detection, weather information, irrigation guidance and crop recommendations.",
        "crop": "Crop",
        "soil": "Soil",
        "rain_risk": "Rain Risk",
        "growth_stage": t("growth_stage_label"),
        "not_set": "Not set",
        "features": "🚀 AgriBridge Features",
        "disease_feature": t("disease_feature"),
        "weather_feature": t("weather_feature"),
        "irrigation_feature": t("irrigation_feature"),
        "enter_farm_info": t("enter_farm_info"),
        "farm_area": t("farm_area"),
        "soil_type": t("soil_type"),
        "growth_stage_label": t("growth_stage_label"),
        "irrigation_method": t("irrigation_method"),
        "save_crop": t("save_crop"),
        "crop_saved": t("crop_saved"),
        "upload_crop": t("upload_crop"),
        "uploaded_crop": "Uploaded Crop Image",
        "scan": t("scan"),
        "analyzing": t("analyzing"),
        "model_loaded": "AI model loaded successfully — {n} classes.",
        "model_failed": t("model_failed"),
        "ai_analysis": t("ai_analysis"),
        "prediction": t("prediction"),
        "confidence": t("confidence"),
        "healthy": t("healthy"),
        "disease_detected": t("disease_detected"),
        "care": t("care"),
        "prevention": t("prevention"),
        "top_predictions": t("top_predictions"),
        "coordinates": t("coordinates"),
        "latitude": t("latitude"),
        "longitude": t("longitude"),
        "get_weather": t("get_weather"),
        "loading_weather": t("loading_weather"),
        "temperature": t("temperature"),
        "humidity": t("humidity"),
        "rain": t("rain"),
        "wind": t("wind"),
        "rain_risk_heading": t("rain_risk_heading"),
        "high_rain": "🔴 High Rain Risk — {p}% maximum forecast probability",
        "moderate_rain": "🟠 Moderate Rain Risk — {p}% maximum forecast probability",
        "low_rain": "🟢 Low Rain Risk — {p}% maximum forecast probability",
        "weather_advice": t("weather_advice"),
        "forecast": t("forecast"),
        "generate_irrigation": t("generate_irrigation"),
        "irrigation_desc": t("irrigation_desc"),
        "configure_crop": t("configure_crop"),
        "rain_probability": t("rain_probability"),
        "recommendation": t("recommendation"),
        "advice": t("advice"),
        "generate_recommendations": t("generate_recommendations"),
        "recommendations_heading": t("recommendations_heading"),
        "save_details_first": t("save_details_first"),
        "generating_for": "Generating recommendations for **{crop}**",
        "commodity": t("commodity"),
        "state_optional": t("state_optional"),
        "district_optional": t("district_optional"),
        "get_market": t("get_market"),
        "fetching_market": t("fetching_market"),
        "market_found": t("market_found"),
        "no_market": t("no_market"),
        "footer": t("footer"),
        "unable_weather": "Unable to load weather: {error}",
        "prediction_failed": "Disease prediction failed: {error}",
        "unable_market": "Unable to retrieve market information: {error}",
        "language": "🌐 Language",
        "select_language": "Choose your language",
        "current_crop_info": "Crop: **{crop}** | Soil: **{soil}** | Growth: **{growth}** | Irrigation: **{irrigation}**",
        "crop_image_help": "Supported formats: JPG, JPEG, PNG, WEBP",
        "language_note": "Language changes apply immediately to the interface.",
    },
    "ta": {
        "app_subtitle": "AI மூலம் இயக்கப்படும் வேளாண்மை உதவியாளர்",
        "navigation": "வழிசெலுத்தல்",
        "dashboard": "🏠 முகப்பு",
        "crop_setup": "🌾 பயிர் அமைப்பு",
        "disease_scanner": "🦠 நோய் கண்டறிதல்",
        "weather": "🌦️ வானிலை",
        "smart_irrigation": "💧 ஸ்மார்ட் பாசனம்",
        "recommendations": "🤖 பயிர் பரிந்துரைகள்",
        "market": "💰 சந்தை தகவல்",
        "smart_agriculture": "ஸ்மார்ட் வேளாண்மை உதவியாளர்",
        "dashboard_desc": "AI உதவியுடன் பயிர் நோய் கண்டறிதல், வானிலை தகவல், பாசன வழிகாட்டுதல் மற்றும் பயிர் பரிந்துரைகளைப் பயன்படுத்துங்கள்.",
        "crop": "பயிர்", "soil": "மண்", "rain_risk": "மழை அபாயம்", "growth_stage": "வளர்ச்சி நிலை", "not_set": "அமைக்கப்படவில்லை",
        "features": "🚀 AgriBridge அம்சங்கள்",
        "disease_feature": "🦠 **நோய் கண்டறிதல்**\n\nபயிர் இலைப் படத்தைப் பதிவேற்றி, பயிற்சி பெற்ற AI மாதிரியைப் பயன்படுத்தி நோயை கண்டறியுங்கள்.",
        "weather_feature": "🌦️ **வானிலை**\n\nதற்போதைய வானிலை, மழை வாய்ப்பு மற்றும் விவசாய வழிகாட்டுதலைப் பார்க்கவும்.",
        "irrigation_feature": "💧 **ஸ்மார்ட் பாசனம்**\n\nமண், பயிர் வளர்ச்சி நிலை மற்றும் மழை வாய்ப்பைப் பயன்படுத்தி பாசன வழிகாட்டுதலை உருவாக்குங்கள்.",
        "enter_farm_info": "உங்கள் பண்ணை மற்றும் பயிர் தகவல்களை உள்ளிடுங்கள்.",
        "farm_area": "பண்ணை பரப்பளவு", "soil_type": "மண் வகை", "growth_stage_label": "வளர்ச்சி நிலை", "irrigation_method": "பாசன முறை",
        "save_crop": "💾 பயிர் விவரங்களை சேமிக்கவும்", "crop_saved": "பயிர் விவரங்கள் வெற்றிகரமாக சேமிக்கப்பட்டன!",
        "upload_crop": "📷 பயிர் படத்தைப் பதிவேற்றவும்", "uploaded_crop": "பதிவேற்றிய பயிர் படம்", "scan": "🤖 நோயைக் கண்டறியவும்",
        "analyzing": "பயிர் படத்தை ஆய்வு செய்கிறது...", "model_loaded": "AI மாதிரி வெற்றிகரமாக ஏற்றப்பட்டது — {n} வகைகள்.",
        "model_failed": "நோய் கண்டறிதல் மாதிரியை ஏற்ற முடியவில்லை.", "ai_analysis": "🤖 AI ஆய்வு", "prediction": "கணிப்பு", "confidence": "நம்பகத்தன்மை",
        "healthy": "🌱 இந்தப் படம் ஆரோக்கியமானது என மாதிரி வகைப்படுத்தியுள்ளது.", "disease_detected": "⚠️ நோய் தொடர்பான வகை கண்டறியப்பட்டுள்ளது.",
        "care": "🩺 பரிந்துரைக்கப்படும் பராமரிப்பு", "prevention": "🛡️ தடுப்பு", "top_predictions": "📊 முக்கிய கணிப்புகள்",
        "coordinates": "உங்கள் பண்ணையின் ஆயத்தொலைவுகளை உள்ளிடுங்கள்.", "latitude": "அட்சரேகை", "longitude": "தீர்க்கரேகை",
        "get_weather": "🌦️ வானிலையைப் பெறுக", "loading_weather": "நேரடி வானிலை ஏற்றப்படுகிறது...", "temperature": "🌡️ வெப்பநிலை",
        "humidity": "💧 ஈரப்பதம்", "rain": "🌧️ மழை", "wind": "🌬️ காற்று", "rain_risk_heading": "🌧️ மழை அபாயம்",
        "high_rain": "🔴 அதிக மழை அபாயம் — அதிகபட்ச முன்னறிவிப்பு வாய்ப்பு {p}%", "moderate_rain": "🟠 மிதமான மழை அபாயம் — {p}%",
        "low_rain": "🟢 குறைந்த மழை அபாயம் — {p}%", "weather_advice": "🌱 விவசாய வானிலை ஆலோசனை", "forecast": "📅 3 நாள் முன்னறிவிப்பு",
        "generate_irrigation": "💧 பாசன ஆலோசனையை உருவாக்கவும்", "irrigation_desc": "உங்கள் பயிர் தகவல் மற்றும் மழை வாய்ப்பைப் பயன்படுத்தி பாசன வழிகாட்டுதலை உருவாக்குங்கள்.",
        "configure_crop": "முதலில் உங்கள் பயிரை அமைக்கவும்.", "rain_probability": "மழை வாய்ப்பு (%)", "recommendation": "பரிந்துரை", "advice": "📋 ஆலோசனை",
        "generate_recommendations": "🌱 பரிந்துரைகளை உருவாக்கவும்", "recommendations_heading": "🌱 பரிந்துரைகள்", "save_details_first": "முதலில் பயிர் விவரங்களைச் சேமிக்கவும்.",
        "generating_for": "**{crop}** பயிருக்கான பரிந்துரைகள் உருவாக்கப்படுகின்றன", "commodity": "பொருள்", "state_optional": "மாநிலம் (விருப்பம்)",
        "district_optional": "மாவட்டம் (விருப்பம்)", "get_market": "💰 சந்தை தகவலைப் பெறுக", "fetching_market": "சந்தை தகவல் பெறப்படுகிறது...",
        "market_found": "சந்தை பதிவுகள் கிடைத்தன.", "no_market": "தேர்ந்தெடுக்கப்பட்ட தேடலுக்கு சந்தை பதிவுகள் எதுவும் கிடைக்கவில்லை.",
        "footer": "AI உதவியுடன் செயல்படும் வேளாண்மை தளம்", "unable_weather": "வானிலை ஏற்ற முடியவில்லை: {error}",
        "prediction_failed": "நோய் கணிப்பு தோல்வியடைந்தது: {error}", "unable_market": "சந்தை தகவலைப் பெற முடியவில்லை: {error}",
        "language": "🌐 மொழி", "select_language": "உங்கள் மொழியைத் தேர்ந்தெடுக்கவும்", "current_crop_info": "பயிர்: **{crop}** | மண்: **{soil}** | வளர்ச்சி: **{growth}** | பாசனம்: **{irrigation}**",
        "crop_image_help": "ஆதரிக்கப்படும் வடிவங்கள்: JPG, JPEG, PNG, WEBP", "language_note": "மொழி மாற்றம் இடைமுகத்தில் உடனடியாக அமலாகும்.",
    },
}

# Hindi/Telugu/Kannada/Malayalam are initialized with common UI fallbacks.
# Add translations incrementally without changing application logic.
COMMON_REGIONAL = {
    "hi": {
        "navigation":"नेविगेशन","language":"🌐 भाषा","select_language":"अपनी भाषा चुनें","dashboard":"🏠 डैशबोर्ड","crop_setup":"🌾 फसल सेटअप",
        "disease_scanner":"🦠 रोग स्कैनर","weather":"🌦️ मौसम","smart_irrigation":"💧 स्मार्ट सिंचाई","recommendations":"🤖 फसल सुझाव","market":"💰 बाजार जानकारी",
        "smart_agriculture":"स्मार्ट कृषि सहायक","crop":"फसल","soil":"मिट्टी","rain_risk":"बारिश का जोखिम","growth_stage":"विकास अवस्था","not_set":"सेट नहीं है",
        "save_crop":"💾 फसल विवरण सहेजें","crop_saved":"फसल विवरण सफलतापूर्वक सहेजा गया!","farm_area":"खेत का क्षेत्रफल","soil_type":"मिट्टी का प्रकार",
        "growth_stage_label":"विकास अवस्था","irrigation_method":"सिंचाई विधि","upload_crop":"📷 फसल की तस्वीर अपलोड करें","scan":"🤖 रोग स्कैन करें",
        "temperature":"🌡️ तापमान","humidity":"💧 आर्द्रता","rain":"🌧️ बारिश","wind":"🌬️ हवा","forecast":"📅 3-दिन का पूर्वानुमान",
        "get_weather":"🌦️ मौसम प्राप्त करें","rain_probability":"बारिश की संभावना (%)","recommendation":"सिफारिश","advice":"📋 सलाह",
        "commodity":"वस्तु","state_optional":"राज्य (वैकल्पिक)","district_optional":"जिला (वैकल्पिक)","get_market":"💰 बाजार जानकारी प्राप्त करें",
    },
    "te": {
        "navigation":"నావిగేషన్","language":"🌐 భాష","select_language":"మీ భాషను ఎంచుకోండి","dashboard":"🏠 డ్యాష్‌బోర్డ్","crop_setup":"🌾 పంట సెటప్",
        "disease_scanner":"🦠 వ్యాధి స్కానర్","weather":"🌦️ వాతావరణం","smart_irrigation":"💧 స్మార్ట్ నీటిపారుదల","recommendations":"🤖 పంట సిఫార్సులు","market":"💰 మార్కెట్ సమాచారం",
        "smart_agriculture":"స్మార్ట్ వ్యవసాయ సహాయకుడు","crop":"పంట","soil":"నేల","rain_risk":"వర్షపు ప్రమాదం","growth_stage":"పెరుగుదల దశ","not_set":"సెట్ చేయలేదు",
        "save_crop":"💾 పంట వివరాలను సేవ్ చేయండి","crop_saved":"పంట వివరాలు విజయవంతంగా సేవ్ చేయబడ్డాయి!","farm_area":"వ్యవసాయ విస్తీర్ణం","soil_type":"నేల రకం",
        "growth_stage_label":"పెరుగుదల దశ","irrigation_method":"నీటిపారుదల పద్ధతి","upload_crop":"📷 పంట చిత్రాన్ని అప్‌లోడ్ చేయండి","scan":"🤖 వ్యాధి స్కాన్ చేయండి",
        "temperature":"🌡️ ఉష్ణోగ్రత","humidity":"💧 తేమ","rain":"🌧️ వర్షం","wind":"🌬️ గాలి","forecast":"📅 3-రోజుల అంచనా","get_weather":"🌦️ వాతావరణం పొందండి",
        "rain_probability":"వర్షం అవకాశం (%)","recommendation":"సిఫార్సు","advice":"📋 సలహా","commodity":"వస్తువు","state_optional":"రాష్ట్రం (ఐచ్ఛికం)","district_optional":"జిల్లా (ఐచ్ఛికం)","get_market":"💰 మార్కెట్ సమాచారం పొందండి",
    },
    "kn": {
        "navigation":"ನ್ಯಾವಿಗೇಶನ್","language":"🌐 ಭಾಷೆ","select_language":"ನಿಮ್ಮ ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ","dashboard":"🏠 ಡ್ಯಾಶ್‌ಬೋರ್ಡ್","crop_setup":"🌾 ಬೆಳೆ ಸೆಟಪ್",
        "disease_scanner":"🦠 ರೋಗ ಸ್ಕ್ಯಾನರ್","weather":"🌦️ ಹವಾಮಾನ","smart_irrigation":"💧 ಸ್ಮಾರ್ಟ್ ನೀರಾವರಿ","recommendations":"🤖 ಬೆಳೆ ಶಿಫಾರಸುಗಳು","market":"💰 ಮಾರುಕಟ್ಟೆ ಮಾಹಿತಿ",
        "smart_agriculture":"ಸ್ಮಾರ್ಟ್ ಕೃಷಿ ಸಹಾಯಕ","crop":"ಬೆಳೆ","soil":"ಮಣ್ಣು","rain_risk":"ಮಳೆಯ ಅಪಾಯ","growth_stage":"ಬೆಳವಣಿಗೆಯ ಹಂತ","not_set":"ಹೊಂದಿಸಲಾಗಿಲ್ಲ",
        "save_crop":"💾 ಬೆಳೆ ವಿವರಗಳನ್ನು ಉಳಿಸಿ","crop_saved":"ಬೆಳೆ ವಿವರಗಳನ್ನು ಯಶಸ್ವಿಯಾಗಿ ಉಳಿಸಲಾಗಿದೆ!","farm_area":"ಕೃಷಿ ಪ್ರದೇಶ","soil_type":"ಮಣ್ಣಿನ ವಿಧ",
        "growth_stage_label":"ಬೆಳವಣಿಗೆಯ ಹಂತ","irrigation_method":"ನೀರಾವರಿ ವಿಧಾನ","upload_crop":"📷 ಬೆಳೆ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ","scan":"🤖 ರೋಗ ಸ್ಕ್ಯಾನ್ ಮಾಡಿ",
        "temperature":"🌡️ ತಾಪಮಾನ","humidity":"💧 ಆರ್ದ್ರತೆ","rain":"🌧️ ಮಳೆ","wind":"🌬️ ಗಾಳಿ","forecast":"📅 3 ದಿನಗಳ ಮುನ್ಸೂಚನೆ","get_weather":"🌦️ ಹವಾಮಾನ ಪಡೆಯಿರಿ",
        "rain_probability":"ಮಳೆಯ ಸಾಧ್ಯತೆ (%)","recommendation":"ಶಿಫಾರಸು","advice":"📋 ಸಲಹೆ","commodity":"ಸರಕು","state_optional":"ರಾಜ್ಯ (ಐಚ್ಛಿಕ)","district_optional":"ಜಿಲ್ಲೆ (ಐಚ್ಛಿಕ)","get_market":"💰 ಮಾರುಕಟ್ಟೆ ಮಾಹಿತಿ ಪಡೆಯಿರಿ",
    },
    "ml": {
        "navigation":"നാവിഗേഷൻ","language":"🌐 ഭാഷ","select_language":"നിങ്ങളുടെ ഭാഷ തിരഞ്ഞെടുക്കുക","dashboard":"🏠 ഡാഷ്ബോർഡ്","crop_setup":"🌾 വിള ക്രമീകരണം",
        "disease_scanner":"🦠 രോഗ സ്കാനർ","weather":"🌦️ കാലാവസ്ഥ","smart_irrigation":"💧 സ്മാർട്ട് ജലസേചനം","recommendations":"🤖 വിള ശുപാർശകൾ","market":"💰 വിപണി വിവരം",
        "smart_agriculture":"സ്മാർട്ട് കാർഷിക സഹായി","crop":"വിള","soil":"മണ്ണ്","rain_risk":"മഴ അപകടസാധ്യത","growth_stage":"വളർച്ചാ ഘട്ടം","not_set":"സജ്ജമാക്കിയിട്ടില്ല",
        "save_crop":"💾 വിള വിവരങ്ങൾ സംരക്ഷിക്കുക","crop_saved":"വിള വിവരങ്ങൾ വിജയകരമായി സംരക്ഷിച്ചു!","farm_area":"കൃഷിസ്ഥല വിസ്തീർണ്ണം","soil_type":"മണ്ണിന്റെ തരം",
        "growth_stage_label":"വളർച്ചാ ഘട്ടം","irrigation_method":"ജലസേചന രീതി","upload_crop":"📷 വിളയുടെ ചിത്രം അപ്‌ലോഡ് ചെയ്യുക","scan":"🤖 രോഗം സ്കാൻ ചെയ്യുക",
        "temperature":"🌡️ താപനില","humidity":"💧 ഈർപ്പം","rain":"🌧️ മഴ","wind":"🌬️ കാറ്റ്","forecast":"📅 3 ദിവസത്തെ പ്രവചനം","get_weather":"🌦️ കാലാവസ്ഥ നേടുക",
        "rain_probability":"മഴയ്ക്കുള്ള സാധ്യത (%)","recommendation":"ശുപാർശ","advice":"📋 ഉപദേശം","commodity":"ചരക്ക്","state_optional":"സംസ്ഥാനം (ഓപ്ഷണൽ)","district_optional":"ജില്ല (ഓപ്ഷണൽ)","get_market":"💰 വിപണി വിവരം നേടുക",
    },
}

for _code, _values in COMMON_REGIONAL.items():
    TRANSLATIONS[_code] = {**TRANSLATIONS["en"], **_values}

def t(key, **kwargs):
    """Return the selected-language UI string, falling back safely to English."""
    value = TRANSLATIONS.get(st.session_state.get("language", "en"), TRANSLATIONS["en"]).get(
        key, TRANSLATIONS["en"].get(key, key)
    )
    return value.format(**kwargs) if kwargs else value

def localized_value(value):
    """Translate common selectable values while retaining English internal keys."""
    maps = {
        "ta": {
            "Tomato":"தக்காளி","Rice":"நெல்","Wheat":"கோதுமை","Maize":"மக்காச்சோளம்","Potato":"உருளைக்கிழங்கு",
            "Onion":"வெங்காயம்","Cotton":"பருத்தி","Sugarcane":"கரும்பு","Groundnut":"நிலக்கடலை","Banana":"வாழை",
            "Sandy":"மணற்பாங்கான","Clay":"களிமண்","Loamy":"வண்டல் மண்","Seedling":"நாற்று நிலை","Vegetative":"வளர்ச்சி நிலை",
            "Flowering":"பூக்கும் நிலை","Fruiting":"காய்க்கும் நிலை","Harvest":"அறுவடை","Drip":"சொட்டு நீர்ப்பாசனம்",
            "Sprinkler":"தெளிப்பு நீர்ப்பாசனம்","Flood":"வெள்ளப் பாசனம்","High":"அதிகம்","Moderate":"மிதமான","Low":"குறைவு"
        }
    }
    return maps.get(st.session_state.get("language","en"), {}).get(value, value)




# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "agribridge_disease_model.keras"
)

CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "model",
    "class_names.json"
)


# ============================================================
# LOAD DISEASE MODEL
# ============================================================

@st.cache_resource
def load_disease_model():

    if not os.path.exists(MODEL_PATH):
        return None, [], (
            f"Model not found: {MODEL_PATH}"
        )

    if not os.path.exists(CLASS_NAMES_PATH):
        return None, [], (
            f"class_names.json not found: {CLASS_NAMES_PATH}"
        )

    try:

        model = load_model(MODEL_PATH)

        with open(
            CLASS_NAMES_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            classes = json.load(file)

        return model, classes, None

    except Exception as error:

        return None, [], str(error)


disease_model, class_names, model_error = load_disease_model()


# ============================================================
# DISEASE PREDICTION
# ============================================================

def predict_disease(image):

    if disease_model is None:

        raise RuntimeError(
            "Disease model is not loaded."
        )

    if not class_names:

        raise RuntimeError(
            "class_names.json is empty."
        )

    image = image.convert("RGB")

    # Must match training
    image = image.resize((224, 224))

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # IMPORTANT:
    # Do NOT divide by 255 here.
    #
    # Your trained model already contains:
    #
    # Rescaling(1.0 / 127.5, offset=-1)

    predictions = disease_model.predict(
        image_array,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    if predicted_index >= len(class_names):

        raise RuntimeError(
            "Model output does not match class_names.json."
        )

    predicted_class = class_names[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    top_count = min(
        5,
        len(class_names)
    )

    top_indices = np.argsort(
        predictions
    )[::-1][:top_count]

    top_predictions = []

    for index in top_indices:

        top_predictions.append({

            "class": class_names[
                int(index)
            ],

            "confidence": float(
                predictions[index]
            )

        })

    return {
        "class": predicted_class,
        "confidence": confidence,
        "top_predictions": top_predictions
    }


# ============================================================
# CROP ADVICE
# ============================================================

CROP_ADVICE = {

    "Tomato": [
        "Monitor leaves, stems, flowers and developing fruits regularly.",
        "Maintain suitable soil moisture and avoid both water stress and waterlogging.",
        "Provide adequate sunlight and good airflow around plants.",
        "Monitor regularly for fungal diseases, pests and leaf damage.",
        "Support plants properly during fruit development.",
        "Remove severely affected plant parts and maintain field sanitation."
    ],

    "Rice": [
        "Maintain appropriate field water levels according to the crop stage.",
        "Monitor for weeds, pests and disease symptoms.",
        "Maintain good drainage during periods of heavy rainfall.",
        "Monitor nutrient requirements during vegetative and reproductive stages.",
        "Check crop growth regularly for signs of stress."
    ],

    "Wheat": [
        "Maintain suitable soil moisture during important growth stages.",
        "Monitor for fungal diseases and pest activity.",
        "Avoid excessive irrigation and waterlogging.",
        "Monitor nutrient availability throughout crop development.",
        "Check the crop regularly for lodging and other stress symptoms."
    ],

    "Maize": [
        "Maintain adequate soil moisture during germination and reproductive stages.",
        "Monitor plants for pest and disease symptoms.",
        "Maintain proper field drainage during heavy rainfall.",
        "Monitor nutrient availability for healthy plant development.",
        "Check plant growth and leaf condition regularly."
    ],

    "Potato": [
        "Maintain consistent soil moisture without prolonged waterlogging.",
        "Monitor leaves and stems for disease symptoms.",
        "Maintain good soil drainage.",
        "Monitor tuber development carefully.",
        "Remove severely affected plant material when appropriate."
    ],

    "Onion": [
        "Maintain suitable soil moisture during bulb development.",
        "Avoid excessive irrigation and standing water.",
        "Monitor leaves for disease and pest symptoms.",
        "Maintain field cleanliness and good drainage.",
        "Monitor bulb development throughout the crop cycle."
    ],

    "Cotton": [
        "Monitor plants regularly for insect and disease symptoms.",
        "Maintain suitable soil moisture without waterlogging.",
        "Monitor flowering and boll development carefully.",
        "Maintain good field sanitation.",
        "Check plants regularly for growth stress."
    ],

    "Sugarcane": [
        "Maintain sufficient soil moisture during important growth stages.",
        "Monitor for pest and disease symptoms.",
        "Maintain field drainage during heavy rainfall.",
        "Monitor crop growth and nutrient condition.",
        "Check for signs of water stress regularly."
    ],

    "Groundnut": [
        "Maintain suitable soil moisture during flowering and pod development.",
        "Avoid prolonged waterlogging.",
        "Monitor leaves for disease and pest symptoms.",
        "Maintain good field drainage.",
        "Monitor pod development and overall crop health."
    ],

    "Banana": [
        "Maintain consistent soil moisture.",
        "Avoid waterlogging around the root zone.",
        "Monitor leaves and pseudostem for disease symptoms.",
        "Provide suitable support when necessary.",
        "Monitor nutrient and plant growth conditions regularly."
    ]
}


# ============================================================
# DISEASE CARE
# ============================================================

DISEASE_CARE = {

    "Tomato___Early_blight": {

        "care": [
            "Remove severely affected leaves or plant parts.",
            "Avoid unnecessary leaf wetting during irrigation.",
            "Maintain good airflow around plants.",
            "Keep the field clean of heavily affected plant material.",
            "Monitor nearby plants for new symptoms."
        ],

        "prevention": [
            "Use healthy planting material.",
            "Avoid prolonged leaf wetness.",
            "Maintain proper plant spacing.",
            "Monitor plants regularly."
        ]
    },

    "Tomato___Late_blight": {

        "care": [
            "Inspect the crop frequently for new symptoms.",
            "Remove severely affected plant material where appropriate.",
            "Improve airflow around plants.",
            "Avoid unnecessary overhead watering.",
            "Monitor weather conditions because cool and wet conditions can increase disease risk."
        ],

        "prevention": [
            "Maintain field sanitation.",
            "Avoid prolonged leaf wetness.",
            "Provide good airflow.",
            "Monitor crops carefully during wet weather."
        ]
    },

    "Healthy_Tomato": {

        "care": [
            "Continue regular crop monitoring.",
            "Maintain suitable soil moisture.",
            "Monitor leaves, stems, flowers and fruits.",
            "Continue checking for early pest and disease symptoms."
        ],

        "prevention": [
            "Maintain field sanitation.",
            "Provide adequate airflow.",
            "Monitor weather and soil conditions."
        ]
    }
}


# ============================================================
# GET DISEASE CARE
# ============================================================

def get_disease_care(disease_name):

    return DISEASE_CARE.get(

        disease_name,

        {
            "care": [
                "Monitor the affected crop carefully.",
                "Remove severely affected plant material where appropriate.",
                "Maintain good field sanitation.",
                "Monitor weather, soil moisture and nearby plants."
            ],

            "prevention": [
                "Inspect crops regularly.",
                "Maintain suitable irrigation and drainage.",
                "Keep the field clean.",
                "Seek local agricultural guidance for confirmed disease identification and treatment."
            ]
        }
    )


# ============================================================
# PRODUCTION RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    crop,
    soil,
    irrigation,
    growth_stage,
    rain
):

    advice = []

    # Crop advice

    if crop in CROP_ADVICE:

        advice.extend(
            CROP_ADVICE[crop]
        )

    else:

        advice.append(
            "Monitor the crop regularly for growth, pest, disease and water stress."
        )

    # Soil advice

    if soil == "Sandy":

        advice.append(
            "Sandy soil drains quickly. Monitor soil moisture frequently."
        )

    elif soil == "Clay":

        advice.append(
            "Clay soil retains water. Avoid excessive irrigation and monitor drainage."
        )

    elif soil == "Loamy":

        advice.append(
            "Loamy soil generally provides a good balance of drainage and moisture retention."
        )

    # Irrigation advice

    if irrigation == "Drip":

        advice.append(
            "Drip irrigation can provide water directly near the crop root zone."
        )

    elif irrigation == "Sprinkler":

        advice.append(
            "Monitor soil moisture and avoid unnecessary irrigation during wet conditions."
        )

    elif irrigation == "Flood":

        advice.append(
            "Monitor water levels carefully and avoid prolonged waterlogging where unsuitable."
        )

    # Growth stage

    if growth_stage == "Seedling":

        advice.append(
            "Give special attention to early plant establishment and soil moisture."
        )

    elif growth_stage == "Vegetative":

        advice.append(
            "Monitor leaf development, nutrients, irrigation and pest activity."
        )

    elif growth_stage == "Flowering":

        advice.append(
            "Closely monitor water stress, nutrient condition and pest activity."
        )

    elif growth_stage == "Fruiting":

        advice.append(
            "Monitor crop water requirements, fruit development and disease symptoms."
        )

    elif growth_stage == "Harvest":

        advice.append(
            "Monitor crop maturity and prepare for harvesting under suitable conditions."
        )

    # Rain

    if rain in [
        "High",
        "Heavy",
        "Rain expected"
    ]:

        advice.append(
            "HIGH RAIN RISK: Prepare drainage and avoid unnecessary irrigation."
        )

        advice.append(
            "Before rain: clear drainage pathways and check field water flow."
        )

        advice.append(
            "After rain: check for standing water, root-zone problems and disease symptoms."
        )

    elif rain == "Moderate":

        advice.append(
            "MODERATE RAIN RISK: Monitor soil moisture before irrigation."
        )

        advice.append(
            "Keep drainage pathways clear and monitor the crop after rainfall."
        )

        advice.append(
            "Avoid unnecessary irrigation when sufficient rainfall is expected."
        )

    elif rain == "Low":

        advice.append(
            "LOW RAIN RISK: Monitor soil moisture and provide irrigation when required."
        )

        advice.append(
            "Continue checking weather conditions and crop water requirements."
        )

    else:

        advice.append(
            "Rain information is unavailable. Monitor weather and soil moisture before irrigation."
        )

    return advice


# ============================================================
# SMART IRRIGATION
# ============================================================

def smart_irrigation(
    crop,
    soil,
    growth_stage,
    irrigation_method,
    rain_probability
):

    try:

        rain_probability = float(
            rain_probability
        )

    except:

        rain_probability = 0

    advice = []

    if rain_probability >= 70:

        recommendation = (
            "Delay irrigation if crop and soil conditions permit."
        )

        advice.append(
            "High probability of rain detected."
        )

        advice.append(
            "Check actual soil moisture before applying additional irrigation."
        )

        advice.append(
            "Keep drainage pathways clear."
        )

    elif rain_probability >= 40:

        recommendation = (
            "Monitor soil moisture before irrigation."
        )

        advice.append(
            "Moderate probability of rain detected."
        )

        advice.append(
            "Avoid unnecessary irrigation if sufficient rainfall occurs."
        )

    else:

        recommendation = (
            "Irrigation may be required based on soil moisture and crop condition."
        )

        advice.append(
            "Low rainfall probability detected."
        )

        advice.append(
            "Check soil moisture before irrigation."
        )

    if soil == "Sandy":

        advice.append(
            "Sandy soil loses water quickly, so monitor moisture more frequently."
        )

    elif soil == "Clay":

        advice.append(
            "Clay soil retains water for longer. Avoid excessive irrigation."
        )

    elif soil == "Loamy":

        advice.append(
            "Loamy soil generally provides balanced water retention and drainage."
        )

    if irrigation_method == "Drip":

        advice.append(
            "Drip irrigation can deliver water close to the root zone."
        )

    elif irrigation_method == "Sprinkler":

        advice.append(
            "Use sprinkler irrigation according to soil moisture and weather conditions."
        )

    elif irrigation_method == "Flood":

        advice.append(
            "Monitor field water levels carefully to reduce unnecessary waterlogging."
        )

    if growth_stage == "Flowering":

        advice.append(
            "Pay close attention to water stress during flowering."
        )

    elif growth_stage == "Fruiting":

        advice.append(
            "Monitor soil moisture carefully during fruit development."
        )

    elif growth_stage == "Seedling":

        advice.append(
            "Young plants require careful monitoring of soil moisture."
        )

    return recommendation, advice


# ============================================================
# WEATHER
# ============================================================

def get_weather(latitude, longitude):

    url = (
        "https://api.open-meteo.com/v1/forecast"
    )

    params = {

        "latitude": latitude,

        "longitude": longitude,

        "current": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "rain,"
            "wind_speed_10m,"
            "weather_code"
        ),

        "hourly": (
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation_probability,"
            "precipitation,"
            "rain,"
            "wind_speed_10m"
        ),

        "daily": (
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_probability_max,"
            "rain_sum,"
            "wind_speed_10m_max"
        ),

        "forecast_days": 3,

        "timezone": "auto"
    }

    response = requests.get(
        url,
        params=params,
        timeout=15
    )

    response.raise_for_status()

    weather_data = response.json()

    current = weather_data.get(
        "current",
        {}
    )

    daily = weather_data.get(
        "daily",
        {}
    )

    hourly = weather_data.get(
        "hourly",
        {}
    )

    probabilities = hourly.get(
        "precipitation_probability",
        []
    )

    if probabilities:

        maximum_rain_probability = max(
            probabilities
        )

    else:

        maximum_rain_probability = 0

    if maximum_rain_probability >= 70:

        rain_risk = "High"

    elif maximum_rain_probability >= 40:

        rain_risk = "Moderate"

    else:

        rain_risk = "Low"

    farming_advice = []

    if rain_risk == "High":

        farming_advice.append(
            "High rain probability: prepare drainage and avoid unnecessary irrigation."
        )

        farming_advice.append(
            "Check the crop after rainfall for standing water and disease symptoms."
        )

    elif rain_risk == "Moderate":

        farming_advice.append(
            "Moderate rain probability: monitor soil moisture before irrigation."
        )

    else:

        farming_advice.append(
            "Low rain probability: monitor soil moisture and irrigation requirements."
        )

    humidity = current.get(
        "relative_humidity_2m",
        0
    )

    if humidity >= 80:

        farming_advice.append(
            "High humidity: monitor crops carefully for disease symptoms."
        )

    wind_speed = current.get(
        "wind_speed_10m",
        0
    )

    if wind_speed >= 30:

        farming_advice.append(
            "Strong wind conditions: monitor crop stability and field conditions."
        )

    return {

        "status": "success",

        "rain_risk": rain_risk,

        "rain_probability": maximum_rain_probability,

        "farming_advice": farming_advice,

        "current": current,

        "daily": daily,

        "weather": weather_data
    }


# ============================================================
# MARKET PRICE
# ============================================================

def get_market_price(
    commodity="Tomato",
    state="",
    district=""
):

    base_url = (
        "https://agmarknet.ceda.ashoka.edu.in/api"
    )

    # Commodity list

    commodities_response = requests.get(
        f"{base_url}/commodities",
        timeout=20
    )

    commodities_response.raise_for_status()

    commodities_data = (
        commodities_response.json()
    )

    commodity_id = None

    if isinstance(
        commodities_data,
        list
    ):

        for item in commodities_data:

            item_name = str(
                item.get("commodity_name")
                or item.get("name")
                or ""
            ).lower()

            if item_name == commodity.lower():

                commodity_id = (
                    item.get("commodity_id")
                    or item.get("id")
                )

                break

    if (
        commodity.lower() == "tomato"
        and not commodity_id
    ):

        commodity_id = 78

    geography_data = {}

    try:

        geography_response = requests.get(
            f"{base_url}/geographies",
            timeout=20
        )

        if geography_response.ok:

            geography_data = (
                geography_response.json()
            )

    except Exception:

        geography_data = {}

    today = date.today()

    date_ranges = [
        30,
        180,
        365,
        730,
        1095
    ]

    records = []

    selected_range = None

    for days in date_ranges:

        start_date = (
            today - timedelta(
                days=days
            )
        )

        params = {

            "commodity": commodity_id,

            "start_date":
                start_date.isoformat(),

            "end_date":
                today.isoformat()
        }

        if state:

            params["state"] = state

        if district:

            params["district"] = district

        try:

            prices_response = requests.get(
                f"{base_url}/prices",
                params=params,
                timeout=30
            )

            if not prices_response.ok:

                continue

            prices_data = (
                prices_response.json()
            )

            if isinstance(
                prices_data,
                dict
            ):

                records = (
                    prices_data.get("data")
                    or prices_data.get("records")
                    or prices_data.get("results")
                    or []
                )

            elif isinstance(
                prices_data,
                list
            ):

                records = prices_data

            else:

                records = []

            if records:

                selected_range = days

                break

        except Exception:

            continue

    return {

        "status": "success",

        "commodity": commodity,

        "state": state,

        "district": district,

        "data_label":
            "Latest Available Market Data",

        "date_range_days":
            selected_range,

        "commodity_id":
            commodity_id,

        "records":
            records,

        "geographies":
            geography_data
    }


# ============================================================
# SESSION STATE
# ============================================================

if "crop" not in st.session_state:
    st.session_state.crop = ""

if "soil" not in st.session_state:
    st.session_state.soil = "Loamy"

if "growth_stage" not in st.session_state:
    st.session_state.growth_stage = "Seedling"

if "irrigation" not in st.session_state:
    st.session_state.irrigation = "Drip"

if "area" not in st.session_state:
    st.session_state.area = 0.0

if "rain_risk" not in st.session_state:
    st.session_state.rain_risk = "Low"

if "weather_data" not in st.session_state:
    st.session_state.weather_data = None

if "disease_result" not in st.session_state:
    st.session_state.disease_result = None

if "language" not in st.session_state:
    st.session_state.language = "en"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌱 AgriBridge AI")

language_name = st.sidebar.selectbox(
    t("language"),
    list(LANGUAGES.keys()),
    index=list(LANGUAGES.values()).index(st.session_state.language),
    key="language_selector"
)
st.session_state.language = LANGUAGES[language_name]

st.sidebar.caption(t("language_note"))

st.sidebar.markdown(t("app_subtitle"))

PAGE_KEYS = [
    ("dashboard", "🏠 Dashboard"),
    ("crop_setup", "🌾 Crop Setup"),
    ("disease_scanner", "🦠 Disease Scanner"),
    ("weather", "🌦️ Weather"),
    ("smart_irrigation", "💧 Smart Irrigation"),
    ("recommendations", "🤖 Crop Recommendations"),
    ("market", "💰 Market Information"),
]
page = st.sidebar.radio(
    t("navigation"),
    [t(key) for key, _ in PAGE_KEYS]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == t("dashboard"):

    st.title("🌱 AgriBridge AI")

    st.subheader(t("smart_agriculture"))

    st.markdown(t("dashboard_desc"))

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🌾 Crop",
            st.session_state.crop
            or t("not_set")
        )

    with col2:

        st.metric(
            "🌱 Soil",
            st.session_state.soil
        )

    with col3:

        st.metric(
            "🌦️ Rain Risk",
            st.session_state.rain_risk
        )

    with col4:

        st.metric(
            "🌿 Growth Stage",
            st.session_state.growth_stage
        )

    st.divider()

    st.subheader(t("features"))

    c1, c2, c3 = st.columns(3)

    with c1:

        st.info(
            t("disease_feature")
        )

    with c2:

        st.info(
            t("weather_feature")
        )

    with c3:

        st.info(
            t("irrigation_feature")
        )


# ============================================================
# CROP SETUP
# ============================================================

elif page == t("crop_setup"):

    st.title("🌾 Crop Setup")

    st.write(
        t("enter_farm_info")
    )

    crop_options = [
        "Tomato",
        "Rice",
        "Wheat",
        "Maize",
        "Potato",
        "Onion",
        "Cotton",
        "Sugarcane",
        "Groundnut",
        "Banana"
    ]

    current_crop = (
        st.session_state.crop
        if st.session_state.crop
        in crop_options
        else crop_options[0]
    )

    crop = st.selectbox(
        t("crop"),
        crop_options,
        index=crop_options.index(
            current_crop
        )
    )

    area = st.number_input(
        t("farm_area"),
        min_value=0.0,
        step=0.1,
        value=float(
            st.session_state.area
        )
    )

    soil = st.selectbox(
        t("soil_type"),
        [
            "Sandy",
            "Clay",
            "Loamy"
        ],
        index=[
            "Sandy",
            "Clay",
            "Loamy"
        ].index(
            st.session_state.soil
        )
    )

    growth_stage = st.selectbox(
        t("growth_stage_label"),
        [
            "Seedling",
            "Vegetative",
            "Flowering",
            "Fruiting",
            "Harvest"
        ],
        index=[
            "Seedling",
            "Vegetative",
            "Flowering",
            "Fruiting",
            "Harvest"
        ].index(
            st.session_state.growth_stage
        )
    )

    irrigation = st.selectbox(
        t("irrigation_method"),
        [
            "Drip",
            "Sprinkler",
            "Flood"
        ],
        index=[
            "Drip",
            "Sprinkler",
            "Flood"
        ].index(
            st.session_state.irrigation
        )
    )

    if st.button(
        t("save_crop"),
        type="primary"
    ):

        st.session_state.crop = crop
        st.session_state.area = area
        st.session_state.soil = soil
        st.session_state.growth_stage = growth_stage
        st.session_state.irrigation = irrigation

        st.success(
            t("crop_saved")
        )


# ============================================================
# DISEASE SCANNER
# ============================================================

elif page == t("disease_scanner"):

    st.title("🦠 AI Crop Disease Scanner")

    st.write(
        "Upload a crop leaf image and let the trained AgriBridge AI model analyze it."
    )

    if model_error:

        st.error(
            t("model_failed")
        )

        st.code(
            model_error
        )

        st.info(
            "Make sure the following files exist:\n\n"
            "model/agribridge_disease_model.keras\n\n"
            "model/class_names.json"
        )

    else:

        st.success(
            t("model_loaded", n=len(class_names))
        )

        uploaded_file = st.file_uploader(

            t("upload_crop"),

            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ]
        )

        if uploaded_file:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                image,
                caption=t("uploaded_crop"),
                use_container_width=True
            )

            if st.button(
                t("scan"),
                type="primary"
            ):

                with st.spinner(
                    t("analyzing")
                ):

                    try:

                        result = predict_disease(
                            image
                        )

                        st.session_state.disease_result = result

                    except Exception as error:

                        st.session_state.disease_result = None

                        st.error(
                            t("prediction_failed", error=error)
                        )

        result = (
            st.session_state.disease_result
        )

        if result:

            predicted_class = (
                result["class"]
            )

            confidence = (
                result["confidence"]
            )

            st.divider()

            st.subheader(
                t("ai_analysis")
            )

            result_col1, result_col2 = (
                st.columns(2)
            )

            with result_col1:

                st.metric(
                    t("prediction"),
                    predicted_class
                )

            with result_col2:

                st.metric(
                    t("confidence"),
                    f"{confidence * 100:.2f}%"
                )

            if "healthy" in (
                predicted_class.lower()
            ):

                st.success(
                    t("healthy")
                )

            else:

                st.warning(
                    t("disease_detected")
                )

            # ------------------------------------------------
            # CARE
            # ------------------------------------------------

            care_data = get_disease_care(
                predicted_class
            )

            st.subheader(
                t("care")
            )

            for item in care_data["care"]:

                st.write(
                    f"• {item}"
                )

            st.subheader(
                t("prevention")
            )

            for item in care_data["prevention"]:

                st.write(
                    f"• {item}"
                )

            # ------------------------------------------------
            # TOP PREDICTIONS
            # ------------------------------------------------

            st.subheader(
                t("top_predictions")
            )

            for item in result[
                "top_predictions"
            ]:

                label = item["class"]

                score = (
                    item["confidence"]
                    * 100
                )

                st.write(
                    f"**{label}** — {score:.2f}%"
                )

                st.progress(
                    min(
                        max(
                            item["confidence"],
                            0.0
                        ),
                        1.0
                    )
                )


# ============================================================
# WEATHER
# ============================================================

elif page == t("weather"):

    st.title("🌦️ Live Weather")

    st.write(
        t("coordinates")
    )

    col1, col2 = st.columns(2)

    with col1:

        latitude = st.number_input(
            t("latitude"),
            value=11.0168,
            format="%.6f"
        )

    with col2:

        longitude = st.number_input(
            t("longitude"),
            value=76.9558,
            format="%.6f"
        )

    if st.button(
        t("get_weather"),
        type="primary"
    ):

        with st.spinner(
            t("loading_weather")
        ):

            try:

                weather_result = get_weather(
                    latitude,
                    longitude
                )

                st.session_state.weather_data = (
                    weather_result
                )

                st.session_state.rain_risk = (
                    weather_result[
                        "rain_risk"
                    ]
                )

            except Exception as error:

                st.error(
                    t("unable_weather", error=error)
                )

    weather_result = (
        st.session_state.weather_data
    )

    if weather_result:

        current = weather_result[
            "current"
        ]

        daily = weather_result[
            "daily"
        ]

        rain_risk = weather_result[
            "rain_risk"
        ]

        rain_probability = (
            weather_result[
                "rain_probability"
            ]
        )

        st.divider()

        c1, c2, c3, c4 = st.columns(4)

        with c1:

            st.metric(
                t("temperature"),
                f"{current.get('temperature_2m', '--')} °C"
            )

        with c2:

            st.metric(
                t("humidity"),
                f"{current.get('relative_humidity_2m', '--')} %"
            )

        with c3:

            st.metric(
                t("rain"),
                f"{current.get('rain', 0)} mm"
            )

        with c4:

            st.metric(
                t("wind"),
                f"{current.get('wind_speed_10m', '--')} km/h"
            )

        st.subheader(
            t("rain_risk_heading")
        )

        if rain_risk == "High":

            st.error(
                t("high_rain", p=rain_probability)
            )

        elif rain_risk == "Moderate":

            st.warning(
                t("moderate_rain", p=rain_probability)
            )

        else:

            st.success(
                t("low_rain", p=rain_probability)
            )

        st.session_state.rain_risk = (
            rain_risk
        )

        st.subheader(
            t("weather_advice")
        )

        for advice in weather_result[
            "farming_advice"
        ]:

            st.write(
                f"• {advice}"
            )

        # 3-day forecast

        st.subheader(
            t("forecast")
        )

        forecast_rows = []

        times = daily.get(
            "time",
            []
        )

        for i in range(
            len(times)
        ):

            forecast_rows.append({

                "Date":
                    times[i],

                "Min Temperature":
                    daily.get(
                        "temperature_2m_min",
                        ["--"]
                    )[i],

                "Max Temperature":
                    daily.get(
                        "temperature_2m_max",
                        ["--"]
                    )[i],

                "Rain Probability":
                    daily.get(
                        "precipitation_probability_max",
                        ["--"]
                    )[i],

                "Rain":
                    daily.get(
                        "rain_sum",
                        ["--"]
                    )[i]
            })

        if forecast_rows:

            st.dataframe(
                forecast_rows,
                use_container_width=True
            )


# ============================================================
# SMART IRRIGATION
# ============================================================

elif page == t("smart_irrigation"):

    st.title("💧 Smart Irrigation")

    st.write(
        t("irrigation_desc")
    )

    crop = st.session_state.crop

    if not crop:

        st.warning(
            t("configure_crop")
        )

    else:

        st.info(
            t(
                "current_crop_info",
                crop=localized_value(crop),
                soil=localized_value(st.session_state.soil),
                growth=localized_value(st.session_state.growth_stage),
                irrigation=localized_value(st.session_state.irrigation),
            )
        )

        rain_probability = st.number_input(

            t("rain_probability"),

            min_value=0.0,

            max_value=100.0,

            value=0.0,

            step=1.0
        )

        if st.button(
            t("generate_irrigation"),
            type="primary"
        ):

            recommendation, advice = (
                smart_irrigation(

                    crop,

                    st.session_state.soil,

                    st.session_state.growth_stage,

                    st.session_state.irrigation,

                    rain_probability
                )
            )

            st.subheader(
                t("recommendation")
            )

            st.success(
                recommendation
            )

            st.subheader(
                t("advice")
            )

            for item in advice:

                st.write(
                    f"• {item}"
                )


# ============================================================
# CROP RECOMMENDATIONS
# ============================================================

elif page == t("recommendations"):

    st.title(
        "🤖 Smart Crop Recommendations"
    )

    if not st.session_state.crop:

        st.warning(
            t("save_details_first")
        )

    else:

        st.info(
            f"Generating recommendations for "
            f"**{st.session_state.crop}**"
        )

        if st.button(
            t("generate_recommendations"),
            type="primary"
        ):

            recommendations = (
                generate_recommendations(

                    st.session_state.crop,

                    st.session_state.soil,

                    st.session_state.irrigation,

                    st.session_state.growth_stage,

                    st.session_state.rain_risk
                )
            )

            st.subheader(
                t("recommendations_heading")
            )

            for item in recommendations:

                st.write(
                    f"• {item}"
                )


# ============================================================
# MARKET INFORMATION
# ============================================================

elif page == t("market"):

    st.title(
        "💰 Market Information"
    )

    crop_options = [
        "Tomato",
        "Rice",
        "Wheat",
        "Maize",
        "Potato",
        "Onion",
        "Cotton",
        "Sugarcane",
        "Groundnut",
        "Banana"
    ]

    commodity = st.selectbox(
        t("commodity"),
        crop_options
    )

    state = st.text_input(
        t("state_optional"),
        ""
    )

    district = st.text_input(
        t("district_optional"),
        ""
    )

    if st.button(
        t("get_market"),
        type="primary"
    ):

        with st.spinner(
            t("fetching_market")
        ):

            try:

                market_result = (
                    get_market_price(

                        commodity,

                        state,

                        district
                    )
                )

                if market_result[
                    "records"
                ]:

                    st.success(
                        t("market_found")
                    )

                    st.dataframe(
                        market_result[
                            "records"
                        ],
                        use_container_width=True
                    )

                else:

                    st.info(
                        t("no_market")
                    )

            except Exception as error:

                st.error(
                    t("unable_market", error=error)
                )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption("🌱 AgriBridge AI")

st.sidebar.caption(
    t("footer")
)
