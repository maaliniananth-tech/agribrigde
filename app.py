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


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌱 AgriBridge AI")

st.sidebar.markdown(
    "AI-powered agriculture assistant"
)

page = st.sidebar.radio(

    "Navigation",

    [
        "🏠 Dashboard",
        "🌾 Crop Setup",
        "🦠 Disease Scanner",
        "🌦️ Weather",
        "💧 Smart Irrigation",
        "🤖 Crop Recommendations",
        "💰 Market Information"
    ]

)


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    st.title("🌱 AgriBridge AI")

    st.subheader(
        "Smart Agriculture Assistant"
    )

    st.markdown(
        "Use AI-assisted crop disease detection, weather information, irrigation guidance and crop recommendations."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🌾 Crop",
            st.session_state.crop
            or "Not set"
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

    st.subheader(
        "🚀 AgriBridge Features"
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.info(
            "🦠 **Disease Scanner**\n\n"
            "Upload a crop leaf image and use your trained AI model to identify the predicted class."
        )

    with c2:

        st.info(
            "🌦️ **Weather**\n\n"
            "Check current weather, rainfall probability and farming guidance."
        )

    with c3:

        st.info(
            "💧 **Smart Irrigation**\n\n"
            "Generate irrigation guidance using soil, crop stage and rainfall probability."
        )


# ============================================================
# CROP SETUP
# ============================================================

elif page == "🌾 Crop Setup":

    st.title("🌾 Crop Setup")

    st.write(
        "Enter your farm and crop information."
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
        "Crop",
        crop_options,
        index=crop_options.index(
            current_crop
        )
    )

    # ========================================================
    # FARM AREA - SQUARE FEET
    # ========================================================

    area = st.number_input(
        "Farm Area (sq ft)",
        min_value=0.0,
        step=100.0,
        value=float(
            st.session_state.area
        )
    )

    soil = st.selectbox(
        "Soil Type",
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
        "Growth Stage",
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
        "Irrigation Method",
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
        "💾 Save Crop Details",
        type="primary"
    ):

        st.session_state.crop = crop
        st.session_state.area = area
        st.session_state.soil = soil
        st.session_state.growth_stage = growth_stage
        st.session_state.irrigation = irrigation

        st.success(
            "Crop details saved successfully!"
        )


# ============================================================
# DISEASE SCANNER
# ============================================================

elif page == "🦠 Disease Scanner":

    st.title("🦠 AI Crop Disease Scanner")

    st.write(
        "Upload a crop leaf image and let the trained AgriBridge AI model analyze it."
    )

    if model_error:

        st.error(
            "Disease model could not be loaded."
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
            f"AI model loaded successfully — {len(class_names)} classes."
        )

        uploaded_file = st.file_uploader(

            "📷 Upload crop image",

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
                caption="Uploaded Crop Image",
                use_container_width=True
            )

            if st.button(
                "🤖 Scan for Disease",
                type="primary"
            ):

                with st.spinner(
                    "Analyzing crop image..."
                ):

                    try:

                        result = predict_disease(
                            image
                        )

                        st.session_state.disease_result = result

                    except Exception as error:

                        st.session_state.disease_result = None

                        st.error(
                            f"Disease prediction failed: {error}"
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
                "🤖 AI Analysis"
            )

            result_col1, result_col2 = (
                st.columns(2)
            )

            with result_col1:

                st.metric(
                    "Prediction",
                    predicted_class
                )

            with result_col2:

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

            if "healthy" in (
                predicted_class.lower()
            ):

                st.success(
                    "🌱 The model classified this image as healthy."
                )

            else:

                st.warning(
                    "⚠️ The model detected a disease-related class."
                )

            # ------------------------------------------------
            # CARE
            # ------------------------------------------------

            care_data = get_disease_care(
                predicted_class
            )

            st.subheader(
                "🩺 Suggested Care"
            )

            for item in care_data["care"]:

                st.write(
                    f"• {item}"
                )

            st.subheader(
                "🛡️ Prevention"
            )

            for item in care_data["prevention"]:

                st.write(
                    f"• {item}"
                )

            # ------------------------------------------------
            # TOP PREDICTIONS
            # ------------------------------------------------

            st.subheader(
                "📊 Top Predictions"
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

elif page == "🌦️ Weather":

    st.title("🌦️ Live Weather")

    st.write(
        "Enter the coordinates of your farm."
    )

    col1, col2 = st.columns(2)

    with col1:

        latitude = st.number_input(
            "Latitude",
            value=11.0168,
            format="%.6f"
        )

    with col2:

        longitude = st.number_input(
            "Longitude",
            value=76.9558,
            format="%.6f"
        )

    if st.button(
        "🌦️ Get Weather",
        type="primary"
    ):

        with st.spinner(
            "Loading live weather..."
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
                    f"Unable to load weather: {error}"
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
                "🌡️ Temperature",
                f"{current.get('temperature_2m', '--')} °C"
            )

        with c2:

            st.metric(
                "💧 Humidity",
                f"{current.get('relative_humidity_2m', '--')} %"
            )

        with c3:

            st.metric(
                "🌧️ Rain",
                f"{current.get('rain', 0)} mm"
            )

        with c4:

            st.metric(
                "🌬️ Wind",
                f"{current.get('wind_speed_10m', '--')} km/h"
            )

        st.subheader(
            "🌧️ Rain Risk"
        )

        if rain_risk == "High":

            st.error(
                f"🔴 High Rain Risk — "
                f"{rain_probability}% maximum forecast probability"
            )

        elif rain_risk == "Moderate":

            st.warning(
                f"🟠 Moderate Rain Risk — "
                f"{rain_probability}% maximum forecast probability"
            )

        else:

            st.success(
                f"🟢 Low Rain Risk — "
                f"{rain_probability}% maximum forecast probability"
            )

        st.session_state.rain_risk = (
            rain_risk
        )

        st.subheader(
            "🌱 Farmer Weather Advice"
        )

        for advice in weather_result[
            "farming_advice"
        ]:

            st.write(
                f"• {advice}"
            )

        # 3-day forecast

        st.subheader(
            "📅 3-Day Forecast"
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

elif page == "💧 Smart Irrigation":

    st.title("💧 Smart Irrigation")

    st.write(
        "Generate irrigation guidance using your crop information and rainfall probability."
    )

    crop = st.session_state.crop

    if not crop:

        st.warning(
            "Please configure your crop first."
        )

    else:

        st.info(
            f"Crop: **{crop}** | "
            f"Soil: **{st.session_state.soil}** | "
            f"Growth: **{st.session_state.growth_stage}** | "
            f"Irrigation: **{st.session_state.irrigation}**"
        )

        rain_probability = st.number_input(

            "Rain Probability (%)",

            min_value=0.0,

            max_value=100.0,

            value=0.0,

            step=1.0
        )

        if st.button(
            "💧 Generate Irrigation Advice",
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
                "Recommendation"
            )

            st.success(
                recommendation
            )

            st.subheader(
                "📋 Advice"
            )

            for item in advice:

                st.write(
                    f"• {item}"
                )


# ============================================================
# CROP RECOMMENDATIONS
# ============================================================

elif page == "🤖 Crop Recommendations":

    st.title(
        "🤖 Smart Crop Recommendations"
    )

    if not st.session_state.crop:

        st.warning(
            "Please save crop details first."
        )

    else:

        st.info(
            f"Generating recommendations for "
            f"**{st.session_state.crop}**"
        )

        if st.button(
            "🌱 Generate Recommendations",
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
                "🌱 Recommendations"
            )

            for item in recommendations:

                st.write(
                    f"• {item}"
                )


# ============================================================
# MARKET INFORMATION
# ============================================================

elif page == "💰 Market Information":

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
        "Commodity",
        crop_options
    )

    state = st.text_input(
        "State (optional)",
        ""
    )

    district = st.text_input(
        "District (optional)",
        ""
    )

    if st.button(
        "💰 Get Market Information",
        type="primary"
    ):

        with st.spinner(
            "Fetching market information..."
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
                        "Market records found."
                    )

                    st.dataframe(
                        market_result[
                            "records"
                        ],
                        use_container_width=True
                    )

                else:

                    st.info(
                        "No market records were found for the selected search."
                    )

            except Exception as error:

                st.error(
                    f"Unable to retrieve market information: {error}"
                )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "🌱 AgriBridge AI"
)

st.sidebar.caption(
    "AI-assisted agriculture platform"
)
