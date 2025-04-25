# ✈️ Aircraft Intelligence Hub

The **Aircraft Intelligence Hub** is a comprehensive satellite image analysis platform designed to assist aviation professionals, analysts, and pilots in deriving real-time, actionable insights from airport and aircraft imagery. Leveraging computer vision, OCR, and geospatial data, the hub provides a suite of interactive tools for accurate flight monitoring, airport analysis, and cockpit HUD data extraction.

---

## 🔍 Features

### 1. **Aircraft and Airport Analysis Hub**
- Analyze satellite images of airports for aircraft count and location.
- Compare image sections side-by-side with detection overlays.
- Example Output: `"30 Aircrafts Detected"`

### 2. **Airport Research Tools**
- Time-based analysis of aircraft presence at specific airports.
- View trends such as:
  - Aircrafts per image
  - Airport activity over time
  - Monthly distribution plots

### 3. **Imagery Analysis Tool**
- Upload images to automatically detect aircraft.
- Choose specific airports (e.g., Amsterdam, Atlanta) for contextual analysis.
- Supports `.jpg`, `.jpeg`, `.png` (up to 200MB).

### 4. **Live Flight Tracker**
- Visualize live flights across North America with altitude heatmap.
- Filters for:
  - Time Zone (e.g., Asia/Kolkata)
  - Country (e.g., North America)
  - Data field selection (e.g., `baro_altitude`)
  - Visualization color themes: Rainbow, Ice, Hot

### 5. **HUD Optical Character Recognition (OCR) System**
- Extract vital flight details (e.g., speed, heading, altitude) from F-16 HUD video feeds.
- Supports region-specific OCR tuning for:
  - Frequency
  - Steerpoint
  - Speed
  - Altitude
  - Heading

- Additional OCR options:
  - Contrast, brightness tuning
  - ROI adjustment
  - Tesseract configuration for optimal character recognition

---

## 🚀 Tech Stack

- Python
- OpenCV
- Tesseract OCR
- Geospatial Mapping Libraries
- Satellite Imagery Tools
- Frontend: Streamlit / Dash (assumed from layout)

---
