

# 🚗 CardetectionML - Data Science Project

## Predictive Analysis & Big Data – Vehicle Detection from Live Traffic Streams

CardetectionML ist ein Python-basiertes Machine-Learning-Projekt zur automatisierten Analyse von Fahrzeugen in einem Live-YouTube-Trafficstream.  
Das System erkennt, trackt und analysiert Fahrzeuge mithilfe von Deep-Learning und Computer Vision.
![Alternativtext](SINGLEOPENCVPYTHONSCREENSHOT_edited.png|400)


---

## 🧠 Projektidee

Ziel des Projekts ist die automatische Extraktion von Verkehrsdaten aus einem 24/7 Live-Stream:

- 🚗 Fahrzeugerkennung (Object Detection)
- 📍 Objekt-Tracking über mehrere Frames
- 🎨 Farbklassifikation (in Entwicklung)
- ⚡ Geschwindigkeitsberechnung (in Entwicklung)
- 📊 Speicherung als strukturiertes Dataset (CSV)

Die Daten können später für **Big-Data-Analysen**, Verkehrsstatistiken oder Smart-City-Anwendungen genutzt werden.

---

## ⚙️ Technologie-Stack

- Python 3.10
- OpenCV (DNN + YOLOv4)
- Pandas
- Flask (Webserver)
- Pafy + youtube-dl (Stream-Handling)
- NumPy
- Matplotlib
- Imutils

---

## 🧩 Systemarchitektur

Das Projekt besteht aus drei Hauptkomponenten:

1. 🎥 **Live Stream Processing**
   - Verarbeitung eines YouTube-Livestreams
   - Frame-by-Frame Analyse mit OpenCV

2. 🧠 **AI Detection Pipeline**
   - YOLOv4 Object Detection
   - Multi-Object Tracking
   - Region-of-Interest Filterung

3. 📊 **Data Pipeline**
   - Speicherung der erkannten Fahrzeuge
   - Export als CSV-Datei
   - Analyse mit Pandas & Visualisierung

---

## 🚀 Features

- Echtzeit-Objekterkennung von Fahrzeugen
- Multi-Object Tracking mit ID-Zuweisung
- ROI-basierte Fahrzeugfilterung
- Webserver-Visualisierung via Flask
- Datenexport für Big Data Analyse

---

## 🌐 Web Interface

Das Projekt kann optional über einen Flask-Webserver ausgeführt werden:

- Live-Video Stream im Browser
- Overlay von Bounding Boxes & ROI
- Zugriff auf generierte Datensätze

---

## 📉 Aktueller Stand

✔ Fahrzeugerkennung funktioniert stabil  
✔ Tracking implementiert  
⚠️ Farb- & Geschwindigkeitsanalyse noch in Entwicklung  
⚠️ Performance abhängig von Hardware & Stream  

---

## 🧪 Methodik

Das Projekt folgt dem **CRISP-DM Modell**:

- Business Understanding: Automatisierte Verkehrsanalyse
- Data Understanding: Generierte CSV-Daten
- Data Preparation: Datenbereinigung & Export
- Modelling: Pandas-basierte Analyse
- Evaluation: Teilweise funktional, weitere Optimierung nötig
- Deployment: Optional via Azure/Flask Webserver

---

## 📊 Beispiel Output

Generierte Datensätze enthalten:

- CarID
- Speed (experimentell)
- Color (experimentell)
- Timestamp
- Day

---

## ⚠️ Herausforderungen

- Begrenzte Genauigkeit bei wechselnden Videoquellen
- Performance-Limitierungen auf schwächerer Hardware
- Pafy + youtube-dl Versionskonflikte
- Teilweise unvollständige Feature-Implementierung

---

## 🔮 Zukunftsideen

- Integration in Smart Traffic Systeme (z. B. Google Maps)
- Kennzeichenerkennung (ANPR)
- Echtzeit-Verkehrsprognosen
- Cloud Deployment (Azure / AWS)
- Optimierung der YOLO Pipeline (GPU Acceleration)

---

## 📚 Literatur & Quellen

YOLOv4 Paper: https://arxiv.org/pdf/2004.10934.pdf  
OpenCV Documentation: https://docs.opencv.org/  
Python Documentation: https://docs.python.org/3/  
Pafy Documentation: https://pythonhosted.org/pafy/

---

## 👨‍💻 Team

- Tobias Madaj  
- Rickiel Eric Sympe Nguebong  
- Al Shah Aziz  
- Darwin Hutama Manggala Putra  

---

## 🏁 Fazit

CardetectionML zeigt, wie Computer Vision und Deep Learning genutzt werden können, um aus einfachen Videostreams strukturierte Verkehrsdaten zu erzeugen.  
Trotz aktueller Limitierungen bietet das System eine solide Grundlage für zukünftige Big-Data- und Smart-City-Anwendungen.
