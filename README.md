# Heart Risk Analytics

## Abstract
Heart disease is one of the leading causes of mortality worldwide, and early detection is crucial for effective treatment and prevention.  
This project presents a **mobile-based Heart Disease Prediction System** that uses **machine learning algorithms** to assess the risk of heart disease based on medical attributes such as **age, sex, chest pain type, resting blood pressure, cholesterol level, fasting blood sugar**, and other factors.

The system is trained using a **machine learning classifier**, evaluated on key performance metrics, and then deployed via a **Flask API**, which is integrated into an **Android mobile application**.  
The mobile app allows users to input their medical data and instantly receive a **prediction result** indicating their likelihood of having heart disease.  

This project demonstrates how **machine learning** can be effectively combined with **mobile technology** to provide a **low-cost, accessible, and user-friendly healthcare tool** that enables early disease risk assessment without a hospital visit.  
In the future, the system can be extended with **IoT-based wearable devices** for real-time monitoring and **personalized recommendations** for each individual.

---

## Features
- Machine Learning-based heart disease risk prediction  
- Android mobile app with a simple user interface  
- Flask-based backend API for real-time model inference  
- Model trained and evaluated using real-world heart disease dataset  
- Lightweight and deployable model integration   

---

## Tech Stack
| Category | Technology Used |
|-----------|------------------|
| **Programming Languages** | Python, Java (Android) |
| **Machine Learning Libraries** | scikit-learn, pandas, numpy, joblib |
| **Model Used** | Decision Tree / Random Forest / XGBoost (tested for best accuracy) |
| **Backend** | Flask (REST API) |
| **Frontend** | Android Studio (Java XML UI) |
| **Integration** | Android app communicates with Flask API using HTTP POST requests |

---

## System Architecture
1. **Data Preprocessing & Model Training**
   - Loaded dataset using pandas  
   - Cleaned and normalized data  
   - Split data into training and testing sets  
   - Trained multiple ML models and compared performance  
   - Exported best-performing model using `joblib`  

2. **Backend Development**
   - Built a Flask server  to load the trained model  
   - Created an API endpoint '/predict' to receive user data  
   - Returned prediction results as JSON  

3. **Android App Integration**
   - Designed a clean user interface in Android Studio  
   - Collected user input parameters  
   - Sent data to Flask API via POST request  
   - Displayed the prediction result instantly on-screen  

---

## Future Enhancements
- Integrate wearable sensors (IoT) for real-time monitoring  
- Personalized health recommendations using deep learning  
- Deploy model and backend on cloud (AWS/GCP)  
- Add periodic health tracking and alert system  

---

## References
- Heart Disease UCI Dataset, UCI Machine Learning Repository  
- Research paper on *“Application of Machine Learning in Cardiovascular Disease Prediction”*

---


