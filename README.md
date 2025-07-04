#AQWAPOD
## Analysis of Quality of Water And Prediction Of Diseases

### Problem Statement
- Analyze quality of water using input parameters such as pH and location, leveraging datasets and a Machine Learning model to detect anomalies. Indicate potential health risks and water potability, reducing the need for manual testing and helping prevent waterborne diseases.

### Aim
- Develop software for quick and accurate water quality testing.

- Allow users to input parameters like pH and turbidity.

- Use machine learning to analyze data, detect contamination, and assess safety.

- Provide early detection to reduce waterborne diseases, especially in underserved areas

### Outcomes
- Accurate Water Potability Prediction – Successfully predicts whether water is safe to drink using real-world datasets and ML models.
- Customizable Model Comparison – Enables testing of different machine learning models for performance evaluation.
- User-Friendly Interface – Provides a clean, interactive UI for users to input data and view predictions instantly.
- Real-Time Analysis & Export – Offers real-time predictions with options to download results for further use or reporting.

### How to read:
- The 'Python Scripts' folder contains all the necessary scripts that run as the backend of the website.
- The 'Data' section contains data such as water quality datasets, presentations, etc.
- The 'Frontend' folder contains the code for the frontend of the website, in React.
- The Research folder contains the research papers referred while creating this project.

### Conclusion:
- The biggest hurdle in creating AQWAPOD has been the unavailability of accurate datasets.
- AQWAPOD currently leverages on Large Language Models such as Gemini to provide correct predictions of potability. This is possible due to a large amount of data used to train the LLM, along with
  the fact that these LLMs perform simple checks onto the level of the parameters, to see whether any one of them has exceeded its required value. These simple checks are far more easier to do than complex
  machine learning models, which result in higher accuracy.
- Further progress can be made by training the LLM on accurate datasets to create a requirement-specific AI that uses the information it is trained on to predict water quality and contamination chances.
