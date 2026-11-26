# Fraud Detection System Using LSTM 

# Introduction 
Financial fraud detection is a critical application of machine learning in the banking and digital payment 
industries. With the rapid growth of online transactions, detecting fraudulent activities in real time has 
become essential to reduce financial losses and improve security. Traditional rule-based systems often 
fail to detect complex sequential transaction patterns. To overcome this limitation, deep learning models 
such as Long Short-Term Memory (LSTM) networks are widely used because they can analyze 
sequential data effectively. 
This project presents a Streamlit-based fraud detection application that uses a pre-trained LSTM model 
to predict whether a sequence of recent transactions is fraudulent or legitimate. The system takes the 
last ten transaction amounts as input and provides a fraud probability score along with a clear 
classification result. 
# Methodology 
The fraud detection system follows a structured workflow consisting of data preprocessing, feature 
engineering, model prediction, and result visualization. 
First, the user inputs the last ten transaction amounts through a Streamlit interface. These transaction 
values are processed to create an additional binary feature (fraud flag). If a transaction amount exceeds 
₹220, it is assigned a flag value of 1; otherwise, it is assigned 0. This helps the model identify unusually 
high transaction amounts. 
Next, the transaction amounts and their corresponding flags are combined into a 10×2 feature matrix. 
The data is then normalized using StandardScaler to ensure consistent scaling across inputs. After 
scaling, the input is reshaped into the required three-dimensional format (1, 10, 2), which matches the 
expected input shape of the LSTM model. 
The pre-trained LSTM model (lstm_fraud_model.keras) is loaded and used to predict the probability of 
fraud. A threshold value of 0.5 is applied to classify the transaction sequence: 
• If probability > 0.5 → Fraud Detected 
• If probability ≤ 0.5 → Legitimate Transaction 
Finally, the system displays the prediction result and plots a graph of transaction amounts with a 
horizontal threshold line for better visualization. 
# Output Results 
The system produces two main outputs: 
1. Fraud Probability Score – A numerical value between 0 and 1 representing the likelihood of 
fraud. 
2. Classification Result – A clear message indicating either: 
o “FRAUD DETECTED” (if probability > 0.5), or 
o “LEGITIMATE TRANSACTION” (if probability ≤ 0.5). 
Additionally, a graphical representation of the last ten transactions is displayed. The graph includes a 
threshold reference line (₹220) to visually highlight abnormal transaction spikes. This improves 
interpretability and helps users understand why a transaction sequence may be flagged as suspicious. 
# Conclusion 
The LSTM-based fraud detection system successfully demonstrates how sequential deep learning 
models can be applied to financial transaction monitoring. By analysing the last ten transactions, the 
system identifies suspicious patterns and provides a real-time fraud prediction. The use of Streamlit 
makes the application interactive and user-friendly. 
Although the system works effectively as a prototype, improvements can be made for real-world 
deployment. These include using a scaler fitted on training data, incorporating additional features such 
as transaction time, location, and merchant details, and implementing continuous model evaluation for 
better accuracy.
