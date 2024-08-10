import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Read the file
df = pd.read_csv("fraud.csv")

# Drop the unwanted column
df.drop(columns=["isFlaggedFraud"], inplace=True)

# Remove outliers
def outlier(df, value):
    q1 = df[value].quantile(0.25)
    q3 = df[value].quantile(0.75)
    IQR = q3 - q1
    whisker1 = q1 - 1.5 * IQR
    whisker2 = q3 + 1.5 * IQR
    return whisker1, whisker2

columns = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
for column in columns:
    whisker1, whisker2 = outlier(df, column)
    df = df[(df[column] >= whisker1) & (df[column] <= whisker2)]

# Balance the target
df_1 = df[df['isFraud'] == 0].iloc[:2452]
df_0 = df[df['isFraud'] == 1]
df = pd.concat([df_1, df_0]).reset_index(drop=True)

# Encode categorical features
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])

# Split the data
x = df.drop(columns=["isFraud"])
y = df["isFraud"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True)

# Apply PCA
pca = PCA(n_components=4)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Train the model
clf = LogisticRegression()
clf.fit(x_train_pca, y_train)
pre = clf.predict(x_test_pca)

# Print classification report
report = classification_report(y_test, pre)
print(report)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, pre)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Streamlit app
st.title('Fraud Detection Model Results')

# Display classification report
st.subheader('Classification Report')
st.text(report)

# Display ROC curve
st.subheader('ROC Curve')
st.pyplot(plt)

# User input for new data
st.subheader('Predict Fraud for New Transaction')
step = st.number_input('Step', min_value=0)
amount = st.number_input('Amount', min_value=0.0)
oldbalanceOrg = st.number_input('Old Balance Origin', min_value=0.0)
newbalanceOrig = st.number_input('New Balance Origin', min_value=0.0)
oldbalanceDest = st.number_input('Old Balance Destination', min_value=0.0)
newbalanceDest = st.number_input('New Balance Destination', min_value=0.0)
type_ = st.selectbox('Transaction Type', options=label_encoder.inverse_transform(df['type'].unique()))

# Encode the transaction type
type_encoded = label_encoder.transform([type_])[0]

# Predict fraud
if st.button('Predict'):
    new_data = np.array([[step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, type_encoded]])
    new_data_pca = pca.transform(new_data)
    prediction = clf.predict(new_data_pca)
    result = 'Fraud' if prediction[0] == 1 else 'Not Fraud'
    st.write(f'The transaction is predicted to be: {result}')
