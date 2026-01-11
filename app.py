import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

# Disable unnecessary warnings
warnings.filterwarnings("ignore")

def run_prediction_system():
    try:
        # 1. Load the Dataset
        df = pd.read_csv('diabetes.csv')
        
        # 2. Features and Target
        X = df[['Glucose', 'BMI', 'Age']]
        y = df['Outcome']
        
        # 3. Model Training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 4. Accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        
        print("="*40)
        print("   AI DISEASE PREDICTION SYSTEM   ")
        print(f"   Model Accuracy: {accuracy:.2f}%")
        print("="*40)

        # 5. User Interaction
        print("\nPlease enter patient details:")
        glucose = float(input("➤ Enter Glucose Level: "))
        bmi = float(input("➤ Enter BMI: "))
        age = float(input("➤ Enter Age: "))
        
        patient_data = pd.DataFrame([[glucose, bmi, age]], columns=['Glucose', 'BMI', 'Age'])
        prediction = model.predict(patient_data)
        
        print("\n" + "-"*40)
        print("         FINAL DIAGNOSIS          ")
        print("-"*40)
        
        if prediction[0] == 1:
            print(" RESULT: DIABETES DETECTED (POSITIVE) ⚠️")
        else:
            print(" RESULT: NO DIABETES DETECTED (NEGATIVE) ✅")
        print("-"*40)

    except FileNotFoundError:
        print("ERROR: Please make sure 'diabetes.csv' is in the same folder.")

if __name__ == "__main__":
    run_prediction_system()
    