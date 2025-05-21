import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Initialize FastAPI app
app = FastAPI()

# Load preprocessed data
df = pd.read_excel("New Microsoft Excel Worksheet.xlsx", sheet_name="Sheet1")
district_to_state = {
    'Angul': 'Odisha',
    'Cuttack': 'Odisha',
    'Kanpur Dehat': 'Uttar Pradesh',
    'Kalimpong': 'West Bengal',
    'Muktsar': 'Punjab',
    'Mumbai': 'Maharashtra',
    'Kollam': 'Kerala',
    'Shimla': 'Himachal Pradesh',
    'Mewat': 'Haryana',
    'Badgam': 'Jammu and Kashmir',
    'Neemuch': 'Madhya Pradesh',
    'Umariya': 'Madhya Pradesh',
    'Kurnool': 'Andhra Pradesh',
    'Akola': 'Maharashtra',
    'Jalgaon': 'Maharashtra',
    'Amreli': 'Gujarat',
    'Kachchh': 'Gujarat',
    'Ashoknagar': 'Madhya Pradesh',
    'Ujjain': 'Madhya Pradesh',
    'Jaipur Rural': 'Rajasthan',
    'Koria': 'Chhattisgarh',
    'Chittorgarh': 'Rajasthan',
    'Jamnagar': 'Gujarat',
    'Bhilwara': 'Rajasthan',
    'Bhopal': 'Madhya Pradesh',
    'Bhiwani': 'Haryana',
    'Patan': 'Gujarat',
    'Mandsaur': 'Madhya Pradesh',
    'Amarawati': 'Maharashtra',
    'Surendranagar': 'Gujarat',
    'Datia': 'Madhya Pradesh',
    'Nagaur': 'Rajasthan',
    'Banaskantha': 'Gujarat',
    'Udaipur': 'Rajasthan',
    'Guna': 'Madhya Pradesh',
    'Rayagada': 'Odisha',
    'Morbi': 'Gujarat',
    'Hyderabad': 'Telangana',
    'Jaipur': 'Rajasthan',
    'Devbhumi Dwarka': 'Gujarat',
    'Ratlam': 'Madhya Pradesh',
    'Jodhpur': 'Rajasthan',
    'Mehsana': 'Gujarat',
    'Katni': 'Madhya Pradesh',
    'Buldhana': 'Maharashtra',
    'Anupur': 'Madhya Pradesh',
    'Kurukshetra': 'Haryana',
    'Dausa': 'Rajasthan',
    'Chattrapati Sambhajinagar': 'Maharashtra',
    'Ajmer': 'Rajasthan',
    'Tonk': 'Rajasthan',
    'Ahmedabad': 'Gujarat',
    'Ranga Reddy': 'Telangana',
    'Kheda': 'Gujarat',
    'Nandurbar': 'Maharashtra',
    'Surat': 'Gujarat',
    'Jodhpur Rural': 'Rajasthan',
    'Shivpuri': 'Madhya Pradesh',
    'Porbandar': 'Gujarat',
    'Pratapgarh': 'Rajasthan',
    'Kota': 'Rajasthan',
    'Medak': 'Telangana',
    'Indore': 'Madhya Pradesh',
    'Raipur': 'Chhattisgarh',
    'Panna': 'Madhya Pradesh',
    'Satna': 'Madhya Pradesh',
    'Kanker': 'Chhattisgarh',
    'Dharwad': 'Karnataka',
    'Hassan': 'Karnataka',
    'Bagalkot': 'Karnataka',
    'Bangalore': 'Karnataka',
    'Kotputli- Behror': 'Rajasthan',
    'Bellary': 'Karnataka',
    'Shehdol': 'Madhya Pradesh',
    'Betul': 'Madhya Pradesh',
    'Chhatarpur': 'Madhya Pradesh',
    'Rewa': 'Madhya Pradesh',
    'Chamrajnagar': 'Karnataka',
    'Kolar': 'Karnataka',
    'Chitradurga': 'Karnataka',
    'Davangere': 'Karnataka',
    'Koppal': 'Karnataka',
    'Tumkur': 'Karnataka',
    'Karwar(Uttar Kannad)': 'Karnataka',
    'Jhabua': 'Madhya Pradesh',
    'Mysore': 'Karnataka',
    'Chikmagalur': 'Karnataka',
    'Kalburgi': 'Karnataka',
    'Karimnagar': 'Telangana',
    'Khargone': 'Madhya Pradesh',
    'Seoni': 'Madhya Pradesh',
    'Sagar': 'Madhya Pradesh',
    'Khammam': 'Telangana',
    'Raigarh': 'Chhattisgarh',
    'Bilaspur': 'Chhattisgarh',
    'Gadag': 'Karnataka',
    'Coimbatore': 'Tamil Nadu',
    'Mahbubnagar': 'Telangana',
    'Mandla': 'Madhya Pradesh',
    'Mandya': 'Karnataka',
    'Mangalore(Dakshin Kannad)': 'Karnataka',
    'Rohtak': 'Haryana',
    'Sehore': 'Madhya Pradesh',
    'Belgaum': 'Karnataka',
    'Raichur': 'Karnataka',
    'Haveri': 'Karnataka',
    'Rajgarh': 'Madhya Pradesh',
    'Dindori': 'Madhya Pradesh',
    'Raisen': 'Madhya Pradesh',
    'Surajpur': 'Chhattisgarh',
    'Balaghat': 'Madhya Pradesh',
    'Gir Somnath': 'Gujarat',
    'Rampur': 'Uttar Pradesh',
    'Yadgiri': 'Karnataka',
    'Amritsar': 'Punjab',
    'Jhajar': 'Haryana',
    'Faridabad': 'Haryana',
    'Ferozpur': 'Punjab',
    'UdhamSinghNagar': 'Uttarakhand',
    'Khandwa': 'Madhya Pradesh',
    'Adilabad': 'Telangana',
    'Ernakulam': 'Kerala',
    'Patiala': 'Punjab',
    'Ludhiana': 'Punjab',
    'Shimoga': 'Karnataka',
    'Udupi': 'Karnataka',
    'Bastar': 'Chhattisgarh',
    'Dantewada': 'Chhattisgarh',
    'Sant Kabir Nagar': 'Uttar Pradesh',
    'Swai Madhopur': 'Rajasthan',
    'Sidhi': 'Madhya Pradesh',
    'Singroli': 'Madhya Pradesh',
    'Fazilka': 'Punjab',
    'Sangrur': 'Punjab',
    'Ambedkarnagar': 'Uttar Pradesh',
    'Ambala': 'Haryana',
    'Salem': 'Tamil Nadu',
    'Madurai': 'Tamil Nadu',
    'Bharuch': 'Gujarat',
    'Vellore': 'Tamil Nadu',
    'Ranipet': 'Tamil Nadu',
    'Dharmapuri': 'Tamil Nadu',
    'Delhi': 'Delhi',
    'Ballia': 'Uttar Pradesh',
    'Barnala': 'Punjab',
    'Gurdaspur': 'Punjab',
    'Chandrapur': 'Maharashtra',
    'Thiruvannamalai': 'Tamil Nadu',
    'Dindigul': 'Tamil Nadu',
    'Theni': 'Tamil Nadu',
    'Cuddalore': 'Tamil Nadu',
    'Sirsa': 'Haryana',
    'Hoshiarpur': 'Punjab',
    'Dehradoon': 'Uttarakhand',
    'Sivaganga': 'Tamil Nadu',
    'Gariyaband': 'Chhattisgarh',
    'Dhar': 'Madhya Pradesh',
    'Moga': 'Punjab',
    'Thirupur': 'Tamil Nadu',
    'North and Middle Andaman ': 'Andaman and Nicobar Islands',
    'Ayodhya': 'Uttar Pradesh',
    'Faridkot': 'Punjab',
    'Maharajganj': 'Uttar Pradesh',
    'Erode': 'Tamil Nadu',
    'Nanital': 'Uttarakhand',
    'Hamirpur': 'Himachal Pradesh',
    'Palwal': 'Haryana',
    'Krishnagiri': 'Tamil Nadu',
    'Amethi': 'Uttar Pradesh',
    'Madhubani': 'Bihar',
    'Chengalpattu': 'Tamil Nadu',
    'Prayagraj': 'Uttar Pradesh',
    'Villupuram': 'Tamil Nadu',
    'Kallakuruchi': 'Tamil Nadu',
    'Kancheepuram': 'Tamil Nadu',
    'Karur': 'Tamil Nadu',
    'Idukki': 'Kerala',
    'Anand': 'Gujarat',
    'Kolhapur': 'Maharashtra',
    'Malappuram': 'Kerala',
    'Koraput': 'Odisha',
    'Garhwal (Pauri)': 'Uttarakhand',
    'Namakkal': 'Tamil Nadu',
    'Thanjavur': 'Tamil Nadu',
    'Raebarelli': 'Uttar Pradesh',
    'Thiruchirappalli': 'Tamil Nadu',
    'Mohali': 'Punjab',
    'Gwalior': 'Madhya Pradesh',
    'Mandi': 'Himachal Pradesh',
    'Bhatinda': 'Punjab',
    'Thirunelveli': 'Tamil Nadu',
    'Ropar (Rupnagar)': 'Punjab',
    'Nagapattinam': 'Tamil Nadu',
    'Dhamtari': 'Chhattisgarh',
    'Nagpur': 'Maharashtra',
    'Bijnor': 'Uttar Pradesh',
    'Narayanpur': 'Chhattisgarh',
    'Jammu': 'Jammu and Kashmir',
    'Thirupathur': 'Tamil Nadu',
    'Nawanshahr': 'Punjab',
    'Pathankot': 'Punjab',
    'Perambalur': 'Tamil Nadu',
    'Pudukkottai': 'Tamil Nadu',
    'Pune': 'Maharashtra',
    'Ahmednagar': 'Maharashtra',
    'Tenkasi': 'Tamil Nadu',
    'Muzaffarnagar': 'Uttar Pradesh',
    'Sheopur': 'Madhya Pradesh',
    'Ramanathapuram': 'Tamil Nadu',
    'Thiruvarur': 'Tamil Nadu',
    'Tuticorin': 'Tamil Nadu',
    'Fatehabad': 'Haryana',
    'The Nilgiris': 'Tamil Nadu',
    'Unnao': 'Uttar Pradesh',
    'Nagercoil (Kanniyakumari)': 'Tamil Nadu',
    'Vadodara (Baroda)': 'Gujarat',
    'Kanpur': 'Uttar Pradesh',
    'Kondagaon': 'Chhattisgarh',
    'Sukma': 'Chhattisgarh',
    'Balrampur': 'Uttar Pradesh',
    'Madikeri (Kodagu)': 'Karnataka',
    "Solan":"Himachal Pradesh",
    "Banaskanth":"Gujrat",
    "Pulwama":'Jammu and Kashmir',
    "Nagercoil (Kannyiakumari)":"Tamil Nadu",
    "Srinagar":'Jammu and Kashmir',
    "Vadodara(Baroda)":"Gujrat",
    "Kangra":"Himachal Pradesh",
    "Baramulla":'Jammu and Kashmir',
    "Madikeri(Kodagu)":'Karnataka',
    "Kullu":"Himachal Pradesh",
    "Jalandhar":"Punjab",
    "Anantnag":"Jammu and Kashmir"
    
}

df['State'] = df['District Name'].map(district_to_state)
df.rename(columns={
    'District Name': "District", 
    'Market Name': "Market",
    'Min Price (Rs./Quintal)': "Min Price", 
    'Max Price (Rs./Quintal)': "Max Price",
    'Modal Price (Rs./Quintal)': "Modal Price", 
    'Price Date': "Arrival_Date", 
}, inplace=True)
df.drop(columns=["Sl no.", "Variety", "Grade"], inplace=True)
df = df.iloc[:, [2, 7, 0, 1, 6, 3, 4, 5]]

# Prepare data for modeling
date_dict = {
    1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
    7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"
}
df['Month'] = df['Arrival_Date'].apply(lambda x: date_dict[pd.to_datetime(x).month])
df['Year'] = df['Arrival_Date'].apply(lambda x: pd.to_datetime(x).year)
df['Season'] = df['Month'].apply(lambda month: {'Winter': ["January", "February"], 
    'Spring': ["March", "April"], 
    'Summer': ["May", "June"], 
    'Monsoon': ["July", "August"], 
    'Autumn': ["September", "October"], 
    'Pre-winter': ["November", "December"]
}[next(k for k, v in date_dict.items() if month in v)])

df['Day'] = df['Arrival_Date'].apply(lambda x: pd.to_datetime(x).dayofweek)

# Label encode categorical variables
label_encoders = {}
categorical_columns = ['Commodity', 'State', 'District', 'Market', 'Arrival_Date', "Month", "Season"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Prepare the input features and target variable
X = df.drop(columns=['Modal Price', "Arrival_Date", "Min Price", "Max Price"])
y = df['Modal Price']

# Scaling the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the model class
class PricePredictionModel(nn.Module):
    def __init__(self):
        super(PricePredictionModel, self).__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.output(x)
        return x

# Load the trained model
model = PricePredictionModel()
model.load_state_dict(torch.load('price_prediction_model2.pth'))
model.eval()

# Define input data structure for the API
class PricePredictionInput(BaseModel):
    commodity: str
    state: str
    district: str
    market: str
    date: str  # Date in format YYYY-MM-DD

# Define FastAPI POST route for price prediction
@app.post("/predict")
async def predict_price(input_data: PricePredictionInput):
    # Preprocess input data
    commodity = label_encoders['Commodity'].transform([input_data.commodity])[0]
    state = label_encoders['State'].transform([input_data.state])[0]
    district = label_encoders['District'].transform([input_data.district])[0]
    market = label_encoders['Market'].transform([input_data.market])[0]
    
    # Parse date
    date_parts = input_data.date.split("-")
    year = int(date_parts[0])
    month_num = int(date_parts[1])
    day = int(date_parts[2])
    
    # Get the season based on the month
    month = date_dict[month_num]
    season = next(s for s, months in {
        "Winter": ["January", "February"],
        "Spring": ["March", "April"],
        "Summer": ["May", "June"],
        "Monsoon": ["July", "August"],
        "Autumn": ["September", "October"],
        "Pre-winter": ["November", "December"]
    }.items() if month in months)

    # Convert inputs to the format required by the model
    features = np.array([commodity, state, district, market, label_encoders["Month"].transform([month])[0], year, label_encoders["Season"].transform([season])[0], day])
    scaled_features = scaler.transform([features])

    # Convert to tensor and make the prediction
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    with torch.no_grad():
        prediction = model(input_tensor).item()
    
    return {"predicted_price": prediction}

# To run the API, use the following command:
# uvicorn script_name:app --reload
