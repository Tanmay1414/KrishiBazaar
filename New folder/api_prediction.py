import streamlit as st
import torch
import pandas as pd
import numpy as np
from torch import nn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import shutil
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
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


device="cuda" if torch.cuda.is_available() else "cpu"
df=pd.read_excel("New Microsoft Excel Worksheet.xlsx",sheet_name="Sheet1")
df['State'] = df['District Name'].map(district_to_state)
df.rename(columns={
    # 'Sl no.', 
    'District Name':"District", 
    'Market Name':"Market",
    # 'Commodity', 
    # 'Variety',
    # 'Grade',
    'Min Price (Rs./Quintal)':"Min Price", 
    'Max Price (Rs./Quintal)':"Max Price",
    'Modal Price (Rs./Quintal)':"Modal Price", 
    'Price Date':"Arrival_Date", 
    # 'State'
},inplace=True)
df.drop(columns=["Sl no.","Variety","Grade"],inplace=True)
df=df.iloc[:,[2,7,0,1,6,3,4,5]]
date_dict= {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"}

Day=[]
Month=[]
Year=[]
for d in df["Arrival_Date"]:
    d=str(d)
    date=d.split("-")
    Day.append(int(date[1]))
    Month.append(date_dict[int(date[1])])
    Year.append(int(date[0]))
    # print(date)
season=[]
for month in Month:
    if month == "January" or month=="February":
        season.append("Winter")
    elif month == "March" or month=="April":
        season.append("Spring")
    elif month == "May" or month=="June":
        season.append("Summer")
    elif month == "July" or month=="August":
        season.append("Monsoon")
    elif month=="September" or month=="October":
        season.append("Autumn")
    elif month=="November" or month=="December":
        season.append("Pre-winter")
# ! we are covering day month season and the dayta has year 

df["Month"]=Month
df["Year"]=Year
df["Season"]=season
# Check price for each day as it can vary on weekly market
day_of_week=[]
for w in df["Arrival_Date"]:
    dt=pd.Timestamp(w)
    day=dt.day_of_week
    day_of_week.append(day)
df["Day"]=day_of_week
label_encoders = {}
categorical_columns = ['Commodity', 'State', 'District', 'Market', 'Arrival_Date',"Month","Season"]

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
X = df.drop(columns=['Modal Price',"Arrival_Date","Min Price","Max Price"])
# print(X)
y = df['Modal Price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

class PricePredictionModel(nn.Module):
    def __init__(self):
        super(PricePredictionModel, self).__init__()
        # self.fc1 = nn.Linear(7, 64)
        # self.fc1 = nn.Linear(8, 512)
        # self.fc1 = nn.Linear(8, 32)
        self.fc1 = nn.Linear(8, 16)

        # self.fc2 = nn.Linear(32, 256)
        self.fc2 = nn.Linear(16, 256)
        

        # self.fc2 = nn.Linear(512, 256)
        self.fc3=nn.Linear(256,64)
        # self.conv=nn.Conv1d(in_channels=128,out_channels=64,kernel_size=2)

        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        # x=self.conv(x)
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.output(x)
        return x
date_dict= {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"}
def date(Date):
    Day=[]
    month=[]
    Year=[]
    season=[]
    # for d in user_input[4]:
    date=Date.split("-")
    print(date)
    Day=date[0]
    month=date_dict[int(date[1])]
    Year=date[2]

    if month == "January" or month=="February":
        season="Winter"
    elif month == "March" or month=="April":
        season="Spring"
    elif month == "May" or month=="June":
        season="Summer"
    elif month == "July" or month=="August":
        season="Monsoon"
    elif month=="September" or month=="October":
        season="Autumn"
    elif month=="November" or month=="December":
        season="Pre-winter"
    dt=pd.Timestamp(Date)
    Day=dt.day_of_week
    return int(month),int(Year),season,int(Day)
# Initialize the model, loss function, and optimizer
model = PricePredictionModel().to(device)
model.load_state_dict(torch.load('price_prediction_model2.pth'))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

app=FastAPI()
@app.post("/predict_price/")
def price_prediction(Commodity,State,District,Market,Date):
    month,Year,season,Day=date(Date)
    label_encoded=le.fit_transform([Commodity, State,District,Market,month,Year,season])
    label_encoded=np.append(label_encoded,Day)
    scaled=scaler.transform(label_encoded.reshape(1, -1))
    user_input_tensor=torch.tensor(scaled,dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred=model(user_input_tensor)
    return {"Prediction":pred}