# Campus Energy Usage Dashboard

## 📌 Objective
The objective of this project is to analyze electricity usage across multiple campus buildings, identify consumption patterns, and visualize trends using a dashboard.  
The project demonstrates data ingestion, cleaning, aggregation, OOP modeling, and visualization techniques in Python.

---

## 📌 Dataset Source
The dataset consists of **building-wise electricity consumption files** created for demonstration purposes.

Each CSV contains:
- `timestamp` – date & time of reading  
- `kwh` – energy consumed in kilowatt-hours  
- One file per building (e.g., Library.csv, Hostel.csv, Canteen.csv)

All CSV files are stored inside the `data` folder.

---

## 📌 Methodology

### **1. Data Ingestion**
- Load all CSV files from the `data` folder  
- Validate timestamps and numeric values  
- Log missing or invalid entries  
- Combine all building data into a unified DataFrame  

### **2. Data Aggregation**
- Daily usage per building  
- Weekly totals  
- Building-wise summary (total, mean, min, max consumption)  
- Peak consumption timestamp  

### **3. OOP Model**
- `MeterReading` class for individual readings  
- `Building` class for storing and analyzing readings  
- `BuildingManager` to manage multiple buildings and generate reports  

### **4. Visualization Dashboard**
The dashboard includes:
- Daily consumption trends for all buildings  
- Average weekly energy usage (bar chart)  
- Peak consumption scatter plot  

### **5. Exports**
The script generates:
- `cleaned_energy_data.csv`  
- `building_summary.csv`  
- `summary.txt` (executive summary)  
- `dashboard.png`  

---

## 📌 Key Insights

- Each building shows distinct consumption patterns (Hostel > Library > Canteen).  
- Daily usage varies by activity level and time of day.  
- Weekly averages reveal which buildings are consistently consuming more energy.  
- Peak consumption timestamps help identify high-load periods.  
- Dashboard helps visually compare buildings for better energy planning.  

---
