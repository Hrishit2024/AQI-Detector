# 📥 How to Download Delhi NCR AQI Data from Kaggle

## Method 1: Manual Download (Recommended - Easiest!)

### Step 1: Create Kaggle Account
1. Go to https://www.kaggle.com/
2. Sign up for free using Google/Email
3. Verify your email

### Step 2: Download Dataset

**Option A: Air Quality Data in India (Recommended)**
1. Visit: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
2. Click the **"Download"** button (top right)
3. Extract the ZIP file
4. Look for files related to Delhi/NCR
5. Copy the CSV file to your project's `data/` folder
6. Rename it to `aqi_data.csv`

**Option B: Delhi Air Quality Dataset**
1. Visit: https://www.kaggle.com/datasets/apurv19/delhi-air-quality-dataset
2. Click **"Download"**
3. Extract the ZIP file
4. Copy `delhi_aqi.csv` to your project's `data/` folder
5. Rename to `aqi_data.csv`

### Step 3: Verify Data
```bash
# Check if file exists
ls -lh data/aqi_data.csv

# Preview the data
head -n 5 data/aqi_data.csv
```

---

## Method 2: Kaggle API (Advanced)

### Step 1: Get API Credentials
1. Log in to Kaggle.com
2. Click your profile picture (top right) → **"Settings"**
3. Scroll to **"API"** section
4. Click **"Create New API Token"**
5. A file `kaggle.json` will download

### Step 2: Install Kaggle API
```bash
pip install kaggle
```

### Step 3: Setup Credentials
```bash
# Create kaggle directory
mkdir -p ~/.kaggle

# Move the downloaded kaggle.json
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Step 4: Download Dataset
```bash
# Run our download script
python download_kaggle_data.py
```

OR manually:
```bash
# Download Air Quality Data in India
kaggle datasets download -d rohanrao/air-quality-data-in-india -p data/

# Extract
unzip data/air-quality-data-in-india.zip -d data/

# OR Download Delhi specific dataset
kaggle datasets download -d apurv19/delhi-air-quality-dataset -p data/
unzip data/delhi-air-quality-dataset.zip -d data/
```

---

## 📊 Expected Data Format

Your `aqi_data.csv` should have these columns:
- `datetime` or `date` - Timestamp
- `PM2.5` - Particulate Matter 2.5
- `PM10` - Particulate Matter 10
- `NO2` - Nitrogen Dioxide
- `SO2` - Sulfur Dioxide
- `CO` - Carbon Monoxide
- `O3` - Ozone
- `AQI` - Air Quality Index (target variable)

**Example:**
```csv
datetime,PM2.5,PM10,NO2,SO2,CO,O3,AQI
2020-01-01 00:00:00,45.3,85.2,32.1,15.4,0.8,42.3,87
2020-01-01 01:00:00,48.7,89.5,35.6,16.2,0.9,45.1,92
...
```

---

## 🔄 After Downloading Data

### 1. Update main_train.py (if needed)
The current code automatically detects and loads `data/aqi_data.csv`. If your data has different column names, update this section in `src/preprocessing.py`:

```python
# Line ~50 in src/preprocessing.py
def load_data(self, filepath):
    # Adjust column names if needed
    df = pd.read_csv(filepath)
    
    # Rename columns to match our expected format
    column_mapping = {
        'PM2.5': 'PM2_5',  # if your data has PM2.5 instead of PM2_5
        'Date': 'datetime',  # if your data has Date instead of datetime
        # Add more mappings as needed
    }
    df = df.rename(columns=column_mapping)
    return df
```

### 2. Filter for Delhi NCR (if dataset has multiple cities)
Add this to `src/preprocessing.py` after loading data:

```python
def load_data(self, filepath):
    df = pd.read_csv(filepath)
    
    # Filter for Delhi NCR if 'city' column exists
    if 'city' in df.columns:
        delhi_cities = ['Delhi', 'NCR', 'New Delhi', 'Gurgaon', 'Noida', 'Faridabad']
        df = df[df['city'].isin(delhi_cities)]
        print(f"Filtered for Delhi NCR: {len(df)} rows")
    
    return df
```

### 3. Re-train with Real Data
```bash
# Train models with real Kaggle data
python main_train.py
```

### 4. Launch Dashboard
```bash
streamlit run app.py
```

---

## ✅ Quick Checklist

- [ ] Kaggle account created
- [ ] Dataset downloaded (Option A or B)
- [ ] File moved to `data/aqi_data.csv`
- [ ] Verified data has required columns
- [ ] Filtered for Delhi NCR (if needed)
- [ ] Re-trained models: `python main_train.py`
- [ ] Dashboard tested: `streamlit run app.py`

---

## 🚨 Troubleshooting

**Problem: "File not found" error**
- Check file path: `ls data/aqi_data.csv`
- Ensure file is in correct location
- Check spelling (case-sensitive)

**Problem: "Missing columns" error**
- Open CSV in Excel/text editor
- Check column names
- Update column mapping in preprocessing.py

**Problem: "Too few samples" error**
- Ensure you have at least 1000 rows
- Check if filtering removed too much data
- Try a different Kaggle dataset

**Problem: "DateTime parsing error"**
- Check date format in CSV
- Update `pd.to_datetime()` format parameter
- See line ~80 in `src/preprocessing.py`

---

## 📞 Need Help?

**Recommended Datasets:**
1. **Best Choice:** https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
   - Has Delhi data
   - Well maintained
   - 10K+ downloads

2. **Alternative:** https://www.kaggle.com/datasets/apurv19/delhi-air-quality-dataset
   - Delhi specific
   - Hourly data
   - Clean format

**What to do if dataset is different:**
1. Check column names
2. Update column mapping in `src/preprocessing.py`
3. Ensure you have: PM2.5, PM10, NO2, SO2, CO, O3, AQI
4. Run `python main_train.py` to test

---

**You're ready! Download the data and run your project! 🚀**
