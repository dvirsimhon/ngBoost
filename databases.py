import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing


class Database(object):
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    def __init__(self, x, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2)

def get_db(name):
    if name == "life expectancy":
        return create_life_expectancy_database()
    elif name == "diamonds":
        return create_diamonds_database()
    elif name == "car prices":
        return create_car_prices_database()
    elif name == "housing":
        return create_housing_database()
    elif name == "laptop":
        return create_laptop_price_db()
    elif name == "metro":
        return create_india_metro_db()
    elif name == "used_cars":
        return create_used_cars_database()
    elif name == 'airbnb':
        return create_airbnb_database()
    elif name == 'flight':
        return create_flight_database()
    elif name == "electric motor":
        return create_electric_motor_temp_database()



def create_life_expectancy_database():
    raw_data = pd.read_csv('resources/Life Expectancy Data.csv')

    # print(((raw_data.isna().sum() / 3938)*100))

    raw_data = raw_data.fillna(raw_data.mean().iloc[0])

    numeric_cols = raw_data[["Adult Mortality",	"infant deaths", "Alcohol", "percentage expenditure", "Hepatitis B", "Measles ",
                 " BMI ", "under-five deaths ", "Polio", "Total expenditure", "Diphtheria ", " HIV/AIDS", "GDP",
                 "Population", " thinness  1-19 years", " thinness 5-9 years", "Income composition of resources", "Schooling", "Life expectancy "]]

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(numeric_cols)
    df = pd.DataFrame(scaler.transform(numeric_cols), index=numeric_cols.index, columns=numeric_cols.columns)

    df["Country"] = raw_data["Country"]
    df["Year"] = raw_data["Year"]
    df["Status"] = raw_data["Status"]

    encoder = LabelEncoder()
    df['Country'] = encoder.fit_transform(df['Country'])
    df['Status'] = encoder.fit_transform(df['Status'])
    df['Year'] = encoder.fit_transform(df['Year'])

    # df = pd.get_dummies(df, columns=['Country', 'Year', 'Status'])

    y = np.array(df['Life expectancy '])

    x = df.drop(columns="Life expectancy ")

    life_expectancy_db = Database(x, y)

    return life_expectancy_db


def create_diamonds_database():
    data = pd.read_csv('resources/diamonds.csv')
    # data = data.drop(columns=['Unnamed: 0'])
    encoder = LabelEncoder()
    data['cut'] = encoder.fit_transform(data['cut'])
    data['color'] = encoder.fit_transform(data['color'])
    data['clarity'] = encoder.fit_transform(data['clarity'])
    data[['x', 'y', 'z']] = data[['x', 'y', 'z']].replace(0, np.NaN)
    data = data.dropna(how='any')
    # print(data.isnull().sum())

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(data)
    data = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)

    x = data.drop(['price'], 1)
    y = data['price']

    return Database(x, y)


def create_car_prices_database():
    raw_data = pd.read_csv('resources/car_train.csv')
    raw_data = raw_data.drop(columns=['rownum','acquisition_date','last_updated'])
    raw_data = raw_data.dropna(subset=['price', 'body_type', 'fuel', 'transmission'])
    raw_data = raw_data.dropna()

    numeric_cols = raw_data[["odometer","price"]]
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(numeric_cols)
    df = pd.DataFrame(scaler.transform(numeric_cols), index=numeric_cols.index, columns=numeric_cols.columns)

    df['badge'] = raw_data['badge']
    df['body_type'] = raw_data['body_type']
    df['category'] = raw_data['category']
    df['colour'] = raw_data['colour']
    df['cylinders'] = raw_data['cylinders']
    df['economy'] = raw_data['economy']
    df['fuel'] = raw_data['fuel']
    df['litres'] = raw_data['litres']
    df['location'] = raw_data['location']
    df['make'] = raw_data['make']
    df['model'] = raw_data['model']
    df['transmission'] = raw_data['transmission']
    df['year'] = raw_data['year']
    # df['price'] = raw_data['price']

    encoder = LabelEncoder()
    df['badge'] = encoder.fit_transform(df['badge'])
    df['body_type'] = encoder.fit_transform(df['body_type'])
    df['category'] = encoder.fit_transform(df['category'])
    df['colour'] = encoder.fit_transform(df['colour'])
    df['cylinders'] = encoder.fit_transform(df['cylinders'])
    df['economy'] = encoder.fit_transform(df['economy'])
    df['fuel'] = encoder.fit_transform(df['fuel'])
    df['litres'] = encoder.fit_transform(df['litres'])
    df['make'] = encoder.fit_transform(df['make'])
    df['model'] = encoder.fit_transform(df['model'])
    df['transmission'] = encoder.fit_transform(df['transmission'])
    df['year'] = encoder.fit_transform(df['year'])

    y = np.array(df['price'])
    x = df.drop(columns=['price'])
    car_price_db = Database(x, y)

    return car_price_db


def create_housing_database():
    data = pd.read_csv('resources/housing.csv')
    data.isnull()
    # data = data.dropna(how='any')
    # print(data.isnull().sum())

    #only bedrooms variable contains empty values, fill in with avg
    data.fillna(data.mean(), inplace=True)

    # normalize all columns to the same scale
    encoder = LabelEncoder()
    data['ocean_proximity'] = encoder.fit_transform(data['ocean_proximity'])

    scaler = preprocessing.MinMaxScaler()
    scaler.fit(data)
    data = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)

    x = data.drop(['median_house_value'], 1)
    y = data['median_house_value']

    return Database(x, y)

def create_laptop_price_db():
    raw_data = pd.read_csv('resources/laptop_price.csv', encoding = "ISO-8859-1")
    raw_data = raw_data.drop(columns=['laptop_ID'])
    raw_data['Weight'] = raw_data['Weight'].str.replace("kg", "")
    raw_data.Weight = raw_data.Weight.astype(float)

    numeric_cols = raw_data[["Weight"]]
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(numeric_cols)
    df = pd.DataFrame(scaler.transform(numeric_cols), index=numeric_cols.index, columns=numeric_cols.columns)

    df['Company'] = raw_data['Company']
    df['Product'] = raw_data['Product']
    df['TypeName'] = raw_data['TypeName']
    df['Inches'] = raw_data['Inches']
    df['ScreenResolution'] = raw_data['ScreenResolution']
    df['Cpu'] = raw_data['Cpu']
    df['Ram'] = raw_data['Ram']
    df['Memory'] = raw_data['Memory']
    df['Gpu'] = raw_data['Gpu']
    df['OpSys'] = raw_data['OpSys']
    df['Price_euros'] = raw_data['Price_euros']

    encoder = LabelEncoder()
    df['Company'] = encoder.fit_transform(df['Company'])
    df['Product'] = encoder.fit_transform(df['Product'])
    df['TypeName'] = encoder.fit_transform(df['TypeName'])
    df['Inches'] = encoder.fit_transform(df['Inches'])
    df['ScreenResolution'] = encoder.fit_transform(df['ScreenResolution'])
    df['Cpu'] = encoder.fit_transform(df['Cpu'])
    df['Ram'] = encoder.fit_transform(df['Ram'])
    df['Memory'] = encoder.fit_transform(df['Memory'])
    df['Gpu'] = encoder.fit_transform(df['Gpu'])
    df['OpSys'] = encoder.fit_transform(df['OpSys'])

    y = np.array(df['Price_euros'])
    x = df.drop(columns=['Price_euros'])
    laptop_price_db = Database(x, y)

    return laptop_price_db


def create_india_metro_db():
    raw_data = pd.read_csv('resources/india_metro.csv')
    raw_data['date_time'] = raw_data['date_time'].str[11:13] # get only hours

    numeric_cols = raw_data[["air_pollution_index", "humidity", "wind_direction", "temperature",
                             "rain_p_h", "snow_p_h", "clouds_all"]]
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(numeric_cols)
    df = pd.DataFrame(scaler.transform(numeric_cols), index=numeric_cols.index, columns=numeric_cols.columns)

    df['traffic_volume'] = raw_data['traffic_volume']
    df['is_holiday'] = raw_data['is_holiday']
    df['weather_type'] = raw_data['weather_type']
    df['weather_description'] = raw_data['weather_description']
    df['date_time'] = raw_data['date_time']

    encoder = LabelEncoder()
    df['is_holiday'] = encoder.fit_transform(df['is_holiday'])
    df['weather_type'] = encoder.fit_transform(df['weather_type'])
    df['weather_description'] = encoder.fit_transform(df['weather_description'])
    df['date_time'] = encoder.fit_transform(df['date_time'])

    y = np.array(df['traffic_volume'])
    x = df.drop(columns=['traffic_volume'])
    india_metro_db = Database(x, y)

    return india_metro_db

def create_used_cars_database():
    data = pd.read_csv('resources/used_cars.csv')
    data = data.replace('null', np.nan, regex=True)
    # print(data.isna().sum())
    data = data.drop(['Unnamed: 0', 'Name', 'New_Price'], 1)
    data = data.dropna()
    cat_features = ['Location', 'Year', 'Fuel_Type', 'Transmission', 'Owner_Type']
    encoder = LabelEncoder()
    for feature in cat_features:
        data[feature] = encoder.fit_transform(data[feature])

    data['Power'] = data['Power'].str.replace(' bhp', '')
    data['Mileage'] = data['Mileage'].str.replace(' km/kg', '')
    data['Mileage'] = data['Mileage'].str.replace(' kmpl', '')
    data['Engine'] = data['Engine'].str.replace(' CC', '')
    data['Power'] = pd.to_numeric(data['Power'])
    data['Mileage'] = pd.to_numeric(data['Mileage'])
    data['Engine'] = pd.to_numeric(data['Engine'])

    dbclass = data['Price']
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(data)
    data = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)
    data['Price'] = dbclass

    x = data.drop(['Price'], 1)
    y = np.array(data['Price'])
    return Database(x, y)


def create_airbnb_database():
    data = pd.read_csv('resources/airbnb_train.csv')
    data_target = np.array(pd.read_csv('resources/airbnb_y_train.csv', header=None))
    data['price'] = data_target
    # print(data.isna().sum())
    dbclass = data['price']
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(data)
    data = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)
    data['price'] = dbclass

    x = data.drop(['price'], 1)
    y = np.array(data['price'])
    return Database(x, y)


def create_flight_database():
    data = pd.read_csv('resources/flight_fares.csv')
    print(data.isna().sum())
    data = data.dropna(how='any')

    # print(data.dtypes)

    data['Arrival_Time'] = data['Arrival_Time'].map(lambda x: x[0:2])
    data['Dep_Time'] = data['Dep_Time'].map(lambda x: x[0:2])
    data['Airport_depart'] = data['Route'].map(lambda x: x.split(' ')[0])
    data['Airport_arrive'] = data['Route'].map(lambda x: x.split(' ')[-1])
    data['Date_of_Journey'] = data['Date_of_Journey'].map(lambda x: x.split('/')[1])
    data['Duration'] = data['Duration'].map(lambda x: hours_to_minutes(x))
    data = data.drop(['Route', 'Additional_Info'], 1)

    cat_features = ['Airline', 'Date_of_Journey', 'Source', 'Destination', 'Total_Stops', 'Airport_depart',
                    'Airport_arrive']
    encoder = LabelEncoder()
    for feature in cat_features:
        data[feature] = encoder.fit_transform(data[feature])

    dbclass = data['Price']
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(data)
    data = pd.DataFrame(scaler.transform(data), index=data.index, columns=data.columns)
    data['Price'] = dbclass

    x = data.drop(['Price'], 1)
    y = np.array(data['Price'])
    return Database(x, y)


def hours_to_minutes(x):
    if len(x.split(' ')) > 1:
        return int(x.split('h')[0]) * 60 + int(x.split(' ')[1][0:-1])
    elif 'h' in x:
        return int(x.split('h')[0]) * 60
    else:
        return int(x.split('m')[0])


def create_electric_motor_temp_database():
    raw_data = pd.read_csv('resources/pmsm_temperature_data.csv')
    raw_data_4 = raw_data.loc[raw_data['profile_id'] == 4]
    y = raw_data_4['pm']
    x = raw_data_4.drop(columns=["pm","profile_id","stator_yoke","stator_tooth","stator_winding","torque"])
    return Database(x, y)
