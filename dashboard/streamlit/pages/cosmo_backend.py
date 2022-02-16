from sklearn.preprocessing import LabelEncoder, PowerTransformer
from catboost import CatBoostClassifier

import pandas as pd
import numpy as np

import joblib
import os

class COSMO:
    NEWTONG = 6.67e-11
    STEPBOLTS = 5.670e-8

    def __init__(self, df):
        self.raw_df = df

        self.results_df = df[['ObjectOfInterest']] 
        
        self.preprocess(df.drop(columns=['ObjectOfInterest']))

        ### Generate Discrete Results
        self.results_df['Disposition'] = self.generate_disposition()

        ### Generate Probabilistic Results
        self.dis_prob_df = pd.DataFrame(self.generate_disposition(out='prob'), index=[0], columns=['P(False Positive)', 'P(Exoplanet)'])
        self.dis_prob_df['ObjectOfInterest'] = df[['ObjectOfInterest']]
        self.dis_prob_df = self.dis_prob_df[['ObjectOfInterest', 'P(False Positive)', 'P(Exoplanet)']].set_index('ObjectOfInterest')

        if self.dis_df['Disposition'].values[0] == 'Possible Exoplanet':
            ### Generate Discrete Results
            self.results_df['PlanetType'] = self.generate_type()

            ### Generate Probabilistic Results
            self.planet_prob_df = pd.DataFrame(self.generate_type(out='prob'), columns=['P(Gas Giant)', 'P(Neptune-like)', 'P(Super Earth)', 'P(Terrestrial)'])
            self.planet_prob_df['ObjectOfInterest'] = df[['ObjectOfInterest']]
            self.planet_prob_df = self.planet_prob_df[['ObjectOfInterest', 'P(Gas Giant)', 'P(Neptune-like)', 'P(Super Earth)', 'P(Terrestrial)']].set_index('ObjectOfInterest')
        else:
            self.results_df['PlanetType'] = 'N/A'
            self.type_df['PlanetType'] = 'N/A'

            self.planet_prob_df = pd.DataFrame({'P(Gas Giant)': 0, 'P(Neptune-like)': 0, 'P(Super Earth)': 0, 'P(Terrestrial)': 0}, index=[0], columns=['P(Gas Giant)', 'P(Neptune-like)', 'P(Super Earth)', 'P(Terrestrial)'])
            self.planet_prob_df['ObjectOfInterest'] = df[['ObjectOfInterest']]
            self.planet_prob_df = self.planet_prob_df[['ObjectOfInterest', 'P(Gas Giant)', 'P(Neptune-like)', 'P(Super Earth)', 'P(Terrestrial)']].set_index('ObjectOfInterest')

        self.results_df = self.results_df.set_index('ObjectOfInterest')

    def preprocess(self, df):
        # Generate the dis_df
        dis_df = df.copy()

        dis_df = self.calculate_features(dis_df)
        dis_df = self.scale_features(dis_df, 'd')
        self.dis_df = dis_df

        # Generate the type_df
        type_df = df.copy()

        type_df = self.calculate_features(type_df)
        type_df = self.scale_features(type_df, 't')
        self.type_df = type_df

    def generate_disposition(self, out='disc'):
        # Get the model
        model = COSMO.get_COS()

        # Get the Numerics
        numerics = [col for col in self.dis_df.columns if self.dis_df[col].dtype != object]

        if out == 'disc':
            # Get the prediction
            pred = model.predict(self.dis_df[numerics])

            # Decode the Prediction
            pred = 'Possible Exoplanet' if pred == 1 else 'False Positive'

            # Set the prediction
            self.dis_df['Disposition'] = pred
        elif out == 'prob':
            # Get the prediction
            pred = model.predict_proba(self.dis_df[numerics])
        else:
            raise Exception("Out Value can only be disc (Discrete) or prob (Probability)")

        return pred

    def generate_type(self, out='disc'):
        # Get the model
        model = COSMO.get_MO()

        # Get the numerics
        numerics = [col for col in self.type_df.columns if self.type_df[col].dtype != object]

        if out == 'disc':
            # Get the Predictions
            pred = model.predict(self.type_df[numerics])

            # Decode the prediction
            pred = COSMO.type_decode(pred)

            # Set the prediction
            self.type_df['PlanetType'] = pred
        elif out == 'prob':
            # Get the prediction
            pred = model.predict_proba(self.type_df[numerics])

            print([model.classes_])
        else:
            raise Exception("Out Value can only be disc (Discrete) or prob (Probability)")

        return pred

    @staticmethod
    def get_train():
        return pd.read_csv(os.getcwd() + '\\pages\\OutlierFree.csv', index_col=0)

    def get_raw(self):
        return self.raw_df

    def get_results(self):
        return self.results_df

    def get_results_proba(self):
        if self.results_df['Disposition'].values[0] == 'False Positive':
            return [self.dis_prob_df, None]
        else:
            return [self.dis_prob_df, self.planet_prob_df]

    @staticmethod
    def calculate_features(ooi):
        # Conversion to Emperical Units 
        ooi['StarRadius'] = ooi['StellarSunRadius'] * 6.96340e8  # Convert to Meters
        ooi['PlanetRadius'] = ooi['PlanetEarthRadius'] * 6.371e6 # Convert to Meters
        ooi['StarSurfaceG'] = (10**ooi['StellarLogG']) / 100     # Convert to m/s^2
        ooi['PeriodInSeconds'] = 86400 * ooi['OrbitalPeriod']    # Convert to Earth Seconds

        # Conversion of Needed Components
        ooi['StarPlanetDistance'] = 1.498e11 * (ooi['OrbitalPeriod'] / 365.25)**(2/3)    # Out -> AU :-> m
        ooi['StarMass'] = (ooi['StarSurfaceG']  * ooi['StarRadius']**2) / (COSMO.NEWTONG)  # Out -> Kg

        # Calculation of Star different values
        ooi['StarWavelengthPeak'] = (2900000) / ooi['StellarEffectiveTemperature']       # Out -> nm
        ooi['StarSunDiameter'] = (2 * ooi['StarRadius']) / (3.19e9)

        # Calculation of Planet different values
        ooi['PlanetEarthVolume'] = (4/3) * np.pi * ooi['PlanetRadius'] / (1.08321e12)

        # Classify Stars and MainSequence Stars
        ooi = ooi.apply(COSMO.classify_star, axis=1)
        ooi = ooi.apply(COSMO.determine_main_seq, axis=1)

        # Convert Measures into Simple / SI / Ratio units
        ooi['StarPlanetDistance'] = ooi['StarPlanetDistance'] / 1.498e11
        ooi['StarWavelengthPeak'] = ooi['StarWavelengthPeak'] / 500
        ooi['PlanetEarthTemperature'] = ooi['PlanetEquilibriumTemperature'] / 255
        ooi['StarSunMass'] = ooi['StarMass'] / 1.989e30
        ooi['OrbitalPeriod'] = ooi['OrbitalPeriod'] / 365.25

        ooi.drop(columns=[
            'StarRadius',
            'PlanetRadius',
            'StellarSunRadius',
            'PlanetEarthRadius',
            'StellarEffectiveTemperature',
            'StarPlanetDistance',
            'StellarLogG',
            'StarSurfaceG',
            'PeriodInSeconds',
            'StarMass',
            'PlanetEquilibriumTemperature',
            'StarWavelengthPeak'
        ], inplace=True)

        return ooi

    @staticmethod
    def scale_features(df, transform_code):
        # Get the Numeric Columns
        numerics = [col for col in df.columns if df[col].dtype != object]

        # Load transformers
        transformers = COSMO.get_transformers(transform_code)

        # Box-Cox Transform
        for col in numerics:
            df[col] = transformers[col].transform(df[col].values.reshape(-1,1))

        return df
               
    @staticmethod
    def get_COS():
        model = joblib.load(os.getcwd()+'\\pages\\models\\COS.joblib')
        return model

    @staticmethod
    def get_MO():
        model = joblib.load(os.getcwd()+'\\pages\\models\\MO.joblib')
        return model

    @staticmethod
    def get_type_decoder():
        le = joblib.load(os.getcwd()+'\\pages\\models\\PlanetLabelEncoder.joblib')
        return le

    @staticmethod
    def type_decode(pred):
        le = COSMO.get_type_decoder()

        print(le.classes_)

        return le.inverse_transform(pred)

    @staticmethod
    def get_transformers(transform_code):
        if transform_code == 'd':
            return joblib.load(os.getcwd()+'\\pages\\models\\DispositionPowerTransformers.joblib')
        elif transform_code == 't':
            return joblib.load(os.getcwd()+'\\pages\\models\\TypePowerTransformers.joblib')
        else:
            raise Exception("Invalid transformer code! only t and d is allowed!")

    @staticmethod    
    def classify_star(row):
        if row['StellarEffectiveTemperature'] >= 30000:
            row['StarClassification'] = 7
        elif row['StellarEffectiveTemperature'] >= 10000:
            row['StarClassification'] = 6
        elif row['StellarEffectiveTemperature'] >= 7500:
            row['StarClassification'] = 5
        elif row['StellarEffectiveTemperature'] >= 6000:
            row['StarClassification'] = 4
        elif row['StellarEffectiveTemperature'] >= 5200:
            row['StarClassification'] = 3
        elif row['StellarEffectiveTemperature'] >= 3700:
            row['StarClassification'] = 2
        elif row['StellarEffectiveTemperature'] >= 2400:
            row['StarClassification'] = 1
        else:
            row['StarClassification'] = 0
        return row

    @staticmethod
    def determine_main_seq(row):
        if row['StarClassification'] == 'O':
            if row['StellarSunRadius'] >= 6.6:
                row['MainSequence'] = 1
            else:
                row['MainSequence'] = 0
        elif row['StarClassification'] == 'B':
            if 1.8 <= row['StellarSunRadius'] <= 6.6:
                row['MainSequence'] = 1
            else:
                row['MainSequence'] = 0
        elif row['StarClassification'] == 'A':
            if 1.4 <= row['StellarSunRadius'] <= 1.8:
                row['MainSequence'] = 1
            else:
                row['MainSequence'] = 0
        elif row['StarClassification'] == 'F':
            if 1.15 <= row['StellarSunRadius'] <= 1.4:
                row['MainSequence'] = 1
            else:
                row['MainSequence'] = 0
        elif row['StarClassification'] == 'G':
            if 0.96 <= row['StellarSunRadius'] <= 1.15:
                row['MainSequence'] = 1
            else:
                row['MainSequence'] = 0
        elif row['StarClassification'] == 'K':
            if 0.7 <= row['StellarSunRadius'] <= 0.96:
                row['MainSequence'] = 1
            else:
                row['MainSequence'] = 0
        elif row['StarClassification'] == 'M':
            if row['StellarSunRadius'] <= 0.7:
                row['MainSequence'] = 1
            else:
                row['MainSequence'] = 0
        else:
            row['MainSequence'] = 0
        return row
