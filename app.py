#core Pkgs
import streamlit as st

# EDA Pkgs
import pandas as pd
import numpy as np

# Data Viz Pkg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
import altair as alt
st.set_option('deprecation.showPyplotGlobalUse', False)

# ML pkgs
# ML pkgs
import pickle


model_reg = open('model.pkl', 'rb')

reg=pickle.load(model_reg) # our model


def predict_chance(age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall,):
    prediction=reg.predict_proba([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]]) #predictions using our model
    return prediction




def main():

	activities = ["About", "EDA","ML Model",]
	choice = st.sidebar.selectbox("Select Activities",activities)

	if choice =='About':
		st.subheader("About section")
		st.write("Creator of App Hassan Jama. The app has 2 other section EDA and ML model you can check them out on the side bar.")
		st.write("""We will use the heart attack dataset containing following parameters, to predict the chances of heart attack in patients:

- Age: Age of the patient,
- Sex: Sex of the patient,
- exang: exercise induced angina (1 = yes; 0 = no),
- caa: number of major vessels (0-3),
- cp: Chest Pain type chest pain type Value 1: typical angina Value 2: atypical angina Value 3: non-anginal pain Value 4: asymptomatic,
- trtbps: resting blood pressure (in mm Hg),
- chol: cholestoral in mg/dl fetched via BMI sensor,
- fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false),
- rest_ecg: resting electrocardiographic results Value 0: normal Value 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV) Value 2: showing probable or definite left ventricular hypertrophy by Estes' criteria,
- thalach: maximum heart rate achieved,
- target: 0= less chance of heart attack 1= more chance of heart attack.

""")
		st.write("""Dataset biases/shortcomings:
- Sample size is quite small with only 303 samples, so more data is needed.
- The overall health of the sampled population is unknown such as any underlying conditions that might effect certain prameters.
	""")
		st.write(""" Datasource: https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset""")
		st.write("Algorithm used: The algroithm used on the ML section of the app is an Logistic Regression with an accuracy score of 91.8%. To learn more about logistic regression algrothim click on this link https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html")

	elif choice == 'EDA':
		st.subheader("Exploratory Data Analysis & Data Visualization")

	# Load Our Dataset
		df = pd.read_csv("heart-3.csv")

		group_names = ['20\'s', '30\'s', '40\'s', '50\'s', '60\'s', '70\'s']
		df['AgeBD'] = pd.cut(df['age'], bins=[20,29,39,49,59,69,79], labels=group_names, include_lowest=True)
		AgeBD = df.AgeBD.value_counts().to_frame()

		df['M/F'] = df['sex'].apply(lambda x : 'Male' if x == 1 else 'Female')

		if st.checkbox("Show DataSet"):
			st.dataframe(df)

		if st.checkbox("Show missing Data"):
			st.write(df.isnull().sum())

		if st.checkbox("Show Shape"):
			st.write(df.shape)

		if st.checkbox("Show Columns"):
			all_columns = df.columns.to_list()
			st.write(all_columns)

		if st.checkbox("Summary"):
			st.write(df.describe())

		if st.checkbox("Selected Columns"):
			selected_columns = st.multiselect("Select Columns", df.columns.to_list())
			new_df = df[selected_columns]
			st.dataframe(new_df)

		if st.checkbox("Show Value Count of Age groups"):
			st.dataframe(df['AgeBD'].value_counts())
			st.write(sns.barplot(AgeBD.index, df["AgeBD"].value_counts()))
			st.pyplot()

		if st.checkbox("Show value count of Target Variable"):
			st.write('Show value count of Target Variable ',df['output'].value_counts())
			st.write(sns.countplot(x='output', data=df, palette = 'deep'))
			st.pyplot()

		if st.checkbox("Male to Female Ratio"):
			st.write(sns.countplot(x='M/F', data=df, palette = 'deep'))
			st.pyplot()

		if st.checkbox("Show correlation plot"):
			st.write('Correlation Plot',sns.heatmap(df.corr(),annot=True, linewidths=1, cmap = "YlGnBu"))
			st.pyplot()

		if st.checkbox('output distrubtion using Independent Variable'):
			all_columns_names1 = df.columns.tolist()
			columnsx = st.selectbox("Select X Column",all_columns_names1)
			st.write(sns.countplot(x=columnsx, data = df, hue='output', palette = 'deep'))
			st.pyplot()

		if st.checkbox("Pie Plot"):
			all_columns = df.columns.to_list()
			column_to_plot = st.selectbox("Select 1 Column",all_columns)
			pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
			st.write(pie_plot)
			st.pyplot()

	elif choice == 'ML Model':
		st.subheader("ML Model")
		html_temp="""
        <div>
        <h2>Heart Attack Prediction ML app</h2>
        </div>
        """




		#sex1 = {"Female": 0, "Male": 1}
		#cp1 = {"Value 1: typical angina":0, "Value 2: atypical angina": 1, "Value 3: non-anginal pain": 2,"Value 4: asymptomatic": 3}
		#fbs1 = {"fasting blood sugar > 120":1, "fasting blood sugar < 120 ": 0}
		#restecg1 = {"0":0, "1":0}
		#exng1 = {"Yes":1, "No":0}
		#slp1 = {"0":0, "1":1,"2":2}


		st.markdown(html_temp,unsafe_allow_html=True) #a simple html
		age=st.slider("Age", 10, 100)
		sex=st.selectbox("Sex", ("Male", "Female"))
		cp=st.selectbox("Chest Pain type chest",("Value 1: typical angina","Value 2: atypical angina","Value 3: non-anginal pain","Value 4: asymptomatic"))
		trtbps=st.number_input("Resting Blood Pressure")
		chol=st.number_input("Cholestoral in mg/dl")
		fbs=st.selectbox("Fasting Blood Sugar > 120",("fasting blood sugar > 120", "fasting blood sugar < 120 "))
		restecg=st.selectbox("Resting Electrocardiographic Results",("0","1"))
		thalachh=st.number_input("Maximum Heart Rate Achieved")
		exng=st.selectbox("Exercise Induced Angina", ("Yes", "No"))
		oldpeak=st.number_input("Previous Peak")
		slp=st.slider("Slope",0,2)
		caa=st.slider("Number of Major Vessels", 0,4)
		thall=st.slider("Thal Rate",0, 3)


		if sex =="Male":
			sex = 1
		else:
			sex = 0

		if cp =="Value 1: typical angina":
			cp = 0
		if cp == "Value 2: atypical angina":
			cp = 1
		if cp == "Value 3: non-anginal pain":
			cp = 2
		else:
			cp == 3


		if fbs =="fasting blood sugar > 120":
			fbs = 1
		else:
			fbs = 0

		if restecg =="1":
			restecg = 1
		else:
			restecg = 0

		if exng =="Yes":
			exng = 1
		else:
			exng = 0




     #giving inputs as used in building the model
		result=""
		if st.button("Predict"):
				result=predict_chance(age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall)
				#if result == 1:
					#st.success("more chance of heart attack{}".format(result))
				#else:
				 	#st.success("less chance of heart attack{}".format(result))
		st.success("Based on the inputs above the first number indicate the percent chance of not having a heart attack, while the second number indicates the percent chance of having a heart attack.")
		st.success(result)





if __name__ == '__main__':
	main()
