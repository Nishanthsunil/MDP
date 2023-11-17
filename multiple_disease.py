
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
ss=StandardScaler()
le=LabelEncoder()


parkinson_model=pickle.load(open("C:\\Users\\Nishanth S\\Desktop\\class\\streamlit project_MDP\\parkinsons.sav",'rb'))
alzhimers_model=pickle.load(open("C:\\Users\\Nishanth S\\Desktop\\class\\streamlit project_MDP\\alzhimers.sav","rb"))
heart_disease_model=pickle.load(open("C:\\Users\\Nishanth S\\Desktop\\class\\streamlit project_MDP\\heart_disease.sav",'rb'))
diabetes_prediction_model=pickle.load(open("C:\\Users\\Nishanth S\\Desktop\\class\\streamlit project_MDP\\diabetes.sav","rb"))
breast_cancer_model=pickle.load(open("C:\\Users\\Nishanth S\\Desktop\\class\\streamlit project_MDP\\breastcancer.sav",'rb'))

with st.sidebar:
    selected=option_menu("Multiple Disease Prediction System",
                         ['Diabetes Prediction','Heart Disease Prediction','Breast Cancer Prediction','Parkinson Prediction','Alzhimers Prediction'],icons=["activity",'heart','person','paragraph','radioactive'],default_index=0)


if selected=='Diabetes Prediction':

    data1=pd.read_csv("C:\\Users\\Nishanth S\\Downloads\\archive (4)\\diabetes.csv")
    def median_target(var):
        temp = data1[data1[var].notnull()]
        temp = temp[[var,'Outcome']].groupby(["Outcome"])[[var]].median().reset_index()
        return temp
    columns = data1.columns
    columns = columns.drop("Outcome")
    for i in columns:
        median_target(i)
        data1.loc[(data1['Outcome'] == 0 ) & (data1[i].isnull()), i] = median_target(i)[i][0]
        data1.loc[(data1['Outcome'] == 1 ) & (data1[i].isnull()), i] = median_target(i)[i][1]
    x=data1.drop("Outcome",axis=1)
    y=data1.Outcome
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
    x_train=ss.fit_transform(x_train)
    x_test=ss.transform(x_test)



    st.header("Diabetes Prediction using ML")
    st.subheader("")
    col1,col2,col3=st.columns(3)
    with col1:
        Pregnancies=st.text_input("Number of Pregnancies:")
        SkinThickness=st.text_input("SkinThickness")
        DiabetesPedigreeFunction=st.text_input("Diabetes Pedigree Function")
    with col2:
        Glucose=st.text_input("Glucose level:")
        Insulin=st.text_input("Insulin")
        Age=st.text_input("Age")
    with col3:
        BloodPressure=st.text_input("Blood Pressure level:")
        bmi=st.text_input("BMI")

    diabetes_prediction=""

    if st.button("Diabetes Prediction"):
        prediction=diabetes_prediction_model.predict(ss.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,
                                                       Insulin,bmi,DiabetesPedigreeFunction,Age]]))

        if (prediction[0]==1):
            diabetes_prediction="This Person has Diabetes."
        else:
            diabetes_prediction="This person has no diabetes."
            st.balloons()
    st.success(diabetes_prediction)


if selected=="Heart Disease Prediction":
    st.header("Heart Disease Prediction using ML")
    st.subheader("")

    col1,col2,col3=st.columns(3)
    with col1:
        Age=st.text_input("Age of the Person")
        resting_blood_pressure=st.text_input("Resting Blood Pressure")
        Resting_electro=st.text_input("Resting Electrocardiographic result")
        old_peak=st.text_input("Old Peak Value")
        Thal=st.text_input("Thal")
    with col2:
        sex=st.text_input("Sex of the person")
        Serum_cholestrol=st.text_input("Serum Cholestrol")
        Maximum_heart_rate=st.text_input("Maximum Heart Rate Acheived")
        slope_peak=st.text_input("Slope of Peak exercise ST segment")
    with col3:
        Chest_pain_type=st.text_input("Chest Pain Type")
        fasting_bloog_sugar=st.text_input("Fasting_Blood_Sugar")
        exercise=st.text_input("Exercise Induced Angina")
        number_vessels=st.text_input("Number of Major Vessels")
    
    disease_pred=''

    if st.button("Heart Disease Predction"):
        prediction=heart_disease_model.predict([[Age,sex,Chest_pain_type,resting_blood_pressure,
                                                 Serum_cholestrol,fasting_bloog_sugar,Resting_electro,Maximum_heart_rate,exercise,
                                                 old_peak,slope_peak,number_vessels,Thal]])
        if (prediction[0]==1):
            disease_pred = "This Person has Heart Disease"
        else:
            disease_pred = "This person has no Heart Disease"
            st.balloons()
    
    st.success(disease_pred)

if selected=="Breast Cancer Prediction":

    data=pd.read_csv("C:\\Users\\Nishanth S\\Downloads\\breast-cancer.csv")
    x=data.drop(["diagnosis","id"],axis=1)
    data["diagnosis"]=le.fit_transform(data["diagnosis"])
    y=data['diagnosis']

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
    x_train=ss.fit_transform(x_train)

    st.header("Breast Cancer Prediction using ML")
    st.subheader("")
    col1,col2,col3=st.columns(3)
    with col1:
        radius_mean=st.text_input("Mean Radius:")
        area_mean=st.text_input("Mean Area:")
        concavity_mean=st.text_input("Mean Concavity:")
        fractal_dimension_mean=st.text_input("Mean Fractal Dimension:")
        perimeter_se=st.text_input("Perimeter se:")
        compactness_se=st.text_input("Compactness se:")
        symmetry_se=st.text_input("Symmetry se:")
        texture_worst=st.text_input("Worst Texture:")
        smoothness_worst=st.text_input("Worst Smoothness:")
        concave_points_worst=st.text_input("Worst Concave Points:")
    with col2:
        texture_mean=st.text_input("Mean Texture:")
        smoothness_mean=st.text_input("Mean Smoothness")
        concave_points_mean=st.text_input("Mean Concave Points:")
        radius_se=st.text_input("Radius se")
        area_se=st.text_input("Area se:")
        concavity_se=st.text_input("Concavity se:")
        fractal_dimension_se=st.text_input("Fractal Dimension se:")
        perimeter_worst=st.text_input("Worst Perimeter:")
        compactness_worst=st.text_input("Worst Compactness:")
        symmetry_worst=st.text_input("Worst Symmetry:")
    with col3:
        perimeter_mean=st.text_input("Mean Perimeter:")
        compactness_mean=st.text_input("Mean Compactness:")
        symmetry_mean=st.text_input("Mean Symmetry:")
        texture_se=st.text_input("Texture se:")
        smoothness_se=st.text_input("Smoothness se:")
        concave_points_se=st.text_input("Concave Points se:")
        radius_worst=st.text_input("Worst Radius:")
        area_worst=st.text_input("Worst Area:")
        concavity_worst=st.text_input("Worst Concavity:")
        fractal_dimension_worst=st.text_input("Fractal Dimension Worst:")

    cancer_pred=""

    if st.button("Breast Cancer Prediction"):
        
        prediction=breast_cancer_model.predict(ss.transform([[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,
                                                              symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,
                                                              fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]]))
        if (prediction[0]==1):
            cancer_pred = "This Person has Malignant level of Cancer.\n ie, At the severe stage."
        else:
            cancer_pred = "This person has Benign level of Cancer.\n ie , At the begining stage."
    st.success(cancer_pred)    

if selected=='Parkinson Prediction':

    data=pd.read_csv("C:\\Users\\Nishanth S\\Desktop\\class\\streamlit project_MDP\\Parkinsson disease.csv")
    x=data.drop(['name','status'],axis=1)
    y=data['status']

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
    x_train=ss.fit_transform(x_train)

    st.header("Parkinsons Prediction using ML")
    st.subheader("")
    col1,col2,col3=st.columns(3)
    with col1:
        mdvpfo=st.text_input("Average vocal fundamental frequency(MDVP : Fo(Hz))")
        mdvpjitter=st.text_input("MDVP Jitter(%)")
        mdvpppq=st.text_input("MDVP PPQ")
        mdvpshimmerdb=st.text_input("MDVP shimmer(db)")
        mdvpapq=st.text_input("MDVP : APQ")
        hnr=st.text_input("HNR")
        spread1=st.text_input("Spread1")
        ppe=st.text_input("PPE")

    
    with col2:
        mdvpfhi=st.text_input("Maximum vocal fundamental frequency(MDVP : Fhi(Hz))")
        mdvpabs=st.text_input("MDVP Jitter(abs)")
        jitterddp=st.text_input("Jitter DDP")
        shimmerapq3=st.text_input("Shimmer APQ3")
        shimmerdda=st.text_input("Shimmer DDA")
        rpde=st.text_input("RPDE")
        spread2=st.text_input("Spread2")


    with col3:
        mdvpflo=st.text_input("Minimum vocal fundamental frequency(MDVP : Flo(Hz))")
        mdvprap=st.text_input("MDVP RAP")
        mdvpshimmer=st.text_input("MDVP SHIMMER")
        shimmerapq5=st.text_input("Shimmer APQ5")
        nhr=st.text_input("NHR")
        dfa=st.text_input("DFA")
        d2=st.text_input("D2")
    
    parkinson_pred=""

    if st.button("Parkinson Prediction"):
        prediction=parkinson_model.predict(ss.transform([[mdvpfo,mdvpfhi,mdvpflo,mdvpjitter,mdvpabs,mdvprap,mdvpppq,jitterddp,mdvpshimmer,mdvpshimmerdb,shimmerapq3,
                         shimmerapq5,mdvpapq,shimmerdda,nhr,hnr,rpde,d2,dfa,spread1,spread2,ppe]]))
        if (prediction[0]==1):
            parkinson_pred="THis Person has Parkinson Disease"
        else:
            st.balloons()
            parkinson_pred="This person has no Parkinson disease"

    st.success(parkinson_pred)

if selected=='Alzhimers Prediction':

    data = pd.read_csv("C:\\Users\\Nishanth S\\Desktop\\class\\streamlit project_MDP\\oasis_longitudinal.csv")
    data['Group']=le.fit_transform(data['Group'])
    data['M/F']=le.fit_transform(data['M/F'])
    data['Hand']=le.fit_transform(data['Hand'])
    x=data.drop(["Subject ID","MRI ID","Group",'Hand'],axis=1)
    y=data.Group
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
    x_train=ss.fit_transform(x_train)

    st.header("Alzhimers Prediction Using ML")
    st.subheader("")
    col1,col2,col3=st.columns(3)
    with col1:
        visit=st.text_input("Visit")
        age=st.text_input("AGE")
        mmse=st.text_input("MMSE")
        nwbv=st.text_input("nWBV")

    with col2:
        mrdelay=st.text_input("MR Delay")
        educ=st.text_input("EDUC")
        cdr=st.text_input("CDR")
        asf=st.text_input("ASF")

    with col3:
        sex=st.text_input("SEX")
        ses=st.text_input("SES")
        etiv=st.text_input("eTIV")

    alzhimers_pred=''

    if st.button("Alzhimers Disease Prediction"):
        prediction=alzhimers_model.predict(ss.transform([[visit,mrdelay,sex,age,educ,ses,mmse,cdr,etiv,nwbv,asf]]))

        if (prediction[0]==1):
            alzhimers_pred="Demented"
        if (prediction[0]==2):
            alzhimers_pred="NonDemented"
        else:
            alzhimers_pred="Converted"
    st.success(alzhimers_pred)
