import os
import streamlit as st
import numpy as np
import pickle
from streamlit_extras.colored_header import colored_header
from streamlit_extras.card import card
from streamlit_extras.stylable_container import stylable_container
from langchain.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Set page config
st.set_page_config(
    page_title="HealthGuard AI - Disease Prediction & Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    /* Main styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #ffffff;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }
    
    /* Card styling */
    .card {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    
    .card:hover {
        transform: translateY(-5px);
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 20px;
        padding: 10px 24px;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
    }
    
    /* Input field styling */
    .stNumberInput, .stSelectbox {
        margin-bottom: 1rem;
    }
    
    /* Success message */
    .stAlert.success {
        border-left: 5px solid #28a745;
    }
    
    /* Warning message */
    .stAlert.warning {
        border-left: 5px solid #ffc107;
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 25px;
        background-color: #f8f9fa;
        border-radius: 10px 10px 0 0;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    working_dir = os.path.dirname(os.path.abspath(__file__))
    models = {
        'diabetes': pickle.load(open(f'{working_dir}/saved_models/diabetes.pkl', 'rb')),
        'heart': pickle.load(open(f'{working_dir}/saved_models/heart.pkl', 'rb')),
        'kidney': pickle.load(open(f'{working_dir}/saved_models/kidney.pkl', 'rb'))
    }
    return models

models = load_models()

DB_FAISS_PATH = "vectorstore/db_faiss"
HF_TOKEN = os.environ.get("CHAT_BOT_TOKEN")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Load LLM for Chatbot
def load_llm(huggingface_repo_id):
    llm = HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        task="text-generation",
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )
    return llm

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
Don't provide anything outside of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk, please.
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

# Load Vector Store for Chatbot
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Create QA Chain for Chatbot
def create_qa_chain():
    vectorstore = get_vectorstore()
    qa_chain = RetrievalQA.from_chain_type(
        llm=load_llm(HUGGINGFACE_REPO_ID),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
    )
    return qa_chain

# Sidebar navigation
with st.sidebar:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("logo.png", width=120)  # Image will be centered
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <h3 style="color: #2c3e50; margin-bottom: 5px;">HealthGuard AI</h3>
        <p style="color: #7f8c8d; font-size: 14px;">Your personal health assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    selected = st.radio(
        "Navigation",
        ["🏠 Home", "🩸 Diabetes", "🫀 Heart", "🧬 Kidney", "💬 HealthBot"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; margin-top: 30px;">
        <p style="color: #7f8c8d; font-size: 12px;">For emergency medical help, please contact:</p>
        <p style="color: #e74c3c; font-weight: bold;">911 or your local emergency number</p>
    </div>
    """, unsafe_allow_html=True)

# Home Page
if selected == "🏠 Home":
    colored_header(
        label="Welcome to HealthGuard AI",
        description="Your AI-powered health companion",
        color_name="blue-70"
    )
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 40px;">
        <p style="font-size: 18px; color: #555;">
            Early detection saves lives. Our AI models help assess your risk for various diseases 
            based on your health metrics.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        card(
            title="Diabetes Prediction",
            text="Assess your risk for diabetes based on key health indicators",
            image="https://images.unsplash.com/photo-1532938911079-1b06ac7ceec7?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=500&q=80",
            url="#diabetes-prediction"
        )
    
    with col2:
        card(
            title="Heart Disease Prediction",
            text="Evaluate your cardiovascular health with our advanced model",
            image="https://images.unsplash.com/photo-1579684385127-1ef15d508118?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=500&q=80",
            url="#heart-disease-prediction"
        )
    
    with col3:
        card(
            title="Kidney Disease Prediction",
            text="Check your kidney health status with key biomarkers",
            image="https://images.unsplash.com/photo-1631815588090-d4bfec5b1ccb?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=500&q=80",
            url="#kidney-disease-prediction"
        )
    
    st.markdown("---")
    
    # How it works section
    colored_header(
        label="How It Works",
        description="Simple steps to get your health assessment",
        color_name="green-70"
    )
    
    steps = [
        {"icon": "1️⃣", "title": "Select a Health Check", "desc": "Choose from diabetes, heart, or kidney assessment"},
        {"icon": "2️⃣", "title": "Enter Your Health Metrics", "desc": "Provide the required health information"},
        {"icon": "3️⃣", "title": "Get Instant Results", "desc": "Receive AI-powered risk assessment"},
        {"icon": "4️⃣", "title": "Understand Next Steps", "desc": "Get personalized recommendations based on results"}
    ]
    
    for step in steps:
        with st.container():
            col1, col2 = st.columns([1, 10])
            with col1:
                st.markdown(f"<h3 style='text-align: center;'>{step['icon']}</h3>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<h4>{step['title']}</h4><p style='color: #555;'>{step['desc']}</p>", unsafe_allow_html=True)
            st.markdown("---")

# Diabetes Prediction Page
elif selected == "🩸 Diabetes":
    colored_header(
        label="Diabetes Risk Assessment",
        description="Enter your health metrics to evaluate diabetes risk",
        color_name="orange-70"
    )
    
    with st.expander("ℹ️ About Diabetes", expanded=False):
        st.markdown("""
        Diabetes is a chronic disease that occurs when the pancreas is no longer able to make insulin, or when the body cannot make good use of the insulin it produces. There are several types of diabetes:
        
        - **Type 1 Diabetes**: An autoimmune condition where the body attacks insulin-producing cells
        - **Type 2 Diabetes**: Often related to lifestyle factors and insulin resistance
        - **Gestational Diabetes**: Occurs during pregnancy
        
        Early detection and proper management can prevent complications.
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        with stylable_container(
            key="diabetes_form",
            css_styles="""
                {
                   
                    padding: 20px;
                    
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                }
            """
        ):
            st.markdown("### Health Metrics")
            Pregnancies = st.number_input("Number of Pregnancies", min_value=0, help="Enter 0 if not applicable")
            Glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, help="Normal range: 70-99 mg/dL (fasting)")
            BloodPressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, help="Normal: <120/80 mmHg")
            SkinThickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, help="Triceps skin fold thickness")
            
    with col2:
        with stylable_container(
            key="diabetes_form_cont",
            css_styles="""
                {
                    
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                }
            """
        ):
            st.markdown("### Additional Metrics")
            Insulin = st.number_input("Insulin Level (μU/mL)", min_value=0, max_value=300, help="Normal fasting level: <25 μU/mL")
            BMI = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=50.0, step=0.1, help="Normal range: 18.5-24.9")
            DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01, help="Indicates genetic influence")
            Age = st.number_input("Age (years)", min_value=1, max_value=120)
    
    if st.button("🔍 Assess Diabetes Risk", use_container_width=True):
        with st.spinner("Analyzing your health data..."):
            NewBMI_Underweight = 0
            NewBMI_Overweight = 0
            NewBMI_Obesity_1 = 0
            NewBMI_Obesity_2 = 0
            NewBMI_Obesity_3 = 0
            NewInsulinScore_Normal = 0
            NewGlucose_Low = 0
            NewGlucose_Normal = 0
            NewGlucose_Overweight = 0
            NewGlucose_Secret = 0
            
            if BMI <= 18.5:
                NewBMI_Underweight = 1
            elif 24.9 < BMI <= 29.9:
                NewBMI_Overweight = 1
            elif 29.9 < BMI <= 34.9:
                NewBMI_Obesity_1 = 1
            elif 34.9 < BMI <= 39.9:
                NewBMI_Obesity_2 = 1
            elif BMI > 39.9:
                NewBMI_Obesity_3 = 1
            if 16 <= Insulin <= 166:
                NewInsulinScore_Normal = 1
                
            if Glucose <= 70:
                NewGlucose_Low = 1
            elif 70 < Glucose <= 99:
                NewGlucose_Normal = 1
            elif 99 < Glucose <= 126:
                NewGlucose_Overweight = 1
            elif Glucose > 126:
                NewGlucose_Secret = 1


            input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                  Insulin, BMI, DiabetesPedigreeFunction, Age,
                                  NewBMI_Underweight, NewBMI_Overweight, NewBMI_Obesity_1,
                                  NewBMI_Obesity_2, NewBMI_Obesity_3, NewInsulinScore_Normal,
                                  NewGlucose_Low, NewGlucose_Normal, NewGlucose_Overweight,
                                  NewGlucose_Secret]])
            result = models['diabetes'].predict(input_data)[0]
            
            if result:
                st.error("## 🚨 High Risk of Diabetes Detected")
                st.markdown("""
                Our assessment indicates you may be at risk for diabetes. This is not a diagnosis but suggests you should consult a healthcare professional.

                ### Recommended Actions:
                - 🩺 Schedule a doctor's appointment
                - � Get a fasting blood glucose test
                - 🏃‍♂️ Increase physical activity
                - 🥗 Adopt a balanced diet low in processed sugars
                
                ### Common Symptoms to Watch For:
                - Excessive thirst and hunger
                - Frequent urination
                - Unexplained weight loss
                - Fatigue and blurred vision
                """)
                
                with st.expander("📚 Learn More About Diabetes Management"):
                    st.markdown("""
                    **Dietary Recommendations:**
                    - Focus on non-starchy vegetables
                    - Choose whole grains over refined
                    - Limit sugary beverages
                    - Control portion sizes
                    
                    **Lifestyle Changes:**
                    - Aim for 150 mins of exercise weekly
                    - Maintain healthy weight
                    - Monitor blood sugar if advised
                    - Quit smoking if applicable
                    """)
            else:
                st.success("## ✅ Low Risk of Diabetes")
                st.markdown("""
                Our assessment suggests you currently have a low risk of diabetes. Maintain your healthy habits!

                ### Prevention Tips:
                - 🍎 Continue eating a balanced diet
                - 🚶‍♀️ Stay physically active
                - ⚖️ Maintain healthy weight
                - 🩺 Get regular check-ups
                
                ### Healthy Blood Sugar Levels:
                - Fasting: 70-99 mg/dL
                - 2 hours after eating: <140 mg/dL
                - A1C: <5.7%
                """)
                
                with st.expander("📊 Understanding Your Results"):
                    st.markdown("""
                    **What Does Low Risk Mean?**
                    - Your current health metrics are within normal ranges
                    - No immediate signs of diabetes
                    - Continue monitoring as risk increases with age
                    
                    **When to Retest:**
                    - If you develop symptoms
                    - Every 3 years if over 45
                    - More often if overweight or family history
                    """)

# Heart Disease Prediction Page
elif selected == "🫀 Heart":
    colored_header(
        label="Heart Disease Risk Assessment",
        description="Evaluate your cardiovascular health",
        color_name="red-70"
    )
    
    with st.expander("ℹ️ About Heart Disease", expanded=False):
        st.markdown("""
        Heart disease refers to various types of heart conditions. The most common is coronary artery disease, which can lead to heart attacks. 
        
        **Key risk factors include:**
        - High blood pressure
        - High cholesterol
        - Smoking
        - Diabetes
        - Obesity
        - Physical inactivity
        
        Early detection of risk factors can help prevent serious complications.
        """)
    
    tabs = st.tabs(["Basic Info", "Medical History", "Test Results"])
    
    with tabs[0]:
        st.markdown("### Personal Information")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age (years)", min_value=1, max_value=120)
            sex = st.selectbox("Sex", ["Female", "Male"], help="Biological sex for medical assessment")
            cp = st.selectbox("Chest Pain Type", 
                             ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"],
                             help="Type of chest pain if experienced")
        with col2:
            trestbps = st.number_input("Resting Blood Pressure (mmHg)", min_value=50, max_value=250, value=120)
            chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200,
                                  help="Serum cholesterol level")
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No", "Yes"])
    
    with tabs[1]:
        st.markdown("### Medical History")
        exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"],
                           help="Chest pain during physical activity")
        thal = st.selectbox("Thalassemia", 
                          ["Normal", "Fixed Defect", "Reversible Defect"],
                          help="Blood disorder affecting hemoglobin")
    
    with tabs[2]:
        st.markdown("### Medical Test Results")
        col1, col2 = st.columns(2)
        with col1:
            restecg = st.selectbox("Resting ECG", 
                                 ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"])
            thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220,
                                    help="Highest heart rate during exercise")
        with col2:
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=6.0, step=0.1,
                                    help="ST segment depression during exercise")
            slope = st.selectbox("Slope of Peak Exercise ST", 
                               ["Upsloping", "Flat", "Downsloping"])
            ca = st.selectbox("Major Vessels Colored", [0, 1, 2, 3],
                            help="Number of major vessels colored by fluoroscopy")
    
    if st.button("❤️ Assess Heart Disease Risk", use_container_width=True):
        # Convert inputs to model format
        sex_num = 1 if sex == "Male" else 0
        cp_num = ["Typical angina", "Atypical angina", "Non-anginal pain", "Asymptomatic"].index(cp)
        fbs_num = 1 if fbs == "Yes" else 0
        exang_num = 1 if exang == "Yes" else 0
        thal_num = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal) + 1
        restecg_num = ["Normal", "ST-T wave abnormality", "Left ventricular hypertrophy"].index(restecg)
        slope_num = ["Upsloping", "Flat", "Downsloping"].index(slope)
        
        with st.spinner("Analyzing cardiovascular health..."):
            input_data = np.array([[age, sex_num, cp_num, trestbps, chol, fbs_num,
                                  restecg_num, thalach, exang_num, oldpeak, slope_num, ca, thal_num]])
            result = models['heart'].predict(input_data)[0]
            
            if result:
                st.error("## ❤️‍🩹 Elevated Heart Disease Risk Detected")
                st.markdown("""
                Our assessment suggests you may be at increased risk for heart disease. This doesn't mean you have heart disease now, but indicates higher risk factors.

                ### Immediate Recommendations:
                - 🩺 Schedule a cardiology consultation
                - ❤️ Monitor blood pressure regularly
                - 🚭 Quit smoking if applicable
                - 🏋️ Begin supervised exercise program
                
                ### Warning Signs to Watch For:
                - Chest pain or discomfort
                - Shortness of breath
                - Palpitations or irregular heartbeat
                - Dizziness or fainting
                """)
                
                with st.expander("💓 Heart-Healthy Lifestyle Tips"):
                    st.markdown("""
                    **Diet for Heart Health:**
                    - Emphasize fruits, vegetables, whole grains
                    - Include healthy fats (avocados, nuts, olive oil)
                    - Limit saturated and trans fats
                    - Reduce sodium intake
                    
                    **Exercise Guidelines:**
                    - 150 mins moderate exercise weekly
                    - Include aerobic and strength training
                    - Start slowly if new to exercise
                    - Monitor intensity (able to talk during exercise)
                    """)
            else:
                st.success("## ❤️ Healthy Heart Assessment")
                st.markdown("""
                Your current health metrics suggest low risk for heart disease. Keep up the good work!

                ### Heart Health Maintenance:
                - 🍎 Continue heart-healthy diet
                - 🏃‍♀️ Maintain regular physical activity
                - ⚖️ Keep healthy weight
                - 🧘 Manage stress effectively
                
                ### Ideal Heart Health Numbers:
                - Blood Pressure: <120/80 mmHg
                - Cholesterol: <200 mg/dL
                - BMI: 18.5-24.9
                - Fasting Glucose: <100 mg/dL
                """)
                
                with st.expander("🛡️ Prevention Strategies"):
                    st.markdown("""
                    **Lifelong Heart Protection:**
                    - Know your family history
                    - Get regular check-ups
                    - Don't ignore symptoms
                    - Limit alcohol consumption
                    
                    **When to Reassess:**
                    - Annually for adults over 40
                    - If you develop new symptoms
                    - With significant weight changes
                    - If family history changes
                    """)

# Kidney Disease Prediction Page
elif selected == "🧬 Kidney":
    colored_header(
        label="Kidney Health Assessment",
        description="Evaluate your kidney function metrics",
        color_name="violet-70"
    )
    
    with st.expander("ℹ️ About Kidney Disease", expanded=False):
        st.markdown("""
        Chronic kidney disease (CKD) means your kidneys are damaged and can't filter blood properly. 
        
        **Stages of CKD:**
        1. Stage 1: Normal function but signs of damage
        2. Stage 2: Mild loss of function
        3. Stage 3: Moderate loss
        4. Stage 4: Severe loss
        5. Stage 5: Kidney failure
        
        Early detection can slow progression and prevent complications.
        """)
    
    st.markdown("### Kidney Function Metrics")
    col1, col2 = st.columns(2)
    
    with col1:
        with stylable_container(
            key="kidney_form1",
            css_styles="""
                {
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                }
            """
        ):
            age = st.number_input("Age (years)", min_value=1, max_value=120)
            bp = st.number_input("Blood Pressure (mmHg)", min_value=50, max_value=250, value=120)
            sg = st.number_input("Specific Gravity", min_value=1.000, max_value=1.030, value=1.015, step=0.001,
                                help="Urine concentration measurement")
            al = st.selectbox("Albumin Level", 
                             ["Normal", "Above normal", "High", "Very high"],
                             help="Protein in urine indicator")
            rbc = st.selectbox("Red Blood Cells", 
                              ["Normal", "Abnormal"],
                              help="Presence of red blood cells in urine")
            pc = st.selectbox("Pus Cells", 
                             ["Normal", "Abnormal"],
                             help="Presence of white blood cells in urine")
            pcc = st.selectbox("Pus Cell Clumps", 
                              ["Not present", "Present"],
                              help="Clumps of white blood cells")
            ba = st.selectbox("Bacteria", 
                             ["Not present", "Present"],
                             help="Presence of bacteria in urine")
    
    with col2:
        with stylable_container(
            key="kidney_form2",
            css_styles="""
                {
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
                }
            """
        ):
            su = st.selectbox("Sugar Level", 
                             ["Normal", "Above normal", "High", "Very high"],
                             help="Glucose in urine")
            bgr = st.number_input("Blood Glucose Random (mg/dL)", min_value=50, max_value=500, value=100)
            bu = st.number_input("Blood Urea (mg/dL)", min_value=5, max_value=200, value=15,
                               help="Normal: 7-20 mg/dL")
            sc = st.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=10.0, value=0.8, step=0.1,
                               help="Normal: 0.6-1.2 mg/dL (men), 0.5-1.1 mg/dL (women)")
            sod = st.number_input("Sodium (mEq/L)", min_value=100, max_value=200, value=140,
                                help="Normal: 135-145 mEq/L")
            pot = st.number_input("Potassium (mEq/L)", min_value=2.0, max_value=7.0, value=4.0, step=0.1,
                                help="Normal: 3.5-5.0 mEq/L")
            hemo = st.number_input("Hemoglobin (g/dL)", min_value=3.0, max_value=20.0, value=12.0, step=0.1,
                                 help="Normal: 12-16 g/dL (women), 13-17 g/dL (men)")
            pcv = st.number_input("Packed Cell Volume (%)", min_value=10, max_value=70, value=40,
                                help="Normal: 36-46% (women), 40-54% (men)")
    
    # Additional inputs in an expander to keep UI clean
    with st.expander("➕ Additional Health Indicators", expanded=False):
        col3, col4 = st.columns(2)
        with col3:
            wbcc = st.number_input("White Blood Cell Count (cells/cumm)", min_value=1000, max_value=20000, value=7000)
            rbcc = st.number_input("Red Blood Cell Count (millions/cumm)", min_value=2.0, max_value=8.0, value=4.5, step=0.1)
            htn = st.selectbox("Hypertension", ["No", "Yes"])
            dm = st.selectbox("Diabetes Mellitus", ["No", "Yes"])
        with col4:
            cad = st.selectbox("Coronary Artery Disease", ["No", "Yes"])
            appet = st.selectbox("Appetite", ["Good", "Poor"])
            pe = st.selectbox("Pedal Edema", ["No", "Yes"])
            ane = st.selectbox("Anemia", ["No", "Yes"])
    
    if st.button("🧪 Assess Kidney Health", use_container_width=True):
        # Convert categorical inputs to numerical values
        al_num = ["Normal", "Above normal", "High", "Very high"].index(al)
        su_num = ["Normal", "Above normal", "High", "Very high"].index(su)
        rbc_num = ["Normal", "Abnormal"].index(rbc)
        pc_num = ["Normal", "Abnormal"].index(pc)
        pcc_num = ["Not present", "Present"].index(pcc)
        ba_num = ["Not present", "Present"].index(ba)
        htn_num = ["No", "Yes"].index(htn)
        dm_num = ["No", "Yes"].index(dm)
        cad_num = ["No", "Yes"].index(cad)
        appet_num = ["Good", "Poor"].index(appet)
        pe_num = ["No", "Yes"].index(pe)
        ane_num = ["No", "Yes"].index(ane)
        
        with st.spinner("Analyzing kidney function..."):
            input_data = np.array([[age, bp, sg, al_num, su_num, rbc_num, pc_num, pcc_num,
                                  ba_num, bgr, bu, sc, sod, pot, hemo, pcv, wbcc, rbcc,
                                  htn_num, dm_num, cad_num, appet_num, pe_num, ane_num]])
            
            result = models['kidney'].predict(input_data)[0]
            
            if result:
                st.error("## 🚨 Potential Kidney Issues Detected")
                st.markdown("""
                Our assessment suggests possible kidney function impairment. This requires medical evaluation.

                ### Recommended Actions:
                - 🩺 Consult a nephrologist
                - 🩸 Get comprehensive kidney function tests
                - 💊 Review current medications with doctor
                - 🧂 Reduce sodium intake
                
                ### Symptoms to Monitor:
                - Swelling in feet/ankles
                - Fatigue and weakness
                - Changes in urination patterns
                - Nausea or loss of appetite
                """)
                
                with st.expander("🩺 Understanding Kidney Tests"):
                    st.markdown("""
                    **Common Kidney Tests:**
                    - eGFR: Estimates filtration rate (normal >60)
                    - Urine albumin: Checks for protein
                    - Serum creatinine: Waste product measurement
                    - BUN: Blood urea nitrogen
                    
                    **Next Steps After Abnormal Results:**
                    - Repeat tests for confirmation
                    - Ultrasound or CT scan if needed
                    - Possible biopsy in some cases
                    - Referral to specialist
                    """)
            else:
                st.success("## ✅ Healthy Kidney Function")
                st.markdown("""
                Your kidney function metrics appear within normal ranges. Maintain kidney-healthy habits!

                ### Kidney Health Tips:
                - 💧 Stay well hydrated
                - 🍎 Eat balanced diet
                - 🚭 Avoid smoking
                - 💊 Use NSAIDs cautiously
                
                ### Normal Kidney Values:
                - eGFR: ≥60 mL/min/1.73m²
                - Urine albumin: <30 mg/g
                - Serum creatinine: 0.6-1.2 mg/dL
                - BUN: 7-20 mg/dL
                """)
                
                with st.expander("🧊 Kidney Protection Strategies"):
                    st.markdown("""
                    **Preventive Measures:**
                    - Control blood pressure
                    - Manage diabetes if present
                    - Maintain healthy weight
                    - Limit alcohol consumption
                    
                    **When to Retest:**
                    - Annually if over 60
                    - With diabetes or hypertension
                    - If symptoms develop
                    - With family history of kidney disease
                    """)
# Chatbot Page
elif selected == "💬 HealthBot":
    colored_header(
        label="HealthGuard AI Assistant",
        description="Ask me anything about health and wellness",
        color_name="blue-green-70"
    )
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm HealthGuard AI. How can I help you with your health questions today?"}
        ]
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Accept user input
    if prompt := st.chat_input("Ask your health question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            try:
                # Use the QA chain from your old code
                qa_chain = create_qa_chain()
                response = qa_chain.invoke({'query': prompt})
                
                # Get the response and source documents
                result = response["result"]
                source_documents = response["source_documents"]
                
                # Format the response (you can modify this part)
                response_to_show = result + "\n\n*Sources:*\n" + "\n".join([f"- {doc.metadata['source']}" for doc in source_documents])
                
                st.markdown(response_to_show)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_to_show})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error processing your request: {str(e)}"
                st.markdown(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    st.markdown("---")
    st.markdown("### 💡 Common Questions to Ask:")
    
    cols = st.columns(3)
    with cols[0]:
        if st.button("Diabetes prevention tips"):
            st.session_state.messages.append({"role": "user", "content": "What are some diabetes prevention tips?"})
            st.rerun()
    
    with cols[1]:
        if st.button("Heart-healthy foods"):
            st.session_state.messages.append({"role": "user", "content": "What foods are good for heart health?"})
            st.rerun()
    
    with cols[2]:
        if st.button("Kidney function tests"):
            st.session_state.messages.append({"role": "user", "content": "What tests check kidney function?"})
            st.rerun()