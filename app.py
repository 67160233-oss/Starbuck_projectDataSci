import streamlit as st
import pandas as pd
import joblib

# ===== 1. การตั้งค่าหน้าเว็บ =====
st.set_page_config(
    page_title="Starbucks Spend Predictor",
    page_icon="☕",
    layout="centered"
)

# ===== 2. โหลดโมเดล =====
@st.cache_resource
def load_model():
    # ตรวจสอบให้แน่ใจว่าชื่อไฟล์ .pkl ตรงกับที่คุณเซฟไว้ในขั้นตอนก่อนหน้า
    return joblib.load("starbucks_model.pkl")

with st.spinner("กำลังโหลดโมเดล..."):
    model = load_model()

# ===== 3. หน้าจอหลักของ App =====
st.title("☕ Starbucks Spend Predictor")
st.write("ระบบพยากรณ์ยอดใช้จ่ายของลูกค้าโดยใช้ Machine Learning (Regression)")
st.divider()

# สร้างส่วนการกรอกข้อมูลแบ่งเป็น 2 คอลัมน์เพื่อให้ดูสวยงาม
col1, col2 = st.columns(2)

with col1:
    st.subheader("พฤติกรรมการสั่ง")
    cart_size = st.number_input("จำนวนรายการที่สั่ง (Cart Size)", min_value=1, max_value=20, value=1)
    num_customizations = st.number_input("จำนวนการปรับแต่งเครื่องดื่ม", min_value=0, max_value=10, value=0)
    order_channel = st.selectbox("ช่องทางการสั่งซื้อ", ['Drive-Thru', 'Mobile App', 'In-Store', 'Kiosk'])
    drink_category = st.selectbox("ประเภทเครื่องดื่ม", ['Coffee', 'Tea', 'Frappuccino', 'Refresher', 'Bakery', 'Other'])

with col2:
    st.subheader("ข้อมูลลูกค้าและร้าน")
    is_rewards_member = st.selectbox("เป็นสมาชิก Rewards?", [True, False])
    has_food_item = st.selectbox("สั่งอาหารด้วยหรือไม่?", [True, False])
    store_location_type = st.selectbox("ประเภททำเลร้าน", ['Urban', 'Suburban', 'Rural'])
    order_ahead = st.selectbox("สั่งล่วงหน้า (Order Ahead)?", [True, False])

# ข้อมูลคงที่ (กำหนดไว้เผื่อโมเดลต้องการใช้แต่ไม่ได้ให้ผู้ใช้เลือก)
region = "Northeast"
customer_age_group = "25-34"
customer_gender = "Other"

# ===== 4. การทำนายผล =====
if st.button("ทำนายยอดเงินที่ลูกค้าจะจ่าย", type="primary", use_container_width=True):
    
    # สร้าง DataFrame ให้ชื่อคอลัมน์และลำดับตรงกับตอน Train เป๊ะๆ
    input_df = pd.DataFrame([{
        'order_channel': order_channel,
        'store_location_type': store_location_type,
        'region': region,
        'customer_age_group': customer_age_group,
        'customer_gender': customer_gender,
        'is_rewards_member': is_rewards_member,
        'cart_size': cart_size,
        'num_customizations': num_customizations,
        'drink_category': drink_category,
        'has_food_item': has_food_item,
        'order_ahead': order_ahead
    }])

    # ทำนายผล
    prediction = model.predict(input_df)

    # แสดงผล
    st.divider()
    st.balloons()
    st.markdown(f"""
    <div style="text-align: center;">
        <h3>💰 ยอดใช้จ่ายที่คาดการณ์</h3>
        <h1 style="color: #00704A;">${prediction[0]:.2f}</h1>
    </div>
    """, unsafe_allow_html=True)

# ===== 5. ข้อมูลเพิ่มเติมใน Sidebar =====
st.sidebar.header("About This Project")
st.sidebar.info("""
โปรเจคนี้เป็นส่วนหนึ่งของวิชา ML Deployment 
โดยใช้ข้อมูลพฤติกรรมการสั่งซื้อของ Starbucks 
มาสร้างโมเดล Regression เพื่อทำนายรายได้ต่อบิล
""")