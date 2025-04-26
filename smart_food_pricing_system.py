import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import pandas as pd

# Load your custom trained model
model = YOLO(r"C:\Users\Yu Jia\Desktop\Smart Food Pricing System\runs\detect\train10\weights\best.pt")  # Replace with your trained model path

# Set page config
st.set_page_config(page_title="Smart Food Pricing System", layout="wide")
st.markdown("""
    <div style="background-color: #3498db; padding: 20px; border-radius: 10px; text-align: center;">
        <h1 style="color: white;">🍛 Smart Food Pricing System</h1>
        <p style="color: white; font-size: 18px;">Upload your meal photo and instantly get an estimated price!</p>
    </div>
""", unsafe_allow_html=True)

# Custom CSS to hide Streamlit footer and add your name
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .footer-text {
        position: fixed;
        bottom: 0;
        right: 0;
        padding: 10px;
        font-size: 14px;
        color: gray;
    }
    </style>
    <div class="footer-text">Made by Fighting17918</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
# Create two columns
left_col, right_col = st.columns(2)

# --- LEFT SIDE (Upload Image) ---
with left_col:
    if uploaded_file is not None:
        st.markdown("<h2 style='text-align: center; color: white;'>🍴 Uploaded Image</h2>", unsafe_allow_html=True)
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

# --- RIGHT SIDE (Detection Results) ---
with right_col:
    if uploaded_file is not None:
        st.markdown("<h2 style='text-align: center; color: white;'>🔎 Detection Results</h2>", unsafe_allow_html=True)
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        total_pixels = height * width
        
        # Run inference
        results = model.predict(image)

        # Draw boxes on the image
        annotated_img = results[0].plot()
        st.image(annotated_img, caption="Detection Result", use_column_width=True)

        # Calculate total price
        class_names = model.names
        boxes = results[0].boxes
        total_price = 0
        detected_items = []

        price_config = {
            # (your price_config dictionary here, keep it unchanged)
            "AW cola": {"base_price": 1.50, "price_per_pixel": 0.00002},
            "Beijing Beef": {"base_price": 3.50, "price_per_pixel": 0.0001},
            "Chow Mein": {"base_price": 2.50, "price_per_pixel": 0.00007},
            "Fried Rice": {"base_price": 2.00, "price_per_pixel": 0.00005},
            "Hashbrown": {"base_price": 1.00, "price_per_pixel": 0.00004},
            "Honey Walnut Shrimp": {"base_price": 4.00, "price_per_pixel": 0.00012},
            "Kung Pao Chicken": {"base_price": 3.00, "price_per_pixel": 0.0001},
            "String Bean Chicken Breast": {"base_price": 3.00, "price_per_pixel": 0.00009},
            "Super Greens": {"base_price": 2.00, "price_per_pixel": 0.00007},
            "The Original Orange Chicken": {"base_price": 3.50, "price_per_pixel": 0.0001},
            "White Steamed Rice": {"base_price": 1.50, "price_per_pixel": 0.00005},
            "black pepper rice bowl": {"base_price": 3.00, "price_per_pixel": 0.00007},
            "burger": {"base_price": 2.50, "price_per_pixel": 0.00008},
            "carrot_eggs": {"base_price": 1.80, "price_per_pixel": 0.00006},
            "cheese burger": {"base_price": 3.00, "price_per_pixel": 0.00008},
            "chicken waffle": {"base_price": 3.50, "price_per_pixel": 0.00009},
            "chicken_nuggets": {"base_price": 2.00, "price_per_pixel": 0.00008},
            "chinese_cabbage": {"base_price": 1.70, "price_per_pixel": 0.00006},
            "chinese_sausage": {"base_price": 2.20, "price_per_pixel": 0.00007},
            "crispy corn": {"base_price": 1.80, "price_per_pixel": 0.00005},
            "curry": {"base_price": 2.50, "price_per_pixel": 0.00007},
            "french fries": {"base_price": 1.50, "price_per_pixel": 0.00004},
            "fried chicken": {"base_price": 3.00, "price_per_pixel": 0.0001},
            "fried_chicken": {"base_price": 3.00, "price_per_pixel": 0.0001},
            "fried_dumplings": {"base_price": 2.50, "price_per_pixel": 0.00008},
            "fried_eggs": {"base_price": 1.50, "price_per_pixel": 0.00005},
            "mango chicken pocket": {"base_price": 3.20, "price_per_pixel": 0.00009},
            "mozza burger": {"base_price": 3.00, "price_per_pixel": 0.00008},
            "mung_bean_sprouts": {"base_price": 1.50, "price_per_pixel": 0.00005},
            "nugget": {"base_price": 2.00, "price_per_pixel": 0.00008},
            "perkedel": {"base_price": 1.20, "price_per_pixel": 0.00004},
            "rice": {"base_price": 1.50, "price_per_pixel": 0.00005},
            "sprite": {"base_price": 1.50, "price_per_pixel": 0.00002},
            "tostitos cheese dip sauce": {"base_price": 2.00, "price_per_pixel": 0.00006},
            "triangle_hash_brown": {"base_price": 1.20, "price_per_pixel": 0.00004},
            "water_spinach": {"base_price": 1.80, "price_per_pixel": 0.00006}
        }

        for box in boxes:
            cls = int(box.cls)
            label = class_names[cls]
            pretty_label = label.replace("_", " ").title()

            x1, y1, x2, x3 = box.xyxy[0].cpu().numpy()

            box_area = abs((x3 - x1) * (x3 - y1))
            portion_ratio = box_area / total_pixels

            if box_area > 0 and label in price_config:
                portion_ratio = box_area / total_pixels
                item_price = (price_config[label]["base_price"] * portion_ratio * 10)
                total_price += item_price
                detected_items.append([pretty_label, f"RM {item_price:.2f}"])

        if detected_items:
            st.subheader("🧾 Detected Items and Prices")
            price_table = pd.DataFrame(detected_items, columns=["Item", "Price (RM)"])
            st.table(price_table)
        
        # Show total price
        st.success(f"💰 Estimated Total Price: RM {total_price:.2f}")

        # Portion Analysis
        st.subheader("Portion Analysis")
        st.write(f"Total image area: {total_pixels} pixels")
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            st.write(f"{class_names[int(box.cls)]}: {area:.0f} pixels ({area/total_pixels:.1%} of image)")
    else:
        pass

