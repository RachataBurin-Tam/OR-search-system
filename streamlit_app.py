import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import re
warnings.filterwarnings('ignore')

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="ระบบค้นหาข้อมูล OR",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS สำหรับ Theme สี OR
st.markdown("""
<style>
    /* Minimal color scheme */
    :root {
        --or-blue: #5BA4CF;
        --or-navy: #3A4D6A;
        --or-light: #E8F4F8;
        --text-primary: #2C3E50;
        --text-secondary: #6C757D;
    }
    
    /* Header และ Title */
    .main h1 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    .main h2, .main h3 {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    /* Buttons - minimal style */
    .stButton>button {
        background-color: #5BA4CF !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton>button:hover {
        background-color: #4A90BA !important;
        box-shadow: 0 2px 8px rgba(91, 164, 207, 0.2) !important;
    }
    
    .stButton>button[kind="primary"] {
        background-color: #5BA4CF !important;
        font-size: 16px !important;
        padding: 10px 24px !important;
    }
    
    /* Download Buttons */
    .stDownloadButton>button {
        background-color: #6C757D !important;
        color: white !important;
        border-radius: 6px !important;
    }
    
    .stDownloadButton>button:hover {
        background-color: #5A6268 !important;
    }
    
    /* Sidebar - minimal */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
    }
    
    /* Sliders - subtle */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #5BA4CF !important;
    }
    
    .stSlider [data-baseweb="slider"] {
        background-color: rgba(91, 164, 207, 0.2) !important;
    }
    
    /* Text Input - minimal */
    .stTextInput>div>div>input {
        border: 1px solid #DEE2E6 !important;
        border-radius: 6px !important;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #5BA4CF !important;
        box-shadow: 0 0 0 0.15rem rgba(91, 164, 207, 0.15) !important;
    }
    
    /* Info boxes - minimal */
    .stInfo {
        background-color: rgba(232, 244, 248, 0.5) !important;
        border-left: 3px solid #5BA4CF !important;
    }
    
    /* Success boxes */
    .stSuccess {
        background-color: rgba(212, 237, 218, 0.5) !important;
        border-left: 3px solid #28A745 !important;
    }
    
    /* Warning boxes */
    .stWarning {
        background-color: rgba(255, 243, 205, 0.5) !important;
        border-left: 3px solid #FFC107 !important;
    }
    
    /* DataFrame - subtle border */
    .stDataFrame {
        border: 1px solid #DEE2E6 !important;
        border-radius: 6px !important;
    }
    
    /* Expander - clean */
    .streamlit-expanderHeader {
        background-color: rgba(248, 249, 250, 0.8) !important;
        border-radius: 6px !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background-color: #5BA4CF !important;
    }
    
    /* File uploader - minimal */
    [data-testid="stFileUploader"] {
        border: 2px dashed #DEE2E6 !important;
        border-radius: 8px !important;
        background-color: rgba(248, 249, 250, 0.5) !important;
    }
    
    /* Logo Container */
    .logo-container {
        text-align: center;
        padding: 15px 0;
        margin-bottom: 15px;
    }
    
    .logo-container img {
        max-width: 180px;
        height: auto;
    }
    
    /* Header - minimal gradient */
    .header-gradient {
        background: linear-gradient(135deg, #F8F9FA 0%, #E8F4F8 100%);
        padding: 25px;
        border-radius: 8px;
        border-left: 4px solid #5BA4CF;
        margin-bottom: 30px;
    }
    
    .header-gradient h1 {
        color: var(--text-primary) !important;
        margin: 0;
        font-size: 2rem;
        font-weight: 600;
    }
    
    .header-gradient p {
        color: var(--text-secondary);
        margin: 8px 0 0 0;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# โหลด model สำหรับ embedding (ใช้ PyTorch backend)
@st.cache_resource
def load_model():
    """โหลด model สำหรับ word embedding - รองรับภาษาไทย"""
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return model
    except:
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            return model
        except:
            return None

# โหลดและเตรียมข้อมูล (รองรับทั้ง Excel และ CSV)
@st.cache_data
def load_and_prepare_data(uploaded_file):
    """อ่านไฟล์ Excel หรือ CSV และรวม text columns"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            encodings = ['utf-8', 'utf-8-sig', 'tis-620', 'cp874', 'iso-8859-11']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except:
                    continue
            
            if df is None:
                st.error("❌ ไม่สามารถอ่านไฟล์ CSV ได้")
                return None, None
                
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file, engine='openpyxl' if file_extension == 'xlsx' else None)
        else:
            st.error(f"❌ ไม่รองรับไฟล์นามสกุล .{file_extension}")
            return None, None
        
        text_columns = ['Material description', 'Short text', 'Content', 'Contract Content']
        available_cols = [col for col in text_columns if col in df.columns]
        
        if not available_cols:
            available_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
        
        if not available_cols:
            st.error("❌ ไม่พบ text columns ในไฟล์")
            return None, None
        
        df['combined_text'] = df[available_cols].fillna('').astype(str).agg(' '.join, axis=1)
        df = df[df['combined_text'].str.strip() != '']
        
        if len(df) == 0:
            st.error("❌ ไม่มีข้อมูลหลังจากทำความสะอาด")
            return None, None
        
        return df, df['combined_text'].tolist()
    
    except Exception as e:
        st.error(f"❌ ไม่สามารถอ่านไฟล์ได้: {str(e)}")
        return None, None

# สร้าง embeddings สำหรับข้อมูลทั้งหมด
@st.cache_data
def create_embeddings(_model, texts):
    """สร้าง embeddings จาก text list"""
    try:
        embeddings = _model.encode(
            texts, 
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=32
        )
        return embeddings
    except Exception as e:
        st.error(f"❌ ไม่สามารถสร้าง embeddings ได้: {str(e)}")
        return None

# คำนวณคะแนน keyword matching
def calculate_keyword_score(query, text):
    """คำนวณคะแนนจากการค้นหา keyword"""
    query_lower = query.lower()
    text_lower = text.lower()
    keywords = query_lower.split()
    
    if not keywords:
        return 0.0
    
    matched_keywords = sum(1 for keyword in keywords if keyword in text_lower)
    keyword_match_ratio = matched_keywords / len(keywords)
    total_occurrences = sum(text_lower.count(keyword) for keyword in keywords)
    frequency_score = min(1.0, np.log1p(total_occurrences) / 5)
    final_score = (keyword_match_ratio * 0.7) + (frequency_score * 0.3)
    
    return final_score

# ค้นหาแบบผสม: Cosine Similarity + Keyword Search
def hybrid_search(query, model, data_embeddings, df, texts, threshold=0.5, 
                  cosine_weight=0.5, keyword_weight=0.5):
    """ค้นหาข้อมูลแบบผสมระหว่าง cosine similarity และ keyword matching"""
    try:
        query_embedding = model.encode([query], convert_to_numpy=True)
        cosine_scores = cosine_similarity(query_embedding, data_embeddings)[0]
        keyword_scores = np.array([calculate_keyword_score(query, text) for text in texts])
        hybrid_scores = (cosine_scores * cosine_weight) + (keyword_scores * keyword_weight)
        
        df_result = df.copy()
        df_result['cosine_score'] = cosine_scores
        df_result['keyword_score'] = keyword_scores
        df_result['hybrid_score'] = hybrid_scores
        df_result = df_result[df_result['hybrid_score'] > threshold]
        df_result = df_result.sort_values('hybrid_score', ascending=False)
        
        return df_result
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการค้นหา: {str(e)}")
        return pd.DataFrame()

# UI หลัก
def main():
    # Sidebar สำหรับการตั้งค่า
    with st.sidebar:
        # แสดงโลโก้ใน Sidebar
        st.markdown("""
        <div class="logo-container">
            <img src="https://i.imgur.com/YOUR_LOGO_URL.png" alt="OR Logo" style="max-width: 180px;">
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.header("⚙️ การตั้งค่า")
        
        uploaded_file = st.file_uploader(
            "อัพโหลดไฟล์ข้อมูล", 
            type=['xlsx', 'xls', 'csv'],
            help="รองรับไฟล์: Excel (.xlsx, .xls) และ CSV (.csv)"
        )
        
        st.markdown("---")
        st.subheader("🎚️ ปรับน้ำหนักการค้นหา")
        
        cosine_weight = st.slider(
            "น้ำหนัก Semantic Search (AI)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="น้ำหนักของการค้นหาตามความหมาย"
        )
        
        keyword_weight = 1.0 - cosine_weight
        st.info(f"🔤 น้ำหนัก Keyword Search: **{keyword_weight:.1f}**")
        
        st.markdown("---")
        
        threshold = st.slider(
            "ค่าคะแนนขั้นต่ำ", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3,
            step=0.05,
            help="แสดงเฉพาะผลลัพธ์ที่มีคะแนนรวมมากกว่าค่านี้"
        )
        
        st.markdown("---")
        st.info("""
        💡 **คำแนะนำ:**
        
        **Semantic Search (AI):**
        - ค้นหาตามความหมาย
        - เข้าใจคำพ้อง คำคล้าย
        
        **Keyword Search:**
        - ค้นหาคำตรงๆ
        - เร็วและแม่นยำ
        
        **แนะนำ:** ใช้ 50-50 สำหรับผลลัพธ์ที่ดีที่สุด
        """)
    
    # # Header หลักพร้อมโลโก้
    # col_logo, col_title = st.columns([1, 4])
    
    # with col_logo:
    #     # แสดงโลโก้ขนาดใหญ่ในหน้าหลัก
    #     st.markdown("""
    #     <div style="text-align: center; padding: 10px;">
    #         <img src="https://i.imgur.com/YOUR_LOGO_URL.png" alt="OR Logo" style="max-width: 120px; height: auto;">
    #     </div>
    #     """, unsafe_allow_html=True)
    
    # with col_title:
    #     st.markdown("""
    #     <div class="header-gradient">
    #         <h1>🔍 ระบบค้นหาข้อมูลด้วย AI</h1>
    #         <p>Hybrid Search: Semantic Search (AI) + Keyword Matching</p>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ตรวจสอบว่ามีไฟล์อัพโหลดหรือไม่
    if uploaded_file is not None:
        try:
            # โหลด model
            with st.spinner("🔄 กำลังโหลด AI model..."):
                model = load_model()
            
            if model is None:
                st.error("❌ ไม่สามารถโหลด model ได้")
                return
            
            # โหลดข้อมูล
            with st.spinner("📂 กำลังโหลดข้อมูล..."):
                df, combined_texts = load_and_prepare_data(uploaded_file)
                
                if df is None or combined_texts is None:
                    return
            
            # สร้าง embeddings
            with st.spinner("🧠 กำลังเตรียมข้อมูล..."):
                data_embeddings = create_embeddings(model, combined_texts)
                
                if data_embeddings is None:
                    return
            
            st.success(f"✅ พร้อมค้นหา! ({len(df):,} แถว)")
            
            # แสดงสถิติข้อมูล
            with st.expander("📊 ข้อมูลทั่วไป"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("จำนวนแถว", f"{len(df):,}")
                with col2:
                    st.metric("จำนวน Columns", len(df.columns))
                with col3:
                    st.metric("ขนาด Embeddings", f"{data_embeddings.shape[1]}D")
                with col4:
                    st.metric("ประเภทไฟล์", uploaded_file.name.split('.')[-1].upper())
                
                st.subheader("🔍 ตัวอย่างข้อมูล (5 แถวแรก)")
                st.dataframe(df.head(), use_container_width=True)
            
            # ช่องค้นหา
            st.markdown("### 🔎 ค้นหาข้อมูล")
            query = st.text_input(
                "ใส่คำค้นหา:", 
                placeholder="พิมพ์สิ่งที่ต้องการค้นหา...",
                key="search_query"
            )
            
            # ปุ่มค้นหา
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                search_button = st.button("🔍 ค้นหา", use_container_width=True, type="primary")
            
            # แสดงผลการค้นหา
            if search_button and query.strip():
                with st.spinner("🔍 กำลังค้นหา..."):
                    results = hybrid_search(
                        query, model, data_embeddings, df, combined_texts,
                        threshold=threshold,
                        cosine_weight=cosine_weight,
                        keyword_weight=keyword_weight
                    )
                
                st.markdown("---")
                st.markdown("### 📊 ผลการค้นหา")
                
                if len(results) > 0:
                    st.success(f"✅ พบ {len(results):,} รายการที่เกี่ยวข้อง")
                    
                    # แสดงค่าสถิติ
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🏆 คะแนนรวมสูงสุด", f"{results['hybrid_score'].max():.2%}")
                    with col2:
                        st.metric("📊 คะแนนรวมเฉลี่ย", f"{results['hybrid_score'].mean():.2%}")
                    with col3:
                        st.metric("🤖 AI Score เฉลี่ย", f"{results['cosine_score'].mean():.2%}")
                    with col4:
                        st.metric("🔤 Keyword Score เฉลี่ย", f"{results['keyword_score'].mean():.2%}")
                    
                    st.markdown("---")
                    
                    # แสดงผลลัพธ์ทั้งหมด
                    display_columns = [col for col in results.columns if col != 'combined_text']
                    results_display = results[display_columns].copy()
                    
                    # จัดรูปแบบคะแนนเป็น percentage
                    score_columns = ['hybrid_score', 'cosine_score', 'keyword_score']
                    for col in score_columns:
                        if col in results_display.columns:
                            results_display[col] = results_display[col].apply(lambda x: f"{x:.2%}")
                    
                    # เรียงลำดับ columns ให้คะแนนอยู่ข้างหน้า
                    score_cols = [col for col in score_columns if col in results_display.columns]
                    other_cols = [col for col in results_display.columns if col not in score_columns]
                    results_display = results_display[score_cols + other_cols]
                    
                    # แสดงเป็น dataframe แบบ scroll ได้
                    st.dataframe(
                        results_display, 
                        use_container_width=True,
                        hide_index=True,
                        height=600
                    )
                    
                    # คำอธิบายคะแนน
                    with st.expander("ℹ️ คำอธิบายคะแนน"):
                        st.markdown(f"""
                        - **hybrid_score**: คะแนนรวม = (AI Score × {cosine_weight:.1f}) + (Keyword Score × {keyword_weight:.1f})
                        - **cosine_score**: คะแนนจาก Semantic Search (ความหมาย)
                        - **keyword_score**: คะแนนจาก Keyword Matching (คำตรงกัน)
                        
                        ยิ่งคะแนนสูง = ยิ่งเกี่ยวข้องกับคำค้นหามากขึ้น
                        """)
                    
                    # ปุ่มดาวน์โหลด
                    st.markdown("### 💾 ดาวน์โหลดผลลัพธ์")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        csv = results.to_csv(index=False, encoding='utf-8-sig')
                        st.download_button(
                            label="📥 ดาวน์โหลด CSV",
                            data=csv,
                            file_name=f"search_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        from io import BytesIO
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            results.to_excel(writer, index=False, sheet_name='Results')
                        excel_data = output.getvalue()
                        
                        st.download_button(
                            label="📥 ดาวน์โหลด Excel",
                            data=excel_data,
                            file_name=f"search_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    
                else:
                    st.warning("⚠️ ไม่พบข้อมูลที่เกี่ยวข้อง")
                    st.info("💡 ลองปรับค่าคะแนนขั้นต่ำ หรือเปลี่ยนคำค้นหา")
            
            elif search_button and not query.strip():
                st.warning("⚠️ กรุณาใส่คำค้นหา")
                
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
    
    else:
        st.info("👈 กรุณาอัพโหลดไฟล์ที่ Sidebar เพื่อเริ่มต้นใช้งาน")
        
        # แสดงข้อมูลเกี่ยวกับระบบ
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(0,180,216,0.1) 0%, rgba(27,20,100,0.1) 100%); 
                        padding: 30px; border-radius: 12px; border-left: 5px solid #00B4D8;">
                <h3 style="color: #1B1464;">📝 วิธีใช้งาน</h3>
                <ol style="color: #333; line-height: 2;">
                    <li><strong>อัพโหลดไฟล์</strong> (Excel หรือ CSV)</li>
                    <li><strong>ปรับน้ำหนักการค้นหา</strong> ระหว่าง AI และ Keyword</li>
                    <li><strong>ใส่คำค้นหา</strong> แล้วกดปุ่ม "ค้นหา"</li>
                    <li><strong>ดูผลลัพธ์</strong> พร้อม scroll ดูทุกแถว</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(237,28,36,0.1) 0%, rgba(27,20,100,0.1) 100%); 
                        padding: 30px; border-radius: 12px; border-left: 5px solid #ED1C24;">
                <h3 style="color: #1B1464;">🎯 ความแตกต่างของการค้นหา</h3>
                <table style="width: 100%; border-collapse: collapse; margin-top: 15px;">
                    <tr style="background-color: rgba(0,180,216,0.1);">
                        <th style="padding: 12px; text-align: left; color: #1B1464;">ประเภท</th>
                        <th style="padding: 12px; text-align: left; color: #1B1464;">จุดเด่น</th>
                        <th style="padding: 12px; text-align: left; color: #1B1464;">เหมาะกับ</th>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>AI Search</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #ddd;">เข้าใจความหมาย</td>
                        <td style="padding: 10px; border-bottom: 1px solid #ddd;">คำค้นหาซับซ้อน</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px; border-bottom: 1px solid #ddd;"><strong>Keyword</strong></td>
                        <td style="padding: 10px; border-bottom: 1px solid #ddd;">ตรงตัว แม่นยำ</td>
                        <td style="padding: 10px; border-bottom: 1px solid #ddd;">คำเฉพาะเจาะจง</td>
                    </tr>
                    <tr>
                        <td style="padding: 10px;"><strong>Hybrid</strong></td>
                        <td style="padding: 10px;">สมดุลที่สุด</td>
                        <td style="padding: 10px;">ทั่วไป</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
        

if __name__ == "__main__":
    # คำแนะนำ: วิธีเพิ่มโลโก้
    # 1. อัพโหลดโลโก้ไปที่ imgur.com หรือ image hosting service
    # 2. คัดลอก direct link ของรูป
    # 3. แทนที่ 'YOUR_LOGO_URL.png' ด้วย URL จริง ในบรรทัดที่ 255 และ 279
    # หรือใช้วิธีอื่น: วางไฟล์โลโก้ในโฟลเดอร์เดียวกับโค้ด แล้วใช้ st.image('logo.png')
    
    main()
