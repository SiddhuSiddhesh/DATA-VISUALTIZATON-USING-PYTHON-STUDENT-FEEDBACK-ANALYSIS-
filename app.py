import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Course Feedback Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main { padding-top: 0rem; }
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }
    h1 { color: #1f77b4; text-align: center; }
    h2 { color: #2ca02c; margin-top: 30px; }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Course Feedback Analysis Dashboard")
st.markdown("---")

# Load data with caching
@st.cache_data
def load_data():
    return pd.read_csv("course_feedback_dataset_1000.csv.csv")

original_data = load_data()
data = original_data.copy()

# Store original for reset
if 'original_data' not in st.session_state:
    st.session_state.original_data = original_data.copy()

# ============================================
# ADVANCED SIDEBAR FILTERS
# ============================================
st.sidebar.title("🔍 Advanced Filters")
st.sidebar.markdown("---")

# Filter 1: Subject Selection
if "Subject" in data.columns:
    subject_filter = st.sidebar.multiselect(
        "📚 Select Subject(s)",
        sorted(data["Subject"].unique()),
        default=list(data["Subject"].unique())
    )
    if subject_filter:
        data = data[data["Subject"].isin(subject_filter)]

# Filter 2: Rating Range
if "Rating" in data.columns:
    rating_range = st.sidebar.slider(
        "⭐ Rating Range",
        float(data["Rating"].min()),
        float(data["Rating"].max()),
        (float(data["Rating"].min()), float(data["Rating"].max())),
        step=0.5
    )
    data = data[(data["Rating"] >= rating_range[0]) & (data["Rating"] <= rating_range[1])]

# Filter 3: Student ID Range (optional)
if "Student_ID" in data.columns:
    student_id_range = st.sidebar.slider(
        "👤 Student ID Range",
        int(data["Student_ID"].min()),
        int(data["Student_ID"].max()),
        (int(data["Student_ID"].min()), int(data["Student_ID"].max()))
    )
    data = data[(data["Student_ID"] >= student_id_range[0]) & (data["Student_ID"] <= student_id_range[1])]

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset All Filters"):
    st.rerun()

# ============================================
# KEY METRICS
# ============================================
st.subheader("📌 Key Metrics")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("📊 Total Feedbacks", len(data), delta=f"{len(data) - len(original_data)} filtered")
if "Rating" in data.columns:
    avg_rating = data["Rating"].mean()
    col2.metric("⭐ Avg Rating", f"{avg_rating:.2f}/5", delta=f"{avg_rating:.1f}")
if "Subject" in data.columns:
    col3.metric("📚 Unique Subjects", data["Subject"].nunique())
if "Student_ID" in data.columns:
    col4.metric("👤 Total Students", data["Student_ID"].nunique())
col5.metric("💬 Total Comments", len(data[data["Comment"].notna()]))

st.markdown("---")

# ============================================
# VISUALIZATIONS - ROW 1
# ============================================
st.subheader("📈 Main Analytics")

col1, col2 = st.columns(2)

# Chart 1: Rating Distribution (Histogram)
with col1:
    if "Rating" in data.columns:
        st.markdown("### ⭐ Rating Distribution")
        fig = px.histogram(data, x="Rating", nbins=20, 
                          title="Rating Distribution",
                          labels={"Rating": "Rating Score"},
                          color_discrete_sequence=["#1f77b4"])
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Chart 2: Satisfaction Breakdown (Pie Chart)
with col2:
    if "Subject" in data.columns:
        st.markdown("### 📚 Subject Distribution")
        subject_counts = data["Subject"].value_counts()
        fig = px.pie(values=subject_counts.values, names=subject_counts.index,
                    title="Feedback Count by Subject",
                    color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# VISUALIZATIONS - ROW 2
# ============================================
col1, col2 = st.columns(2)

# Chart 3: Course-wise Average Rating
with col1:
    if "Subject" in data.columns and "Rating" in data.columns:
        st.markdown("### 📚 Subject-wise Average Rating")
        subject_avg = data.groupby("Subject")["Rating"].agg(['mean', 'count']).sort_values('mean')
        fig = px.bar(x=subject_avg['mean'], y=subject_avg.index,
                    orientation='h',
                    title="Average Rating by Subject",
                    labels={"x": "Average Rating"},
                    color=subject_avg['mean'],
                    color_continuous_scale="Viridis")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Chart 4: Semester Comparison
with col2:
    if "Student_ID" in data.columns and "Rating" in data.columns:
        st.markdown("### 📊 Rating Distribution Across Students")
        # Group by student and get average rating
        student_avg = data.groupby("Student_ID")["Rating"].mean().head(20)
        fig = px.bar(x=student_avg.index, y=student_avg.values,
                    title="Top 20 Students - Average Rating",
                    labels={"x": "Student ID", "y": "Average Rating"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# VISUALIZATIONS - ROW 3
# ============================================
col1, col2 = st.columns(2)

# Chart 5: Difficulty vs Rating (Scatter)
with col1:
    if "Subject" in data.columns and "Rating" in data.columns:
        st.markdown("### 📊 Rating Count by Subject")
        subject_rating_counts = data.groupby(["Subject", "Rating"]).size().reset_index(name='Count')
        fig = px.scatter(subject_rating_counts, x="Subject", y="Rating", size="Count", color="Rating",
                        title="Rating Distribution Across Subjects",
                        color_continuous_scale="RdYlGn",
                        size_max=15)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# Chart 6: Course Distribution
with col2:
    if "Subject" in data.columns:
        st.markdown("### 📊 Feedback Count by Subject")
        subject_counts = data["Subject"].value_counts().head(15)
        fig = px.bar(x=subject_counts.values, y=subject_counts.index,
                    orientation='h',
                    title="Subjects by Feedback Count",
                    labels={"x": "Feedback Count"},
                    color=subject_counts.values,
                    color_continuous_scale="Blues")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# VISUALIZATIONS - ROW 4
# ============================================
col1, col2 = st.columns(2)

# Chart 7: Satisfaction by Course
with col1:
    if "Subject" in data.columns and "Rating" in data.columns:
        st.markdown("### 📈 Rating Distribution by Subject")
        rating_subject = pd.crosstab(data["Subject"], data["Rating"], normalize='index') * 100
        fig = px.bar(rating_subject, barmode='stack',
                    title="Rating Distribution by Subject",
                    labels={"value": "Percentage", "index": "Subject"})
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# Chart 8: Box Plot - Rating by Satisfaction
with col2:
    if "Subject" in data.columns and "Rating" in data.columns:
        st.markdown("### 📦 Rating Distribution by Subject")
        fig = px.box(data, x="Subject", y="Rating",
                    color="Subject",
                    title="Rating Range by Subject",
                    color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ============================================
# WORD CLOUD
# ============================================
st.markdown("---")
st.subheader("☁️ Comment Analysis")

if "Comment" in data.columns:
    comments_text = " ".join(str(c) for c in data["Comment"] if pd.notna(c) and str(c).strip())
    if comments_text:
        fig, ax = plt.subplots(figsize=(12, 5))
        wc = WordCloud(width=1200, height=500, background_color='white', colormap='viridis').generate(comments_text)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

# ============================================
# DETAILED DATA VIEW
# ============================================
st.markdown("---")
st.subheader("📄 Detailed Dataset")

col1, col2 = st.columns([3, 1])
with col1:
    st.write(f"**Showing {len(data)} records** (out of {len(original_data)} total)")
with col2:
    if st.button("📥 Download Filtered Data (CSV)"):
        csv = data.to_csv(index=False)
        st.download_button("Click to download", csv, "feedback_data.csv", "text/csv")

st.dataframe(data, use_container_width=True, height=400)