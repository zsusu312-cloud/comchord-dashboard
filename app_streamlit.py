# -*- coding: utf-8 -*-
"""
ComChord Leadership Dashboard (Manager-Focused Version)
Author: Zhu Xun
Purpose:
Streamlit web app for visualizing multi-meeting NLP analytics results,
optimized for Sarah’s leadership insights and business relevance.
"""

# ========= IMPORTS =========
import streamlit as st
import pandas as pd
import plotly.express as px
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
import plotly.io as pio

# ========= PAGE CONFIG =========
st.set_page_config(
    page_title="ComChord Leadership Dashboard",
    layout="wide",
    page_icon="💼"
)

# ========= DATA LOADING =========
@st.cache_data
def load_data():
    """Load preprocessed NLP analysis outputs from output_data/ folder"""
    df_sentence = pd.read_csv("output_data/manager_report_sentence_all.csv")
    df_block = pd.read_csv("output_data/manager_report_block_all.csv")
    df_proj = pd.read_csv("output_data/project_mentions_all.csv")
    kpi = pd.read_csv("output_data/leadership_kpis.csv")
    with open("output_data/insight_summary_all.txt", "r", encoding="utf-8") as f:
        summary = f.read()
    return df_sentence, df_block, df_proj, kpi, summary

df_sentence, df_block, df_proj, kpi, summary = load_data()

# ========= OVERVIEW SECTION =========
st.title("💼 ComChord Leadership 1:1 Analysis Dashboard")
st.markdown("#### Author: **Zhu Xun** | Data Source: *1-on-1 meeting transcripts*")

st.markdown("""
### 🔹 Overview & Key Business Insights
This dashboard provides an integrated view of **Sarah’s leadership and communication effectiveness** 
across multiple 1-on-1 meetings.  
By combining emotional sentiment, engagement behavior, and project-level initiative, it connects **communication patterns to business performance**.

**Key Insights:**
- Team morale remains positive overall, with dips aligning to project milestones.  
- Sarah demonstrates strong leadership ownership, though team-initiated input can grow further.  
- Clear communication and low misunderstanding rates indicate efficient dialogue.  
- Strengthening acknowledgment loops and encouraging collaborative initiative could enhance innovation and team empowerment.
""")

# ========= KPI OVERVIEW =========
col1, col2, col3, col4 = st.columns(4)
col1.metric("Meetings Analyzed", int(kpi['Meetings_Analyzed'][0]))
col2.metric("Sarah Talk Ratio", f"{kpi['Sarah_Talk_Ratio'][0]*100:.1f}%")
col3.metric("Empathy Index", f"{kpi['Sarah_Empathy_Index'][0]:.3f}")
col4.metric("Team Positivity Avg", f"{kpi['Team_Positive_Avg'][0]:.3f}")

st.divider()

# ========= FILTERS =========
speaker_list = sorted(df_sentence["Speaker"].unique())
meeting_list = sorted(df_sentence["Meeting"].unique())
col_filter1, col_filter2 = st.columns(2)
selected_speakers = col_filter1.multiselect("👥 Select Speaker(s):", speaker_list, default=speaker_list)
selected_meetings = col_filter2.multiselect("📅 Select Meeting(s):", meeting_list, default=meeting_list)

filtered_df = df_sentence[
    (df_sentence["Speaker"].isin(selected_speakers)) &
    (df_sentence["Meeting"].isin(selected_meetings))
]

# ========= 1. LEADERSHIP OVERVIEW (RADAR) =========
st.subheader("🌟 Leadership Communication Overview")
st.markdown("""
**How Leadership Style Shapes Team Dynamics**  
This radar chart compares Sarah’s communication attributes with team patterns, focusing on direction, empathy, and positivity.
""")

kpi_radar = pd.DataFrame({
    "Metric": ["Talk Ratio", "Empathy Index", "Team Positivity"],
    "Sarah": [kpi['Sarah_Talk_Ratio'][0], kpi['Sarah_Empathy_Index'][0], kpi['Team_Positive_Avg'][0]],
    "Team": [1 - kpi['Sarah_Talk_Ratio'][0], 0.15, kpi['Team_Positive_Avg'][0]],
})
fig_radar = px.line_polar(kpi_radar.melt(id_vars="Metric"), r="value", theta="Metric", color="variable", line_close=True)
fig_radar.update_traces(fill="toself")
st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("""
Sarah demonstrates **strong communication control** with balanced empathy and positivity.  
A slightly higher talk ratio reflects clear direction-setting — future sessions could include more open-ended input to strengthen team ownership.
""")

# ========= 2. SENTIMENT TREND =========
st.subheader("📈 Team Sentiment Evolution")
st.markdown("""
**How Team Emotions Shifted During 1:1 Conversations**  
This view tracks emotional tone across meetings to highlight engagement and morale trends.
""")

trend_df = (
    filtered_df.groupby(["Meeting", "Speaker"])["compound"]
    .mean()
    .reset_index()
)
fig_sent = px.line(
    trend_df,
    x="Meeting",
    y="compound",
    color="Speaker",
    markers=True,
    title="Average Sentiment by Meeting and Speaker"
)
fig_sent.update_layout(xaxis_title="Meeting", yaxis_title="Sentiment (Compound Score)")
st.plotly_chart(fig_sent, use_container_width=True)

st.markdown("""
Sarah’s emotional tone follows a **“high–low–high”** curve, showing adaptability to team dynamics.  
Alex’s sentiment decreases over time, suggesting fatigue or delivery pressure.  
Javier’s “low–high–low” trend implies fluctuating engagement, possibly tied to task load or project stage.  
The pattern overall indicates emotionally intelligent leadership but room for steadier motivation reinforcement.
""")

# ========= 3. ACTION ITEMS & QUESTION RATES =========
st.subheader("🚀 Engagement Behavior & Initiative")
st.markdown("""
**How the Team Acts and Asks — Engagement Patterns**  
Measures the frequency of action-oriented and inquiry-driven dialogue.
""")

colA, colB = st.columns(2)
action_df = filtered_df.groupby("Speaker")["Is_ActionItem"].mean().reset_index()
question_df = filtered_df.groupby("Speaker")["Is_Question"].mean().reset_index()
colA.bar_chart(action_df.set_index("Speaker"))
colB.bar_chart(question_df.set_index("Speaker"))

st.markdown("""
Sarah contributes the highest number of action-driven and questioning statements, 
demonstrating proactive direction.  
Team members’ lower questioning frequency indicates strong compliance but limited upward challenge — 
encouraging curiosity and clarification may boost shared accountability.
""")
# ========= 4. KEY TOPICS (TF-IDF ANALYSIS) =========
st.subheader("🧭 Key Topics Identified from 1:1 Discussions")
st.markdown("""
**Understanding What Drives the Conversation**  
This section identifies major discussion themes extracted from meeting transcripts using the TF-IDF method.

The results show that while a large proportion of meeting records contain scattered or open-ended dialogue,  
certain key terms consistently emerge — such as **process**, **product**, **growth**, **OKR**, **performance**, and **vendor**.  
These terms reflect recurring areas of focus in team communication.
""")

topic_counts = df_block.explode("Topics")["Topics"].value_counts().reset_index()
topic_counts.columns = ["Topic", "Count"]
fig_topics = px.bar(topic_counts, x="Topic", y="Count", title="Most Frequent Discussion Topics")
st.plotly_chart(fig_topics, use_container_width=True)

st.markdown("""
The analysis indicates that conversations are primarily centered on **project execution** and **business growth**.  
Frequent mentions of *process* and *performance* highlight attention to workflow optimization and measurable outcomes,  
while *product* and *growth* relate to development and expansion initiatives.  
Mentions of *OKR* and *vendor* suggest active discussions around goal alignment and external collaboration.
""")


# ========= 4. COMMUNICATION EFFECTIVENESS =========
st.subheader("💬 Communication Effectiveness Analysis")
st.markdown("""
**Clarity, Understanding, and Initiative in Leadership Communication**  
Evaluates how effectively Sarah’s instructions are understood and how actively projects are initiated.
""")

kpi_path = "output_data/leadership_kpis.csv"
kpi_df = pd.read_csv(kpi_path).iloc[0]

col1, col2, col3 = st.columns(3)
col1.metric(
    label="🗣️ Sarah Clarity Rate",
    value=f"{kpi_df['Sarah_Clarity_Rate']*100:.1f}%",
    delta="↓ Needs Reinforcement" if kpi_df['Sarah_Clarity_Rate'] < 0.4 else "↑ Clear Communication"
)
col2.metric(
    label="❓ Team Unclear Rate",
    value=f"{kpi_df['Team_Unclear_Rate']*100:.1f}%",
    delta="↓ Good Understanding" if kpi_df['Team_Unclear_Rate'] < 0.1 else "↑ Needs Clarification"
)
col3.metric(
    label="🏗️ Sarah Project Initiative",
    value=f"{kpi_df['Sarah_Project_Initiation_Rate']*100:.1f}%",
    delta="↓ Encourage Team Input" if kpi_df['Sarah_Project_Initiation_Rate'] < 0.5 else "↑ Strong Leadership"
)

project_data = pd.DataFrame({
    "Initiator": ["Sarah", "Team"],
    "Projects Discussed": [kpi_df["Projects_By_Sarah"], kpi_df["Projects_By_Team"]]
})
st.bar_chart(project_data.set_index("Initiator"), use_container_width=True)

st.markdown("""
Sarah’s clarity rate (2.9%) shows clear messaging but limited acknowledgment feedback.  
The team’s uncertainty rate (0.2%) confirms high comprehension.  
With 41.4% project initiative, Sarah effectively drives progress —  
future improvement lies in inviting more team-originated ideas to balance direction and collaboration.
""")

# ========= 5. PROJECT PROGRESS =========
st.subheader("🔍 Project Momentum & Progress")
st.markdown("""
**What’s Moving — and Who’s Driving It**  
Tracks project mentions, methods, and sentiment patterns across meetings.
""")

st.dataframe(
    df_proj[["Meeting", "Speaker", "Project", "Method", "Progress", "Sentiment"]],
    hide_index=True,
    use_container_width=True,
)

st.markdown("""
Project discussions mainly revolve around **Agile and Product-related** topics, showing a strong operational focus.  
Sentiment remains largely positive (>0.4), reflecting constructive communication.  
Introducing more cross-functional or analytical discussions could foster broader innovation.
""")

# ========= 6. INSIGHTS SUMMARY =========
st.subheader("📝 Leadership Takeaways for the Next Cycle")
st.text_area("Summary of Findings", summary, height=250)
st.markdown("""
- Maintain consistent tone and structure to stabilize morale.  
- Invite team members to restate or summarize key takeaways.  
- Balance direction-setting with collaborative brainstorming.  
- Use affirming and empathetic language to strengthen connection.
""")

st.caption("© 2025 Zhu Xun | ComChord Leadership Analysis Dashboard")
st.divider()

# ========= PDF EXPORT =========
st.subheader("📄 Export Leadership Report")
def generate_pdf(kpi_df, summary_text, fig_sentiment, fig_topics):
    """Generate a structured leadership report as a downloadable PDF"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Sarah Leadership 1-on-1 Report</b>", styles["Title"]))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Paragraph("Author: Zhu Xun", styles["Normal"]))
    story.append(Spacer(1, 20))

    # KPI Table
    story.append(Paragraph("<b>Leadership KPI Summary</b>", styles["Heading2"]))
    data = [["Metric", "Value"]]
    for col in kpi_df.index:
        data.append([col, str(kpi_df[col])])
    table = Table(data, colWidths=[200, 200])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (0,0), (-1,-1), "CENTER")
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # Sentiment Trend
    story.append(Paragraph("<b>Sentiment Trend by Meeting</b>", styles["Heading2"]))
    img_buf = BytesIO()
    pio.write_image(fig_sentiment, img_buf, format="png", scale=2)
    img_buf.seek(0)
    story.append(Image(img_buf, width=400, height=250))
    story.append(Spacer(1, 20))

    # Topic Frequency
    story.append(Paragraph("<b>Topic Frequency Overview</b>", styles["Heading2"]))
    img_buf2 = BytesIO()
    pio.write_image(fig_topics, img_buf2, format="png", scale=2)
    img_buf2.seek(0)
    story.append(Image(img_buf2, width=400, height=250))
    story.append(Spacer(1, 20))

    # Summary
    story.append(Paragraph("<b>Insights Summary</b>", styles["Heading2"]))
    for line in summary_text.split("\n"):
        story.append(Paragraph(line, styles["Normal"]))
    story.append(Spacer(1, 10))
    story.append(Paragraph("© 2025 Zhu Xun | ComChord Leadership Analysis Dashboard", styles["Italic"]))

    doc.build(story)
    buffer.seek(0)
    return buffer

if st.button("📄 Generate PDF Leadership Report"):
    try:
        pdf_buffer = generate_pdf(kpi_df, summary, fig_sent, fig_radar)
        st.success("✅ Leadership report successfully generated!")
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_buffer,
            file_name=f"Sarah_Leadership_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf"
        )
    except Exception as e:
        st.error(f"Error while generating PDF: {e}")
