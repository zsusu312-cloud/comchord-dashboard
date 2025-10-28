# (Disabled in Streamlit Cloud deployment)

# -*- coding: utf-8 -*-
"""
Multi-Meeting Leadership NLP Analysis
Author: Zhu Xun
Purpose:
Automatically process multiple 1:1 meeting Word files
from input_meetings/ and output all analytics to output_data/
"""

import os, re
import pandas as pd
from docx import Document
from textblob import TextBlob
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import spacy

# ========= CONFIG =========
BASE_DIR = r"C:\Users\27338\Desktop\comchord_1on1_analysis"
IN_DIR = os.path.join(BASE_DIR, "input_meetings")
OUT_DIR = os.path.join(BASE_DIR, "output_data")

# 创建输出文件夹（若不存在）
os.makedirs(OUT_DIR, exist_ok=True)

# ========= NLP 模型加载 =========
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.lang.en import English
    print("⚠️ spaCy model not found, using blank English tokenizer.")
    nlp = English()

nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# ========= FUNCTION =========
def run_meeting_analysis(file_path):
    """Process one Word file and return sentence, block, project DataFrames"""
    doc = Document(file_path)
    text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
    fname = os.path.basename(file_path).replace(".docx", "")

    # 提取元数据
    meta_pattern = r"Participants:\s*(.*?)\s*Date:\s*(.*?)\s*Time:\s*(.*?)\s*Format:\s*(.*?)\s*(?:Sarah|Alex|Javier):"
    meta = re.search(meta_pattern, text, re.S)
    if meta:
        participants_raw = meta.group(1).strip()
        meeting_date = meta.group(2).strip()
        meeting_time = meta.group(3).strip()
        meeting_format = meta.group(4).strip()
    else:
        participants_raw, meeting_date, meeting_time, meeting_format = "", "", "", ""
    participants = ", ".join(
        [re.sub(r"^\s*-\s*", "", p.strip()) for p in participants_raw.splitlines() if p.strip()]
    )

    # 拆分块
    dialogue_pattern = r"(Sarah|Alex|Javier):\s*(.*?)(?=(?:\n(?:Sarah|Alex|Javier):)|\Z)"
    matches = re.finditer(dialogue_pattern, text, re.S)
    blocks = [{"BlockID": i + 1, "Speaker": m.group(1), "Content_Block": m.group(2).strip()} for i, m in enumerate(matches)]
    df_block = pd.DataFrame(blocks)

    # 分句
    def split_sentences(t):
        t = re.sub(r"\s+", " ", t).strip()
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if len(s.strip()) > 2]

    sentence_rows = []
    for _, row in df_block.iterrows():
        for j, s in enumerate(split_sentences(row["Content_Block"]), 1):
            sentence_rows.append(
                {"Speaker": row["Speaker"], "BlockID": row["BlockID"], "SentenceID": j, "Sentence": s}
            )
    df_sentence = pd.DataFrame(sentence_rows)

    # 情绪分析（VADER + TextBlob）
    df_sentence["TextBlob_Sentiment"] = df_sentence["Sentence"].apply(lambda x: TextBlob(x).sentiment.polarity)
    vader = df_sentence["Sentence"].apply(lambda x: sia.polarity_scores(x))
    vader_df = pd.DataFrame(list(vader))
    df_sentence = pd.concat([df_sentence, vader_df], axis=1)
    df_sentence = df_sentence[df_sentence["Sentence"].str.len() > 5]
    df_sentence = df_sentence[df_sentence["neu"] < 0.9]

    # TF-IDF (block-level)
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.85)
    X = tfidf.fit_transform(df_block["Content_Block"])
    terms = np.array(tfidf.get_feature_names_out())

    def top_terms(row_vec, k=5):
        arr = row_vec.toarray().ravel()
        idx = arr.argsort()[::-1][:k]
        return [terms[i] for i in idx if arr[i] > 0]

    df_block["TopTerms"] = [top_terms(X[i]) for i in range(X.shape[0])]

    # 主题映射
    TOPIC_MAP = {
        "feature": "product", "dashboard": "product", "search": "product", "mobile": "product",
        "onboarding": "product", "api": "vendor", "integration": "vendor", "performance": "performance",
        "okr": "okr", "adoption": "okr", "engagement": "okr", "metric": "okr",
        "career": "growth", "strategy": "growth", "planning": "growth", "stakeholder": "growth",
        "sprint": "process", "scoring": "process", "model": "process", "priority": "process", "design": "process",
        "communication": "process", "feedback": "process"
    }
    def map_topics(terms_list):
        return list({TOPIC_MAP[t] for t in terms_list if t in TOPIC_MAP})
    df_block["Topics"] = df_block["TopTerms"].apply(map_topics)

    # 行动项 / 提问检测
    df_sentence["Is_ActionItem"] = df_sentence["Sentence"].str.contains(
        r"\b(will|gonna|need to|plan to|should|try to|let's|i'll)\b", case=False
    ).astype(int)
    df_sentence["Is_Question"] = df_sentence["Sentence"].str.contains(r"\?$").astype(int)

    # 项目信息提取
    PROJECT_HINTS = ["project", "dashboard", "report", "system", "app", "platform", "module", "feature"]
    METHOD_MAP = {"sprint": "Agile", "sql": "Analytics", "survey": "Research", "launch": "Product"}
    PROGRESS_KEYWORDS = {"completed": "Done", "working on": "Ongoing", "delayed": "Issue", "plan to": "Planned"}

    records = []
    for _, r in df_sentence.iterrows():
        s = r["Sentence"]
        speaker = r["Speaker"]
        sent_doc = nlp(s)
        projects = [
            " ".join([w.text for w in sent_doc[max(t.i - 2, 0): min(t.i + 3, len(sent_doc))]])
            for t in sent_doc if any(h in t.text.lower() for h in PROJECT_HINTS)
        ]
        methods = list({v for k, v in METHOD_MAP.items() if k in s.lower()})
        progress = None
        for k, v in PROGRESS_KEYWORDS.items():
            if re.search(rf"\b{k}\b", s, re.I):
                progress = v
                break
        owner = "Manager" if speaker == "Sarah" else "DirectReport"
        if projects or methods or progress:
            records.append({
                "Meeting": fname, "Speaker": speaker, "Owner": owner,
                "Sentence": s, "Project": projects, "Method": methods,
                "Progress": progress, "Sentiment": r["compound"]
            })
    df_proj = pd.DataFrame(records)

    # 加元数据
    for df_ in [df_sentence, df_block]:
        df_["Meeting"] = fname
        df_["Meeting_Date"] = meeting_date
        df_["Meeting_Time"] = meeting_time
        df_["Format"] = meeting_format
        df_["Participants"] = participants

    return df_sentence, df_block, df_proj


# ========= MAIN EXECUTION =========
all_sentences, all_blocks, all_projects = [], [], []

for file in os.listdir(IN_DIR):
    if file.endswith(".docx") and not file.startswith("~$"):
        print(f"Processing {file} ...")
        fpath = os.path.join(IN_DIR, file)
        df_s, df_b, df_p = run_meeting_analysis(fpath)
        all_sentences.append(df_s)
        all_blocks.append(df_b)
        all_projects.append(df_p)

# 合并并输出
df_sentence_all = pd.concat(all_sentences, ignore_index=True)
df_block_all = pd.concat(all_blocks, ignore_index=True)
df_proj_all = pd.concat(all_projects, ignore_index=True)

df_sentence_all.to_csv(os.path.join(OUT_DIR, "manager_report_sentence_all.csv"), index=False, encoding="utf-8-sig")
df_block_all.to_csv(os.path.join(OUT_DIR, "manager_report_block_all.csv"), index=False, encoding="utf-8-sig")
df_proj_all.to_csv(os.path.join(OUT_DIR, "project_mentions_all.csv"), index=False, encoding="utf-8-sig")




# ========= SUMMARY & LEADERSHIP METRICS =========
summary = []
for meeting, group in df_sentence_all.groupby("Meeting"):
    s_mean = group[group["Speaker"] == "Sarah"]["compound"].mean()
    others = group[group["Speaker"] != "Sarah"]["compound"].mean()
    summary.append(f"{meeting}: Sarah={s_mean:.2f}, Team={others:.2f}, Δ={(s_mean - others):+.2f}")

summary_text = "Leadership Sentiment Summary Across Meetings\n" + "\n".join(summary)
with open(os.path.join(OUT_DIR, "insight_summary_all.txt"), "w", encoding="utf-8") as f:
    f.write(summary_text)

talk_ratio = df_sentence_all["Speaker"].value_counts(normalize=True)
SARAH_TALK_RATIO = talk_ratio.get("Sarah", 0)
empathy = (df_sentence_all[df_sentence_all["Speaker"] == "Sarah"]["pos"].mean()
           - df_sentence_all[df_sentence_all["Speaker"] == "Sarah"]["neg"].mean()) \
           * df_sentence_all[df_sentence_all["Speaker"] == "Sarah"]["Is_Question"].mean()
team_mean = df_sentence_all[df_sentence_all["Speaker"] != "Sarah"]["compound"].mean()

leadership_metrics = {
    "Sarah_Talk_Ratio": round(SARAH_TALK_RATIO, 2),
    "Sarah_Empathy_Index": round(empathy, 3),
    "Team_Positive_Avg": round(team_mean, 3),
    "Meetings_Analyzed": df_sentence_all["Meeting"].nunique(),
    "Total_Sentences": len(df_sentence_all)
}
pd.DataFrame([leadership_metrics]).to_csv(os.path.join(OUT_DIR, "leadership_kpis.csv"), index=False, encoding="utf-8-sig")

# ========= COMMUNICATION EFFECTIVENESS METRICS =========
print("\n🧠 Computing Communication Effectiveness Metrics...")

# 1️⃣ Sarah 讲话后下属确认的比例（Clarity Rate）
confirmation_keywords = [
    "got it", "i understand", "makes sense", "ok", "okay", "sure", "will do", "sounds good", "understood"
]
clarity_count = 0
total_sarah_sentences = len(df_sentence_all[df_sentence_all["Speaker"] == "Sarah"])

# 遍历所有 Sarah 的句子，检测下一句是否为下属且含确认关键词
for i in range(len(df_sentence_all) - 1):
    row = df_sentence_all.iloc[i]
    next_row = df_sentence_all.iloc[i + 1]
    if row["Speaker"] == "Sarah" and next_row["Speaker"] in ["Alex", "Javier"]:
        if any(kw in next_row["Sentence"].lower() for kw in confirmation_keywords):
            clarity_count += 1

Sarah_Clarity_Rate = clarity_count / total_sarah_sentences if total_sarah_sentences > 0 else 0

# 2️⃣ 下属表达疑惑的比例（Unclear Rate）
unclear_keywords = [
    "not sure", "don't understand", "confused", "unclear", "what do you mean", "could you clarify"
]
unclear_count = sum(
    df_sentence_all[df_sentence_all["Speaker"].isin(["Alex", "Javier"])]["Sentence"]
    .str.lower().apply(lambda x: any(kw in x for kw in unclear_keywords))
)
total_team_sentences = len(df_sentence_all[df_sentence_all["Speaker"].isin(["Alex", "Javier"])])
Team_Unclear_Rate = unclear_count / total_team_sentences if total_team_sentences > 0 else 0

# 3️⃣ 项目主动提及分布
proj_by_manager = len(df_proj_all[df_proj_all["Owner"] == "Manager"])
proj_by_team = len(df_proj_all[df_proj_all["Owner"] == "DirectReport"])
total_proj = proj_by_manager + proj_by_team if (proj_by_manager + proj_by_team) > 0 else 1
Sarah_Project_Initiation_Rate = proj_by_manager / total_proj

# 4️⃣ 汇总至同一 leadership_metrics 结构
leadership_metrics.update({
    "Sarah_Clarity_Rate": round(Sarah_Clarity_Rate, 3),
    "Team_Unclear_Rate": round(Team_Unclear_Rate, 3),
    "Sarah_Project_Initiation_Rate": round(Sarah_Project_Initiation_Rate, 3),
    "Projects_By_Sarah": proj_by_manager,
    "Projects_By_Team": proj_by_team
})

print("Added new communication effectiveness KPIs:")
for k, v in leadership_metrics.items():
    print(f"  {k}: {v}")

# 重新保存扩展后的指标
pd.DataFrame([leadership_metrics]).to_csv(os.path.join(OUT_DIR, "leadership_kpis.csv"), index=False, encoding="utf-8-sig")

# ========= DONE =========
print("\n✅ All meetings processed successfully!")
print(f"Input:  {IN_DIR}")
print(f"Output: {OUT_DIR}")
print("Files generated:")
print(" - manager_report_sentence_all.csv")
print(" - manager_report_block_all.csv")
print(" - project_mentions_all.csv")
print(" - insight_summary_all.txt")
print(" - leadership_kpis.csv")

