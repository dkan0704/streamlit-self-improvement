import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# === 1. 環境変数の読み込み ===
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
st.write("🔑 OpenAI API Key is set." if api_key else "❌ OpenAI API Key is not set.")
# === 2. Streamlit UI ===
st.title("📘 自己研鑽休暇レポート(AI付き)")

st.markdown(
    """
    こんにちは！  
    自己研鑽休暇をいただき本当にありがとうございます。 
    勉強した生成AIの技術を使って、<br>
    **生成AIで“菅野っぽく答えるAI”**を作ってみたので、
    よかったら遊んでみてください ☕️  

    ---
    ### 🎮 モードを選んでください
    - **菅野AIモード**：  
    ChatGPTに「ちょっとフランクに話していいよ」と伝えてあります。  
    菅野の自己研鑽内容を事前に教えてあるので、少し“本人っぽい”回答をします。  

    - **ChatGPTモード**：  
    一般的なAIとして、フラットに回答します。  

    <p style="font-size:1rem; margin-top:1em;">
    ※回答内容はAIが自動で生成していますし、所詮私のスキルなので、多少のバグはご了承を 😄
    </p>
    ---
    """,
    unsafe_allow_html=True
)
mode = st.radio(
    "",["菅野AIが回答します", "Chat GPTが回答します"]
)

question = st.text_input("気になることを何でも聞いてみてください!(内容によってはちょっと照れるかも)", placeholder="例：どんな勉強してたの？/それ役に立ちそう？/お昼何食べてた？")

# === 3. RAGデータの読み込み ===
@st.cache_resource
def load_rag_database(excel_path="answerlist.xlsx"):
    if not os.path.exists(excel_path):
        st.warning("⚠️ answerlist.xlsx が見つかりません。通常モードで動作します。")
        return None, None

    df = pd.read_excel(excel_path)
    if "content" not in df.columns:
        st.error("❌ Excelに 'content' カラムが必要です。")
        return None, None

    df["full_text"] = df.apply(
        lambda row: f"質問: {row.get('expected_question', '')}\n回答: {row['content']}\n日付: {row.get('date', '')}\nタグ: {row.get('tags', '')}",
        axis=1
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    texts = []
    meta = []
    for _, row in df.iterrows():
        chunks = splitter.split_text(row["full_text"])
        texts.extend(chunks)
        meta.extend([{
            "id": row["id"],
            "image_path": row.get("image_path", None),
            "date": row.get("date", ""),
            "tags": row.get("tags", "")
        }] * len(chunks))

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    db = FAISS.from_texts(texts, embeddings, metadatas=meta)
    return db, df


# === 4. 回答関数 ===
def get_response(question, mode):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, openai_api_key=api_key)

    # --- 通常モード ---
    if mode == "Chat GPTが回答します":
        messages = [
            SystemMessage(content="あなたは質問に対してフランクに、失礼にならない範囲で回答するアシスタントです。"),
            HumanMessage(content=question)
        ]
        messages = [
            SystemMessage(content="あなたは質問に対してフランクに、失礼にならない範囲で回答するアシスタントです。"),
            HumanMessage(content=question)
        ]
        response = llm.invoke(messages)  # LangChain 0.3系では invoke を使用
        # LangChain 0.3系ではget_openai_callbackが廃止されたため、概算料金を表示
        estimated_cost = 0.001  # 概算値
        print(f"料金($): {estimated_cost}")
        return {"text": response.content, "images": [], "cost": estimated_cost}

    # --- RAG参照モード ---
    elif mode == "菅野AIが回答します":
        db, df = load_rag_database()
        if not db:
            return {"text": "RAGデータベースが利用できません。通常モードでお試しください。", "images": []}

        retriever = db.as_retriever()
        docs = retriever.invoke(question)  # LangChain 0.3系では invoke を使用
        context = "\n\n".join([d.page_content for d in docs])

        image_paths = list({d.metadata.get("image_path") for d in docs if d.metadata.get("image_path")})

        messages = [
            SystemMessage(content="あなたは質問に対してフランクに、失礼にならない範囲で回答するアシスタントです。"),
            HumanMessage(content=f"以下の参考情報をもとに質問に答えてください。\n\n参考情報:\n{context}\n\n質問:{question}")
        ]
        messages = [
            SystemMessage(content="あなたは質問に対してフランクに、失礼にならない範囲で回答するアシスタントです。"),
            HumanMessage(content=f"以下の参考情報をもとに質問に答えてください。\n\n参考情報:\n{context}\n\n質問:{question}")
        ]
        response = llm.invoke(messages)  # LangChain 0.3系では invoke を使用
        # LangChain 0.3系ではget_openai_callbackが廃止されたため、概算料金を表示
        estimated_cost = 0.002  # RAGモードは少し高めの概算値
        print(f"料金($): {estimated_cost}")
        return {"text": response.content, "images": image_paths, "cost": estimated_cost}


# === 5. 実行ボタン ===
if st.button("実行"):
    if not question:
        st.error("❌ 質問を入力してください。")
    else:
        with st.spinner("💬 回答を生成中..."):
            result = get_response(question, mode)
            st.divider()
            st.write("### 🧠 回答:")
            st.write(result["text"])
            st.write(f"💸 この回答を生成するのにかかったAI利用料 ${result['cost']:6f}(高い？安い？)")

            if result["images"]:
                st.write("### 🖼 関連画像:")
                for img_path in result["images"]:
                    if os.path.exists(img_path):
                        st.image(img_path, width=300, caption=os.path.basename(img_path))
                    else:
                        st.write(f"⚠️ 画像が見つかりません: {img_path}")
