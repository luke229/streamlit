import streamlit as st  # 導入Streamlit庫，用於建立網頁應用
import ollama  # 導入ollama庫，用於自然語言處理
import chromadb  # 導入chromadb庫，用於數據存儲和查詢
import pandas as pd  # 導入pandas庫，用於數據分析和處理

def initialize():
    if "already_executed" not in st.session_state:
        st.session_state.already_executed = False 
    if not st.session_state.already_executed:
        setup_database() 

def setup_database():
    client = chromadb.Client()  # 創建一個chromadb的客戶端，用於與資料庫交互
    file_path = 'QA50.xlsx'  # 指定Excel文件的路徑和名稱
    documents = pd.read_excel(file_path, header=None)  # 使用pandas讀取Excel文件
    collection = client.get_or_create_collection(name="demodocs")
    for index, content in documents.iterrows():
        response = ollama.embeddings(model="mxbai-embed-large", prompt=content[0])  # 通過ollama生成該行文本的嵌入向量
        collection.add(ids=[str(index)], embeddings=[response["embedding"]], documents=[content[0]])  # 將文本和其嵌入向量添加到集合中

    st.session_state.already_executed = True  # 設置'already_executed'為True，表示已完成初始化
    st.session_state.collection = collection  # 將集合保存在會話狀態中，供後續使用  
    
def create_chromadb_client():
    return chromadb.Client()  # 返回一個新的chromadb客戶端實例

def main():
    initialize()  # 呼叫初始化函數
    st.title("我的第一個LLM+RAG本地知識問答")  # 在網頁應用中設置標題
    user_input = st.text_area("您想問什麼？", "")  # 創建一個文本區域供用戶輸入問題

    if st.button("送出"):
        if user_input:
            handle_user_input(user_input, st.session_state.collection)  # 處理用戶輸入，進行查詢和回答
        else:
            st.warning("請輸入問題！")  # 如果用戶沒有輸入，顯示警告消息
            
def handle_user_input(user_input, collection):
    response = ollama.embeddings(prompt=user_input, model="mxbai-embed-large")  # 生成用戶輸入的嵌入向量
    results = collection.query(query_embeddings=[response["embedding"]], n_results=3)  # 在集合中查詢最相關的三個文檔
    data = results['documents'][0]  # 獲取最相關的文檔
    output = ollama.generate(
        model="jcai/llama3-taide-lx-8b-chat-alpha1:Q4_K_M",
        prompt=f"Using this data: {data}. Respond to this prompt and use chinese: {user_input}"  # 生成回應
    )
    st.text("回答：")  # 顯示"回答："
    st.write(output['response'])  # 將生成的回應顯示在網頁上
    
if _name_ == "__main__":
    main()  # 如果直接執行此文件，則執行main函數