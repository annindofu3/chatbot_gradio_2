# 必要なモジュールをインポート
import os
import openai
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

# APIキーを設定
openai.api_key = os.environ["OPENAI_API_KEY"]

# Excelファイルからデータを読み込む
file_path = 'src/仮FAQ_キカガクアプリ作成用.xlsx'
df = pd.read_excel(file_path, sheet_name='FAQ', engine='openpyxl')
df2 = pd.read_excel(file_path, sheet_name='担当者連絡先', engine='openpyxl')
df3 = pd.read_excel(file_path, sheet_name='代理店一覧', engine='openpyxl')
df4 = pd.read_excel(file_path, sheet_name='資料リンク', engine='openpyxl')

## データの加工 ##
# FAQシートについて、質問と回答のリストを作る
# 後にプロンプトに入れ込むことを想定して構造化しておく
doc_list = ("###質問\n" + df['Q'].astype(str) + "\n\n###回答\n" + df['A'].astype(str)).tolist()
print(doc_list[0])

# FAQシートについて、C列をメタデータとして取り込むため辞書形式にする
df_meta = df[['関連テーブルデータ']].to_dict(orient='records')

# ドキュメントのリストを作成する
documents = []
for i, doc in enumerate(doc_list):
    metadata = df_meta[i]
    documents.append(Document(page_content=doc, metadata=metadata))

# df2をマークダウンの形式(文字列型)に変換
df2_no_markdown = df2.to_markdown()

# df3をマークダウンの形式(文字列型)に変換
df3_no_markdown = df3.to_markdown()

# df4をマークダウンの形式(文字列型)に変換
df4_no_markdown = df4.to_markdown()

## QAシステムの構築 ##
# Embeddingsを作成する
embeddings = OpenAIEmbeddings()

# ベクトルデータベースを作成する
vectordb = Chroma.from_documents(documents, embeddings)

# プロンプトの定義
from langchain import PromptTemplate

template = """
あなたは元気な社内アシスタントです。下記の質問に日本語で優しく元気に回答してください。

##質問
{{question}}

回答を生成する際の参考情報として、上記の質問に対して社内データから類似のFAQを検索した結果が以下のとおり得られています。
また、担当者の連絡先、代理店情報、資料リンクを markdown 形式の表で以下に記載しています。
回答の内容に担当者情報、代理店情報、資料リンクを含める必要がある場合は、表そのものや担当者メールアドレス、資料リンクも含めた形で回答しなさい。

##検索結果
{{context}}

##担当者情報
{tantou}

##代理店情報
{dairiten}

##資料リンク
{link}
""".format(tantou=df2_no_markdown, dairiten=df3_no_markdown,link=df4_no_markdown)

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=template,
)

# QAシステムを構築する
retriever = vectordb.as_retriever(search_kwargs={"k": min(len(documents), 4)})
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
    chain_type='stuff',
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt}
)