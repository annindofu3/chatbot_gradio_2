import gradio as gr
from model import prompt, retriever, qa

## チャットボットの作成 ##
def add_text(history, text):
    # 入力されたテキストを履歴に追加し、テキストボックスをクリアにする
    history = history + [(text, None)]
    return history, ""

def bot(history):
    try:
        # 最後に追加されたユーザーの質問を取得
        query = history[-1][0]
        query = prompt.format(question=query, context=retriever)

        # 質問に基づいて回答を生成
        answer = qa.run(query)

        # 履歴の最新エントリに回答を追加
        history[-1][1] = answer
    except Exception as e:
        # エラーが発生した場合、エラーメッセージを追加
        history[-1][1] = f"エラーが発生しました: {e}"
    return history

# Gradioインターフェースを構築
with gr.Blocks() as demo:
    # チャットボットの表示エリアを作成
    chatbot = gr.Chatbot([], elem_id="chatbot")

    # 行を作成
    with gr.Row():
        # 列を作成
        with gr.Column(scale=0.6):
            # テキストボックスを作成
            txt = gr.Textbox(
                show_label=False,
                placeholder="質問を入力しエンターキーを押してください",
            )

    # テキストボックスにテキストが入力され、エンターが押された時
    # add_text関数を実行し、その後bot関数を実行
    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )

# Gradioインターフェースを起動
demo.launch()