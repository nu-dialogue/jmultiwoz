INTERFACE_HTML = """
<html>
    <link rel="stylesheet" href={stylesheet_href} />
    <style>
    .balloon1-left {{
      position: relative;
      display: inline-block;
      margin: 0.3em 0 0.3em 15px;
      padding: 7px 10px;
      min-width: 120px;
      max-width: 100%;
      color: #555;
      font-size: 16px;
      background: lightgray;
      border-radius: 5px 5px 5px 5px;
    }}

    .balloon1-left:before {{
      content: "";
      position: absolute;
      top: 50%;
      left: -30px;
      margin-top: -15px;
      border: 15px solid transparent;
      border-right: 15px solid lightgray;
    }}

    .balloon1-left p {{
      margin: 0;
      padding: 0;
    }}

    .balloon1-right {{
      position: relative;
      display: inline-block;
      margin: 0.3em 15px 0.3em 0;
      padding: 7px 10px;
      min-width: 120px;
      max-width: 100%;
      color: #555;
      font-size: 16px;
      background: rgba(144,238,144,0.7);
      border-radius: 5px 5px 5px 5px;
    }}

    .balloon1-right:before {{
      content: "";
      position: absolute;
      top: 50%;
      left: 100%;
      margin-top: -15px;
      border: 15px solid transparent;
      border-left: 15px solid rgba(144,238,144,0.7);
    }}

    .balloon1-right p {{
      margin: 0;
      padding: 0;
    }}

    .media {{
    align-items: flex-end;
    }}

    </style>
    <script defer src={font_src}></script>
    <head>
        <title>対話ページ</title>
    </head>
    <body>
        <div class="columns" style="height: 100%">
            <div class="column is-three-fifths is-offset-one-fifth">
            <section class="hero is-info is-large has-background-light has-text-grey-dark" style="height: 100%">
                <div id="parent" class="hero-body" style="overflow: auto; height: calc(100% - 76px); padding-top: 1em; padding-bottom: 0;">
                    <article class="media">
                    <div class="media-content">
                        <div class="content">
                        <p>
                            <strong>インストラクション</strong>
                            <br>
                            {instruction}
                        </p>
                        </div>
                    </div>
                    </article>
                </div>
                <div class="hero-foot column is-three-fifths is-offset-one-fifth" style="height: 76px">
                <form id = "interact">
                    <div class="field is-grouped">
                        <p class="control is-expanded">
                        <input class="input" type="text" id="userIn" placeholder="メッセージを入力してください" required>
                        </p>
                        <p class="control">
                        <button id="respond" type="submit" class="button has-text-white-ter has-background-grey-dark">
                            送信
                        </button>
                        </p>
                    </div>
                </form>
                </div>
            </section>
            </div>
        </div>
        <script>
            var sessionId = "{session_id}";
            var outfile = sessionId + ".csv";
            var turn = 0;
            var sessionOver = false;
            var botResponseDelay = 3000;
            // document.getElementById("turn").innerHTML = maxTurns;
            // document.getElementById("turn-half").innerHTML = Math.trunc(maxTurns / 2);
            var finishEval = false;

            window.onload = function(){{
                document.onkeypress = enterForbidden;

                function enterForbidden(){{
                    if(window.event.keyCode == 13){{
                        return false;
                    }}
                }}
            }}

            function createChatRow(agent, text) {{
                var article = document.createElement("article");
                article.className = "media"
                var figure = document.createElement("figure");
                figure.className = "media-right";
                if (agent == "You") {{
                    figure.className = "media-right";
                }}
                else {{
                    figure.className = "media-left";
                }}
                var span = document.createElement("span");
                span.className = "icon is-large";
                var icon = document.createElement("i");
                icon.className = "fas fas fa-2x" + (agent === "You" ? " fa-user " : agent === "Model" ? " fa-robot" :  agent === "System" ? " fa-robot" : "");
                var media = document.createElement("div");
                media.className = "media-content";
                var content = document.createElement("div");
                content.className = "content";
                if (agent == "You") {{
                    content.classList.add("has-text-right");
                }}

                var para2 = document.createElement("p");
                var turn_index = turn + 1;
                if (agent !== "System") {{
                    text = "[" + turn_index + "] " + text;
                }}
                var paraText = document.createTextNode(text);
                para2.className = "balloon1-" + (agent === "You" ? "right" : agent === "Model" ? "left" : agent === "System" ? "left"  : "");
                para2.classList.add("has-text-left");

                var para1 = document.createElement("p");
                var strong = document.createElement("strong");
                strong.innerHTML = (agent === "You" ? "あなた" : agent === "Model" ? "ボット" :  agent === "System" ? " システムアナウンス" : agent);
                var br = document.createElement("br");

                para1.appendChild(strong);
                para2.appendChild(paraText);
                content.appendChild(para1);
                content.appendChild(para2);
                media.appendChild(content);

                // article.appendChild(media);
                span.appendChild(icon);
                figure.appendChild(span);

                if (agent == "You") {{
                    article.appendChild(media);
                    article.appendChild(figure);
                }}
                else {{
                    article.appendChild(figure);
                    article.appendChild(media);
                }}
                return article;
            }}

            var context = [];

            // var myContext = [];
            // myContext.push({{"spk": "[SPK1]", "utt": "こんにちは。よろしくお願いします。"}})

            function exportCSV() {{
                var evalValues = [];
                var messageArea = document.getElementById('messageArea');
                for (var i = 0; i < 3; i++) {{
                    var name = 'q' + String(i+1);
                    var elements = document.getElementsByName(name);
                    for (var j = 0; j < elements.length; j++) {{
                        if (elements.item(j).checked) {{
                            evalValues.push(elements.item(j).value);
                        }}
                    }}
                }}
                if (evalValues.length != 3) {{
                    alert("全ての設問に回答してください");
                    return;
                }}
                var csvData = "";
                // csvData += `HHB_O_${{String(maxOperatorUtterance)}}` + "\\r\\n";
                for (var i = 0; i < context.length; i++) {{
                    csvData += "" + context[i]["Talker"] + ","
                        + context[i]["Uttr"] + "\\r\\n";
                }}

                for (var i = 0; i < evalValues.length; i++) {{
                    csvData += "" + evalValues[i] + ",";
                }}
                // csvData += "\\r\\n" + messageArea.value;

                const link = document.createElement("a");
                document.body.appendChild(link);
                link.style = "display:none";
                const blob = new Blob([csvData], {{ type: "octet/stream" }});
                const url = window.URL.createObjectURL(blob);
                link.href = url;
                link.download = outfile;
                link.click();
                window.URL.revokeObjectURL(url);
                link.parentNode.removeChild(link);

                // setTimeout(() => {{
                // peer.destroy();
                // window.location.href = '/finish';
                // }}, 3000);
            }}

            var parDiv = document.getElementById("parent");
            parDiv.scrollTo(0, parDiv.scrollHeight);

            change_input_state(true);

            document.getElementById("interact").addEventListener("submit", function(event){{
                event.preventDefault()

                userInput = document.getElementById("userIn").value;
                if (userInput.length == 0) {{
                    alert("発話が入力されていません。");
                    return;
                }}
                var ncontext = {{
                  "Talker": "U",
                 "Uttr":   userInput
                }};
                context.push(ncontext);

                document.getElementById('userIn').value = "";
                var parDiv = document.getElementById("parent");
                parDiv.append(createChatRow("You", userInput));
                parDiv.scrollTo(0, parDiv.scrollHeight);
                var send_info = {{"userInput": userInput, "sessionId": sessionId}};

                turn += 1;

                // setTimeout(sendUtterance, botResponseDelay, send_info);
                sendUtterance(send_info);

                if (sessionOver) {{
                    finishDialogue(parDiv, 500);
                }}
                else {{
                    change_input_state(false);
                }}
            }});

            function addBotUtterance(data) {{
                var parDiv = document.getElementById("parent");
                parDiv.append(createChatRow("Model", data));
                parDiv.scrollTo(0, parDiv.scrollHeight);
                change_input_state(true);
                context.push({{"Talker": "S", "Uttr": data}});
                // myContext.push({{"spk": "[SPK1]", "utt": data}});

                turn += 1;

                if (sessionOver) {{
                    finishDialogue(parDiv);
                }}
            }}

            function change_input_state(state) {{
                inp = document.getElementById("userIn");
                submit_btn = document.getElementById("respond");
                if (state) {{
                    inp.placeholder = "発話を入力してください";
                    inp.disabled = false;
                    submit_btn.disabled = false;
                }}
                else {{
                    inp.placeholder = "入力できません";
                    inp.disabled = true;
                    submit_btn.disabled = true;
                }}
            }}

            function sendUtterance(send_info) {{
                fetch('/interact', {{
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    method: 'POST',
                    body: JSON.stringify(send_info)
                }}).then(response=>response.json()).then(data=>{{
                    // myContext.push({{"spk": "[SPK2]", "utt": send_info["utt"]}});
                    var botResponse = processBotUtterance(data.text);
                    sessionOver = data.sessionOver;
                    // botResponseDelay = Math.min(Math.max(10, botResponse.length), 30) * 1000;
                    botResponseDelay = 0;
                    setTimeout(addBotUtterance, botResponseDelay, botResponse);
                }})
            }}

            function processBotUtterance(text) {{
              var utt = text.replace("，", "、");
              utt = utt.replace(",", "、");
              utt = utt.replace("．", "。");
              utt = utt.replace(".", "。");
              utt = utt.replace("!", "！");
              utt = utt.replace("?", "？");
              return utt;
            }}

            function finishDialogue(parDiv, delay=0) {{
                parDiv.append(createChatRow("System", "これにて対話は終了です。下の「評価開始」ボタンより対話の評価を行ってください。"));
                document.getElementById("userIn").remove();
                parDiv.scrollTo(0, parDiv.scrollHeight);
                setTimeout(() => {{
                    document.getElementById("respond").setAttribute("type", "reset");
                    document.getElementById("respond").textContent = "評価開始";
                }}, delay);
            }}

            function createEvalForm() {{
                var parDiv = document.getElementById("parent");
                var questionList = ["1: 対話を通して、ボットの対話の流れはスムーズだった", "2: 対話を通して、ボットは十分な情報を提示していた", "3: 対話を通して、自分は対話に満足した"]
                var answerList = ["1. 同意しない", "2. やや同意しない", "3. どちらでもない", "4. やや同意する", "5. 同意する"]
                for (var j=0; j<3; j++) {{
                    var article = document.createElement("article");
                    article.className = "media";
                    var media = document.createElement("div");
                    media.className = "media-content";
                    var content = document.createElement("div");
                    content.className = "content";
                    var q = document.createElement("p");
                    q.innerHTML = `<strong>${{questionList[j]}}</strong>`
                    content.appendChild(q);
                    for (var i=0; i<5; i++) {{
                        var radio = document.createElement("label");
                        radio.style.display = 'block';
                        radio.style.padding = '5px';
                        radio.innerHTML = `<input type="radio" name=${{"q" + String(j+1)}} value=${{String(i+1)}}>${{answerList[i]}}`
                        content.appendChild(radio);
                    }}
                    media.appendChild(content);
                    article.appendChild(media);
                    parDiv.append(article);
                }}
                parDiv.append(createChatRow("System", "全ての設問に回答したら、下の「評価結果をダウンロード」ボタンをクリックしてください。"));
                parDiv.scrollTo(0, parDiv.scrollHeight);
            }}

            document.getElementById("interact").addEventListener("reset", function(event){{
                event.preventDefault();
                if (finishEval) {{
                    exportCSV();
                }}
                else {{
                    createEvalForm();
                    document.getElementById("respond").textContent = "評価結果をダウンロード";
                    setTimeout(() => {{
                        finishEval = true;
                    }}, 3000)
                }}
            }});
        </script>
    </style>
    </body>
</html>
"""