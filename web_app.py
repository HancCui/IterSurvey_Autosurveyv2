from flask import Flask, render_template_string, request, Response, send_file
import subprocess
import os
import re
import glob

app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>IterSurvey</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 30px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .header h1 {
            color: white;
            font-size: 2.5em;
            font-weight: 700;
            text-shadow: 0 2px 10px rgba(0,0,0,0.2);
        }
        .header p {
            color: rgba(255,255,255,0.85);
            font-size: 1.1em;
            margin-top: 8px;
        }
        .container {
            display: flex;
            gap: 25px;
            max-width: 1500px;
            margin: 0 auto;
        }
        .panel {
            background: white;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        .left { flex: 1; min-width: 380px; }
        .right { flex: 2; display: flex; flex-direction: column; }
        .panel-header {
            background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
            color: white;
            padding: 20px 25px;
            font-size: 1.2em;
            font-weight: 600;
        }
        .panel-body { padding: 25px; }
        .section-title {
            font-size: 0.85em;
            font-weight: 600;
            color: #6b46c1;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin: 20px 0 15px 0;
            padding-bottom: 8px;
            border-bottom: 2px solid #e9d8fd;
        }
        .section-title:first-child { margin-top: 0; }
        .form-group { margin-bottom: 16px; }
        .form-group label {
            display: block;
            font-size: 0.9em;
            font-weight: 500;
            color: #4a5568;
            margin-bottom: 6px;
        }
        .form-group label.required::after {
            content: " *";
            color: #e53e3e;
        }
        .form-group input {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-size: 0.95em;
            transition: all 0.2s;
            background: #f7fafc;
        }
        .form-group input:focus {
            outline: none;
            border-color: #6b46c1;
            background: white;
            box-shadow: 0 0 0 3px rgba(107,70,193,0.1);
        }
        .form-group input::placeholder { color: #a0aec0; }
        .accordion {
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            margin-bottom: 12px;
            overflow: hidden;
        }
        .accordion-header {
            background: #f7fafc;
            padding: 14px 18px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 500;
            color: #4a5568;
            transition: all 0.2s;
        }
        .accordion-header:hover { background: #edf2f7; }
        .accordion-header::after {
            content: "▼";
            font-size: 0.7em;
            transition: transform 0.3s;
        }
        .accordion.open .accordion-header::after { transform: rotate(180deg); }
        .accordion-body {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
            background: white;
        }
        .accordion.open .accordion-body { max-height: 500px; }
        .accordion-content { padding: 18px; }
        .btn {
            width: 100%;
            padding: 16px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 12px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            margin-top: 25px;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(102,126,234,0.4);
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102,126,234,0.5);
        }
        .btn:disabled {
            background: #a0aec0;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .btn:disabled::after {
            content: "";
            display: inline-block;
            width: 16px;
            height: 16px;
            border: 2px solid white;
            border-top-color: transparent;
            border-radius: 50%;
            margin-left: 10px;
            animation: spin 1s linear infinite;
            vertical-align: middle;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        #logs {
            flex: 1;
            background: #1a1a2e;
            color: #0ff;
            font-family: "SF Mono", "Fira Code", "Consolas", monospace;
            font-size: 13px;
            padding: 20px;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
            min-height: 500px;
            line-height: 1.6;
        }
        #logs::-webkit-scrollbar { width: 8px; }
        #logs::-webkit-scrollbar-track { background: #16162a; }
        #logs::-webkit-scrollbar-thumb { background: #4a5568; border-radius: 4px; }
        .download-bar {
            display: none;
            background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
            padding: 18px 25px;
            text-align: center;
        }
        .download-bar a {
            color: white;
            text-decoration: none;
            font-weight: 600;
            font-size: 1.1em;
            display: inline-flex;
            align-items: center;
            gap: 10px;
        }
        .download-bar a:hover { text-decoration: underline; }
        .download-bar svg { width: 24px; height: 24px; }
        .status {
            background: #1a1a2e;
            color: #a0aec0;
            padding: 12px 20px;
            font-size: 0.85em;
            border-top: 1px solid #2d3748;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #48bb78;
        }
        .status-dot.running { animation: pulse 1.5s infinite; background: #ecc94b; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>IterSurvey</h1>
        <p>Automatic Literature Survey Generator powered by LLM</p>
    </div>

    <div class="container">
        <div class="panel left">
            <div class="panel-header">Configuration</div>
            <div class="panel-body">
                <div class="section-title">Required Parameters</div>

                <div class="form-group">
                    <label class="required">Topic</label>
                    <input type="text" id="topic" placeholder="e.g., LLMs for Education">
                </div>

                <div class="form-group">
                    <label class="required">API Key</label>
                    <input type="password" id="api_key" placeholder="sk-xxxxxxxx">
                </div>

                <div class="form-group">
                    <label class="required">Embedding Model</label>
                    <input type="text" id="embedding_model" value="nomic-ai/nomic-embed-text-v1.5">
                </div>

                <div class="section-title">Optional Parameters</div>

                <div class="accordion" onclick="this.classList.toggle('open')">
                    <div class="accordion-header">Basic Settings</div>
                    <div class="accordion-body">
                        <div class="accordion-content">
                            <div class="form-group">
                                <label>Description</label>
                                <input type="text" id="description" placeholder="Additional description for the topic">
                            </div>
                            <div class="form-group">
                                <label>Model</label>
                                <input type="text" id="model" value="gpt-4o-2024-05-13">
                            </div>
                            <div class="form-group">
                                <label>API URL</label>
                                <input type="text" id="api_url" placeholder="https://api.openai.com/v1/chat/completions">
                            </div>
                        </div>
                    </div>
                </div>

                <div class="accordion" onclick="this.classList.toggle('open')">
                    <div class="accordion-header">Survey Structure</div>
                    <div class="accordion-body">
                        <div class="accordion-content">
                            <div class="form-group">
                                <label>Section Number</label>
                                <input type="number" id="section_num" value="8">
                            </div>
                            <div class="form-group">
                                <label>Subsection Length</label>
                                <input type="number" id="subsection_len" value="700">
                            </div>
                            <div class="form-group">
                                <label>RAG Number</label>
                                <input type="number" id="rag_num" value="60">
                            </div>
                        </div>
                    </div>
                </div>

                <div class="accordion" onclick="this.classList.toggle('open')">
                    <div class="accordion-header">System Settings</div>
                    <div class="accordion-body">
                        <div class="accordion-content">
                            <div class="form-group">
                                <label>GPU</label>
                                <input type="text" id="gpu" value="0">
                            </div>
                            <div class="form-group">
                                <label>Saving Path</label>
                                <input type="text" id="saving_path" value="./output">
                            </div>
                            <div class="form-group">
                                <label>Database Path</label>
                                <input type="text" id="db_path" value="./database">
                            </div>
                        </div>
                    </div>
                </div>

                <button class="btn" id="btn" onclick="generate()">Generate Survey</button>
            </div>
        </div>

        <div class="panel right">
            <div class="panel-header">Output Logs</div>
            <div id="logs">Waiting for task...\n</div>
            <div class="download-bar" id="download">
                <a href="#" id="pdf_link">
                    <svg fill="currentColor" viewBox="0 0 20 20"><path d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z"/></svg>
                    Download PDF
                </a>
            </div>
            <div class="status">
                <div class="status-dot" id="status-dot"></div>
                <span id="status-text">Ready</span>
            </div>
        </div>
    </div>

    <script>
        function generate() {
            var btn = document.getElementById('btn');
            var logs = document.getElementById('logs');
            var statusDot = document.getElementById('status-dot');
            var statusText = document.getElementById('status-text');

            btn.disabled = true;
            btn.textContent = 'Generating';
            logs.textContent = '';
            document.getElementById('download').style.display = 'none';
            statusDot.classList.add('running');
            statusText.textContent = 'Running...';

            var params = new URLSearchParams({
                topic: document.getElementById('topic').value,
                api_key: document.getElementById('api_key').value,
                embedding_model: document.getElementById('embedding_model').value,
                description: document.getElementById('description').value,
                model: document.getElementById('model').value,
                api_url: document.getElementById('api_url').value,
                section_num: document.getElementById('section_num').value,
                subsection_len: document.getElementById('subsection_len').value,
                rag_num: document.getElementById('rag_num').value,
                gpu: document.getElementById('gpu').value,
                saving_path: document.getElementById('saving_path').value,
                db_path: document.getElementById('db_path').value
            });

            var evtSource = new EventSource('/run?' + params.toString());

            evtSource.onmessage = function(e) {
                var data = JSON.parse(e.data);
                if (data.log) {
                    logs.textContent += data.log;
                    logs.scrollTop = logs.scrollHeight;
                }
                if (data.done) {
                    btn.disabled = false;
                    btn.textContent = 'Generate Survey';
                    statusDot.classList.remove('running');
                    evtSource.close();
                    if (data.pdf) {
                        document.getElementById('download').style.display = 'block';
                        document.getElementById('pdf_link').href = '/download?path=' + encodeURIComponent(data.pdf);
                        statusText.textContent = 'Completed - PDF Ready';
                    } else {
                        statusText.textContent = 'Completed';
                    }
                }
            };

            evtSource.onerror = function() {
                btn.disabled = false;
                btn.textContent = 'Generate Survey';
                statusDot.classList.remove('running');
                statusText.textContent = 'Error occurred';
                evtSource.close();
            };
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/run')
def run():
    def generate():
        topic = request.args.get('topic', '')
        api_key = request.args.get('api_key', '')
        embedding_model = request.args.get('embedding_model', '')

        if not topic or not api_key or not embedding_model:
            yield f"data: {{\"log\": \"Error: Topic, API Key, Embedding Model are required\\n\", \"done\": true}}\n\n"
            return

        cmd = ['python', 'main.py']
        cmd.extend(['--topic', topic])
        cmd.extend(['--api_key', api_key])
        cmd.extend(['--embedding_model', embedding_model])

        for key in ['description', 'model', 'api_url', 'section_num', 'subsection_len', 'rag_num', 'gpu', 'saving_path', 'db_path']:
            val = request.args.get(key, '')
            if val:
                cmd.extend([f'--{key}', val])

        yield f"data: {{\"log\": \"[Start] python main.py --topic \\\"{topic}\\\" ...\\n\\n\"}}\n\n"

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        for line in iter(process.stdout.readline, ''):
            escaped = line.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
            yield f"data: {{\"log\": \"{escaped}\"}}\n\n"

        process.wait()

        pdf_path = None
        if process.returncode == 0:
            topic_clean = re.sub(r'[^\w\s-]', '', topic.strip())
            topic_str = re.sub(r'\s+', '_', topic_clean)
            model_str = request.args.get('model', 'gpt-4o').replace(' ', '_')
            save_dir = request.args.get('saving_path', './output')

            pattern = os.path.join(save_dir, f"{topic_str}_{model_str}_*", "latex", "main.pdf")
            pdfs = glob.glob(pattern)
            if pdfs:
                pdf_path = max(pdfs, key=os.path.getctime)

        if pdf_path:
            yield f"data: {{\"log\": \"\\n[Done] PDF generated: {pdf_path}\\n\", \"done\": true, \"pdf\": \"{pdf_path}\"}}\n\n"
        else:
            yield f"data: {{\"log\": \"\\n[Finished] Exit code: {process.returncode}\\n\", \"done\": true}}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/download')
def download():
    path = request.args.get('path', '')
    if path and os.path.exists(path):
        return send_file(path, as_attachment=True)
    return 'File not found', 404

if __name__ == '__main__':
    os.makedirs('./database', exist_ok=True)
    os.makedirs('./output', exist_ok=True)
    print("Starting IterSurvey Web UI...")
    print("Open http://localhost:7860 in your browser")
    app.run(host='0.0.0.0', port=7860, threaded=True)
