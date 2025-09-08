from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import subprocess
import json
import os
import sys

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session handling

detection_process = None
COUNT_FILE = "detection_counts.json"  # File to store counts

def read_counts():
    """Read detection counts from file."""
    if os.path.exists(COUNT_FILE):
        try:
            with open(COUNT_FILE, "r") as file:
                counts = json.load(file)
                counts["total"] = counts.get("male", 0) + counts.get("female", 0)  # Handle missing keys
                return counts
        except (json.JSONDecodeError, ValueError):  # Handle corrupt JSON
            return {"male": 0, "female": 0, "total": 0}
    return {"male": 0, "female": 0, "total": 0}

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username == "admin" and password == "password":
            session['user'] = username
            return redirect(url_for('detection'))
        else:
            return "Invalid Credentials", 401

    return render_template('login.html')

@app.route('/detection')
def detection():
    if 'user' not in session:
        return redirect(url_for('login'))
    
    counts = read_counts()
    return render_template('detection.html', male_count=counts["male"], female_count=counts["female"], total_count=counts["total"])

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_process
    if detection_process is None or detection_process.poll() is not None:
        python_executable = sys.executable  # Get current Python interpreter path
        detection_process = subprocess.Popen(
            [python_executable, 'run_detection.py'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return jsonify({"status": "Detection Started"})
    return jsonify({"status": "Already Running"})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_process

    if detection_process and detection_process.poll() is None:
        detection_process.terminate()
        detection_process = None

        counts = read_counts()
        return jsonify({"status": "Detection Stopped", "male": counts["male"], "female": counts["female"], "total": counts["total"]})

    return jsonify({"status": "Not Running"})

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
