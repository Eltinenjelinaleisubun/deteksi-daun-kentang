from flask import (
    Flask,
    render_template,
    request,
    redirect,
    session,
    url_for,
    flash
)

import os
import uuid
import json
import sqlite3
import numpy as np
import random
import smtplib
import tensorflow as tf

from email.mime.text import MIMEText

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =====================================================
# REDUCE TENSORFLOW LOG
# =====================================================

tf.get_logger().setLevel('ERROR')

# =====================================================
# CONFIG
# =====================================================

class Config:

    # ================= APP =================
    SECRET_KEY = 'secretkey'

    # ================= DATABASE =================
    DATABASE = 'database.db'

    # ================= MODEL =================
    MODEL_PATH = 'model/mobilenetv2_model.h5'

    # ================= UPLOAD =================
    UPLOAD_FOLDER = 'static/uploads'

    MAX_CONTENT_LENGTH = 5 * 1024 * 1024

    ALLOWED_EXTENSIONS = {
        'png',
        'jpg',
        'jpeg',
        'webp'
    }

    # ================= EMAIL =================
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587

    # GANTI DENGAN EMAIL KAMU
    MAIL_USERNAME = 'emailkamu@gmail.com'

    # GANTI DENGAN PASSWORD APLIKASI
    MAIL_PASSWORD = 'password_aplikasi_gmail'


# =====================================================
# APP
# =====================================================

app = Flask(__name__)

app.config.from_object(Config)

print("FLASK APP START")

# =====================================================
# EMAIL CONFIG
# =====================================================

EMAIL_ADDRESS = app.config['MAIL_USERNAME']
EMAIL_PASSWORD = app.config['MAIL_PASSWORD']

# =====================================================
# CREATE FOLDER
# =====================================================

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('model', exist_ok=True)

# =====================================================
# DATABASE CONNECTION
# =====================================================

def get_db_connection():

    conn = sqlite3.connect(app.config['DATABASE'])

    conn.row_factory = sqlite3.Row

    return conn

# =====================================================
# INIT DATABASE
# =====================================================

def init_db():

    conn = sqlite3.connect(app.config['DATABASE'])

    c = conn.cursor()

    # USERS
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            email TEXT
        )
    ''')

    # HISTORY
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            result TEXT,
            image TEXT,
            confidence REAL
        )
    ''')

    conn.commit()

    conn.close()

init_db()

print("DATABASE READY")

# =====================================================
# LOAD MODEL
# =====================================================

model = None

try:

    if os.path.exists(app.config['MODEL_PATH']):

        print("Mulai load model...")

        model = load_model(app.config['MODEL_PATH'])

        print("MODEL BERHASIL DIMUAT")

    else:

        print("MODEL TIDAK DITEMUKAN")

except Exception as e:

    print("ERROR LOAD MODEL:")
    print(e)

    model = None

# =====================================================
# LOAD LABEL
# =====================================================

labels = {}

try:

    json_path = 'model/class_indices.json'

    if os.path.exists(json_path):

        with open(json_path) as f:

            class_indices = json.load(f)

        labels = dict((v, k) for k, v in class_indices.items())

        print("LABEL BERHASIL DIMUAT")

    else:

        print("class_indices.json tidak ditemukan")

except Exception as e:

    print("ERROR LOAD LABEL:")
    print(e)

# =====================================================
# HELPER
# =====================================================

def allowed_file(filename):

    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower()
        in app.config['ALLOWED_EXTENSIONS']
    )

# =====================================================
# SEND OTP EMAIL
# =====================================================

def send_otp(email, otp):

    subject = "Kode OTP Reset Password"

    body = f"""
Kode OTP kamu adalah:

{otp}

Jangan berikan kode ini kepada siapa pun.
"""

    msg = MIMEText(body)

    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = email

    server = smtplib.SMTP(
        app.config['MAIL_SERVER'],
        app.config['MAIL_PORT']
    )

    server.starttls()

    server.login(
        EMAIL_ADDRESS,
        EMAIL_PASSWORD
    )

    server.send_message(msg)

    server.quit()

# =====================================================
# PREDICT IMAGE
# =====================================================

def predict_image(img_path):

    if model is None:

        return "Model tidak tersedia", 0

    try:

        # LOAD IMAGE
        img = image.load_img(
            img_path,
            target_size=(224, 224)
        )

        # IMAGE TO ARRAY
        img_array = image.img_to_array(img)

        # PREPROCESS
        img_array = preprocess_input(img_array)

        # EXPAND DIMENSION
        img_array = np.expand_dims(img_array, axis=0)

        # PREDICT
        pred = model.predict(img_array)

        # CLASS INDEX
        class_idx = np.argmax(pred)

        # CONFIDENCE
        confidence = float(np.max(pred)) * 100

        # FILTER NON DAUN
        if confidence < 70:

            return "Error: Gambar bukan daun kentang", round(confidence, 2)

        # LABEL
        result = labels.get(class_idx, "Unknown")

        return result, round(confidence, 2)

    except Exception as e:

        print("ERROR PREDICT:")
        print(e)

        return "Terjadi error saat prediksi", 0

# =====================================================
# LOGIN
# =====================================================

@app.route('/', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        username = request.form['username']

        password = request.form['password']

        conn = get_db_connection()

        c = conn.cursor()

        c.execute(
            "SELECT * FROM users WHERE username=?",
            (username,)
        )

        user = c.fetchone()

        conn.close()

        if user and check_password_hash(
            user['password'],
            password
        ):

            session['user'] = user['username']

            flash('Login berhasil', 'success')

            return redirect(url_for('home'))

        else:

            flash('Username atau password salah', 'error')

            return redirect(url_for('login'))

    return render_template('login.html')

# =====================================================
# REGISTER
# =====================================================

@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == 'POST':

        username = request.form['username']

        password = request.form['password']

        email = request.form['email']

        hashed_password = generate_password_hash(password)

        conn = get_db_connection()

        c = conn.cursor()

        c.execute(
            "SELECT * FROM users WHERE username=?",
            (username,)
        )

        existing_user = c.fetchone()

        if existing_user:

            conn.close()

            flash('Username sudah digunakan', 'error')

            return redirect(url_for('register'))

        c.execute(
            '''
            INSERT INTO users (
                username,
                password,
                email
            )
            VALUES (?, ?, ?)
            ''',
            (
                username,
                hashed_password,
                email
            )
        )

        conn.commit()

        conn.close()

        flash('Register berhasil', 'success')

        return redirect(url_for('login'))

    return render_template('register.html')

# =====================================================
# HOME
# =====================================================

@app.route('/home')
def home():

    if 'user' not in session:

        return redirect(url_for('login'))

    return render_template('home.html')

# =====================================================
# PROFILE
# =====================================================

@app.route('/profil', methods=['GET', 'POST'])
def profil():

    if 'user' not in session:

        return redirect(url_for('login'))

    conn = get_db_connection()

    c = conn.cursor()

    c.execute(
        "SELECT * FROM users WHERE username=?",
        (session['user'],)
    )

    user = c.fetchone()

    if request.method == 'POST':

        otp_input = request.form['otp']

        new_password = request.form['new_password']

        confirm_password = request.form['confirm_password']

        if otp_input != session.get('otp'):

            conn.close()

            return render_template(
                'profil.html',
                user=user,
                message='OTP salah',
                message_category='error'
            )

        if new_password != confirm_password:

            conn.close()

            return render_template(
                'profil.html',
                user=user,
                message='Konfirmasi password tidak cocok',
                message_category='error'
            )

        hashed_password = generate_password_hash(
            new_password
        )

        c.execute(
            '''
            UPDATE users
            SET password=?
            WHERE username=?
            ''',
            (
                hashed_password,
                session['user']
            )
        )

        conn.commit()

        conn.close()

        session.pop('otp', None)

        return render_template(
            'profil.html',
            user=user,
            message='Password berhasil diubah',
            message_category='success'
        )

    conn.close()

    return render_template(
        'profil.html',
        user=user
    )

# =====================================================
# SEND OTP
# =====================================================

@app.route('/send-otp', methods=['POST'])
def send_otp_route():

    if 'user' not in session:

        return redirect(url_for('login'))

    try:

        conn = get_db_connection()

        c = conn.cursor()

        c.execute(
            "SELECT * FROM users WHERE username=?",
            (session['user'],)
        )

        user = c.fetchone()

        conn.close()

        if not user:

            flash('User tidak ditemukan', 'error')

            return redirect(url_for('profil'))

        otp = str(random.randint(100000, 999999))

        session['otp'] = otp

        send_otp(user['email'], otp)

        flash('OTP berhasil dikirim ke email', 'success')

    except Exception as e:

        print(e)

        flash(f'Gagal mengirim OTP: {e}', 'error')

    return redirect(url_for('profil'))

# =====================================================
# DETECT
# =====================================================

@app.route('/detect', methods=['GET', 'POST'])
def detect():

    if 'user' not in session:

        return redirect(url_for('login'))

    if request.method == 'POST':

        if 'image' not in request.files:

            flash('File tidak ditemukan', 'error')

            return redirect(request.url)

        file = request.files['image']

        if file.filename == '':

            flash('File kosong', 'error')

            return redirect(request.url)

        if file and allowed_file(file.filename):

            ext = file.filename.rsplit('.', 1)[1].lower()

            filename = secure_filename(
                str(uuid.uuid4()) + "." + ext
            )

            filepath = os.path.join(
                app.config['UPLOAD_FOLDER'],
                filename
            )

            file.save(filepath)

            result, confidence = predict_image(filepath)

            conn = get_db_connection()

            c = conn.cursor()

            c.execute(
                '''
                INSERT INTO history (
                    username,
                    result,
                    image,
                    confidence
                )
                VALUES (?, ?, ?, ?)
                ''',
                (
                    session['user'],
                    result,
                    filename,
                    confidence
                )
            )

            conn.commit()

            conn.close()

            return render_template(
                'result.html',
                result=result,
                confidence=confidence,
                img=filepath
            )

        else:

            flash('Format file tidak didukung', 'error')

            return redirect(request.url)

    return render_template('detect.html')

# =====================================================
# HISTORY
# =====================================================

@app.route('/history')
def history():

    if 'user' not in session:

        return redirect(url_for('login'))

    conn = get_db_connection()

    c = conn.cursor()

    c.execute(
        '''
        SELECT * FROM history
        WHERE username=?
        ORDER BY id DESC
        ''',
        (session['user'],)
    )

    data = c.fetchall()

    conn.close()

    return render_template(
        'history.html',
        data=data
    )

# =====================================================
# LOGOUT
# =====================================================

@app.route('/logout')
def logout():

    session.pop('user', None)

    session.pop('otp', None)

    flash('Logout berhasil', 'success')

    return redirect(url_for('login'))

# =====================================================
# TEST ROUTE
# =====================================================

@app.route('/test')
def test():

    return "FLASK BERHASIL BERJALAN"

@app.route('/informasi')
def informasi():
    return render_template('informasi.html')
    
# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    print("PORT:", port)

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )