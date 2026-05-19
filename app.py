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

    # GANTI DENGAN PASSWORD APLIKASI GMAIL
    MAIL_PASSWORD = 'password_aplikasi_gmail'


# =====================================================
# APP
# =====================================================

app = Flask(__name__)

app.config.from_object(Config)

# =====================================================
# EMAIL CONFIG
# =====================================================

EMAIL_ADDRESS = app.config['MAIL_USERNAME']
EMAIL_PASSWORD = app.config['MAIL_PASSWORD']

# =====================================================
# CREATE FOLDER
# =====================================================

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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

# =====================================================
# LOAD MODEL
# =====================================================

if not os.path.exists(app.config['MODEL_PATH']):

    print("Model tidak ditemukan!")

    model = None

else:

    model = load_model(app.config['MODEL_PATH'])

    print("Model berhasil dimuat")

# =====================================================
# LOAD LABEL
# =====================================================

with open('model/class_indices.json') as f:

    class_indices = json.load(f)

labels = dict((v, k) for k, v in class_indices.items())

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

    # ================= FILTER NON DAUN =================
    if confidence < 70:

        return "Error: Gambar bukan daun kentang", round(confidence, 2)

    # LABEL
    result = labels.get(class_idx, "Unknown")

    return result, round(confidence, 2)
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

        # CHECK USERNAME
        c.execute(
            "SELECT * FROM users WHERE username=?",
            (username,)
        )

        existing_user = c.fetchone()

        if existing_user:

            conn.close()

            flash('Username sudah digunakan', 'error')

            return redirect(url_for('register'))

        # INSERT USER
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

    # ================= UBAH PASSWORD =================

    if request.method == 'POST':

        otp_input = request.form['otp']

        new_password = request.form['new_password']

        confirm_password = request.form['confirm_password']

        # CEK OTP
        if otp_input != session.get('otp'):

            conn.close()

            return render_template(
                'profil.html',
                user=user,
                message='OTP salah',
                message_category='error'
            )

        # CEK PASSWORD
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

        # UPDATE PASSWORD
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

        # GENERATE OTP
        otp = str(random.randint(100000, 999999))

        # SIMPAN OTP
        session['otp'] = otp

        print("OTP:", otp)

        print("EMAIL:", user['email'])

        # KIRIM EMAIL
        send_otp(user['email'], otp)

        flash('OTP berhasil dikirim ke email', 'success')

    except Exception as e:

        print("ERROR OTP:")
        print(e)

        flash(f'Gagal mengirim OTP: {e}', 'error')

    return redirect(url_for('profil'))

# =====================================================
# RESET PASSWORD DARI LOGIN
# =====================================================

@app.route('/reset-password', methods=['POST'])
def reset_password():

    email = request.form['email']

    conn = get_db_connection()

    c = conn.cursor()

    c.execute(
        "SELECT * FROM users WHERE email=?",
        (email,)
    )

    user = c.fetchone()

    conn.close()

    if not user:

        flash('Email tidak ditemukan', 'error')

        return redirect(url_for('login'))

    otp = str(random.randint(100000, 999999))

    session['reset_otp'] = otp

    try:

        send_otp(email, otp)

        flash('OTP berhasil dikirim ke email', 'success')

    except Exception as e:

        flash(f'Gagal mengirim OTP: {e}', 'error')

    return redirect(url_for('login'))

# =====================================================
# DETECT
# =====================================================

@app.route('/detect', methods=['GET', 'POST'])
def detect():

    if 'user' not in session:

        return redirect(url_for('login'))

    if request.method == 'POST':

        # VALIDASI FILE
        if 'image' not in request.files:

            flash('File tidak ditemukan', 'error')

            return redirect(request.url)

        file = request.files['image']

        # FILE KOSONG
        if file.filename == '':

            flash('File kosong', 'error')

            return redirect(request.url)

        # VALIDASI FORMAT
        if file and allowed_file(file.filename):

            ext = file.filename.rsplit('.', 1)[1].lower()

            filename = secure_filename(
                str(uuid.uuid4()) + "." + ext
            )

            filepath = os.path.join(
                app.config['UPLOAD_FOLDER'],
                filename
            )

            # SAVE IMAGE
            file.save(filepath)

            # PREDICT
            result, confidence = predict_image(filepath)

            # SAVE HISTORY
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
# RUN APP
# =====================================================

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )