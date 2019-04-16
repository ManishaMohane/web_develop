from flask import Flask, render_template, redirect, url_for
from flask_mail import Mail,  Message

app = Flask(__name__)
app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=465,
    MAIL_USE_SSL=True,
    MAIL_USERNAME = 'manishamohane54@gmail.com',
    MAIL_PASSWORD = 'xscu iqbj Wecn ailn'
)

mail = Mail(app)

@app.route('/send_mail')
def send_mail():
    msg = mail.send_message(
        'Send Mail tutorial!',
        sender='manishamohane54@gmail.com',
        recipients=['manishamohane54@gmail.com'],
        body="Congratulations you've succeeded!"
    )
    return 'Mail sent'

if __name__ == '__main__':
 app.run(debug=True)
