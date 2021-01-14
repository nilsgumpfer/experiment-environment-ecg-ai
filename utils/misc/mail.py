import mimetypes
import smtplib
import ssl
from email.message import EmailMessage


def send_file_via_email(filepath, recipient):
    smtp_server = 'smtp.strato.de'
    port = 465
    sender = 'experiments@nilsgumpfer.com'
    password = '4b868e70d26138ef6469e446d1'
    context = ssl.create_default_context()

    # TODO: consider email size restriction!

    splt = str(filepath).split('/')
    splt.reverse()
    filename = splt[0]

    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender, password)

        msg = EmailMessage()
        msg['Subject'] = 'File from Yamata "{}"'.format(filename)
        msg['To'] = recipient
        msg['From'] = sender

        msgcontent = filepath
        msg.set_content(msgcontent)

        ctype, encoding = mimetypes.guess_type(filepath)

        if ctype is None or encoding is not None:
            ctype = 'application/octet-stream'

        maintype, subtype = ctype.split('/', 1)

        with open(filepath, 'rb') as fp:
            msg.add_attachment(fp.read(),
                               maintype=maintype,
                               subtype=subtype,
                               filename=filename)

        server.send_message(msg)

    print('File sent successfully to {}'.format(recipient))