import sys
import smtplib
from email.message import EmailMessage

msg = EmailMessage()
msg['Subject'] = '[Reminder]Summer_Term_Thesis_discussion'
msg['From'] = 'ankojubhan20@iitk.ac.in'
msg['To'] = 'devpriya@iitk.ac.in'
msg['Cc'] = ['ankojubhan20@iitk.ac.in','devpriyak@gmail.com','rlokhande20@iitk.ac.in', 'shivanirp20@iitk.ac.in' ]
msg['Bcc'] = 'ankojubhanuprakash@gmail.com'

#msg_id = str(sys.argv[1])
#if msg_id== '0':
msg.set_content("""\
 A gentle reminder \n Time : 10:00 AM \n Zoom Link : https://zoom.us/j/97437365347?pwd=Kzl6dXNVS3VLMUs3Um5EWVVGdTVTZz09 \n 
 **A system generated email 
                  """)
                


smtpObj = smtplib.SMTP('mmtp.iitk.ac.in',25)
smtpObj.ehlo()
smtpObj.starttls()

smtpObj.login(msg['From'] , 'Bhanu@1997')
print(smtpObj.send_message(msg))
smtpObj.quit()
