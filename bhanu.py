import sys
import smtplib
from email.message import EmailMessage

msg = EmailMessage()
msg['Subject'] = 'hi'
msg['From'] = 'ankojubhan20@iitk.ac.in'
msg['To'] = 'ankojubhanuprakash@gmail.com'

msg_id = str(sys.argv[1])
if msg_id== '0':
  msg.set_content("""\
  drink water \n make juice \n soak nuts \n waterplants \n sow wheat grass
                  """)
elif  msg_id== '1':
  msg.set_content("""\
    exercise,bath and eat curd ice with soak nuts
                  """)
elif  msg_id== '2':
  msg.set_content("""\ 
    Study till from 8 to one 
                  """)
elif  msg_id== '3':
  msg.set_content("""\
    Eat lunch at from 1 
                  """)
elif  msg_id== '4':
  msg.set_content("""\
   study from 2:30to 7 
                  """)
elif  msg_id== '5':
  msg.set_content("""\
  eat dinner at 7
                  """)
elif  msg_id== '6':
  msg.set_content("""\
   read GIta from 9 to 10   and sleep
                  """)
elif  msg_id== '7':
  msg.set_content("""\
   sleep at 10 
                  """)                  


smtpObj = smtplib.SMTP('mmtp.iitk.ac.in',25)
smtpObj.ehlo()
smtpObj.starttls()

smtpObj.login(msg['From'] , 'Bhanu@1997')
print(smtpObj.send_message(msg))
smtpObj.quit()
