# Creating GUI Using Tkinter library

# This is one of the project i've redesigned..
# You can find the original project in GeeksForGeeks site...
# if you have any problem running or understanding this project,
# feel free to contact me...



# imports
import tkinter
from tkinter import *
import datetime
from chatapp import *
from chatbot import *

def send():
    msg = EntryBox.get("1.0",'end-lc').strip()
    EntryBox.delete("0.0",END)


    if msg != "":
        time = datetime.time
        chatlog.config(state=NORMAL)
        chatlog.insert(END, "You: "+ msg+ "at"+ time+ "\n\n" )
        chatlog.config(foreground="#442265",font=("Verdana",12))

        time = datetime.time
        res = chatBot_response(msg)
        chatlog.insert(END, "Bot: "+res+ "at"+ time+ "\n\n")

        chatlog.config(state=DISABLED)
        chatlog.yview(END)



base = Tk()
base.title("Amir ChatBot")
base.geometry("400x500")
base.resizable(width=False,height=False)

# Create Chat window
chatlog = Text(base,bd=0,bg="white",height="8",width="50",font="Arial",relief=SUNKEN)
chatlog.config(state=DISABLED)                      # ???

# bind scrollbar to chat window
scrollbar = Scrollbar(base,command=chatlog.yview,cursor='heart')
chatlog['yscrollcommand'] = scrollbar.set

# Create Text Box
EntryBox = Text(base,bd=0,bg="white",width="29",height="5",font="Arial")


# Send_Button
Send_Button = Button(base,font=('verdana',12,'bold'),text="Send",width="12",height="5",
                     bd=0,bg="#32de97",activebackground="#3c9d9b",fg="#ffffff",
                     command=send)

# place everything
scrollbar.place(x=376,y=6,height=386)
chatlog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
Send_Button.place(x=6, y=401, height=90)

# Let's Do It
base.mainloop()