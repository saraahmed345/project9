from sklearn import preprocessing
import tkinter as tk
from tkinter import ttk
import numpy as np
import Perceptron as t
import Adaline as t2
def home():
    home=tk.Tk()
    home.geometry("600x500")
    home.title("Neural netwark task 1")
    l1=tk.Label(home,text=" Select two features")
    l1.pack(pady=5)
    l3 = tk.Label(home, text="feature number 1")
    l3.pack()
    def handle_selection1(event):
        selected_item = combobox1.get()
        return selected_item
    def handle_selection2(event):
        selected_item = combobox2.get()
        return selected_item
    def handle_selection3(event):
        selected_item = combobox3.get()
        return selected_item
    # Create a list of options for the combobox
    options = ["Area", "Perimeter", "MajorAxisLength","MinorAxisLength","roundnes"]
    # Create the combobox
    combobox1 = ttk.Combobox(home, values=options)
    combobox1.pack()
    # Set the default selection
    combobox1.current()
    l2 = tk.Label(home, text="feature number 2")
    l2.pack()
    combobox2 = ttk.Combobox(home, values=options)
    combobox2.pack()
    combobox1.bind("<<ComboboxSelected>>", handle_selection1)
    combobox2.bind("<<ComboboxSelected>>", handle_selection2)
    l2 = tk.Label(home, text="Select two classes ")
    l2.pack(pady=5)
    classes = ["c1&c2", "c1&c3", "c2&c3"]
    combobox3 = ttk.Combobox(home, values=classes)
    combobox3.pack()
    l4= tk.Label(home, text="Enter learning rate(eta)")
    l4.pack(pady=5)
    eta=tk.Text(home,width=20,height=1)
    eta.pack(padx=5)

    l5 = tk.Label(home, text="Enter number of epochs (m)")
    l5.pack(pady=5)
    m = tk.Text(home, width=20, height=1)
    m.pack(padx=5)
    l6 = tk.Label(home, text=" Enter MSE threshold (mse_threshold)")
    l6.pack(pady=5)
    mse = tk.Text(home, width=20, height=1)
    mse.pack(padx=5)
    # Set the default selection
    combobox3.current()
    Checkbutton1 = tk.IntVar()
    button1 = tk.Checkbutton(home, text="bais or not",
                         variable=Checkbutton1,
                         onvalue=1,
                         offvalue=0,
                         height=2,
                         width=10)
    button1.pack()
    v = tk.StringVar(home, "1")
     
    # Dictionary to create multiple buttons
    values = {"perceptron ": "1",
              "Adaline ": "2"}
    # Loop is used to create multiple Radiobuttons
    # rather than creating each button separately
    for (text, value) in values.items():
        tk.Radiobutton(home, text=text, variable=v,
                    value=value).pack(pady=5)
    def dotask():
        feature1=combobox1.get()
        feature2=combobox2.get()
        clas=combobox3.get()
        function = v.get()
        if clas=="c1&c2":
            c1="BOMBAY"
            c2="CALI"
        elif clas=="c1&c3":
            c1 = "BOMBAY"
            c2 = "SIRA"
        else:
            c1="CALI"
            c2="SIRA"
        et=eta.get("1.0", "end-1c")
        epochs=m.get("1.0", "end-1c")
        MSE=mse.get("1.0", "end-1c")
        bais=Checkbutton1.get()
        if function=="1":
            t.Perc(feature1,feature2,c1,c2,et,int(epochs),bais)
        elif function=="2":
            t2.Perc2(feature1, feature2, c1, c2,et,int(epochs), bais)
    btn=tk.Button(home,text="Do task",command=dotask)
    btn.pack()
    home.mainloop()
home()