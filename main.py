import tkinter as tk
from tkinter import ttk
from GUI import ImageProcessorApp


def main():
    root = tk.Tk()
    style = ttk.Style()
    style.configure('TButton', font=('Arial', 10), padding=6)
    style.map('TButton', background=[('active', 'lightgray')])
    
    app = ImageProcessorApp(root)
    
    root.mainloop()

if __name__ == "__main__":
    main()
