import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np 
from canny import canny_algorithm 

class ImageProcessorApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Canny Edge Detection")
        self.root.geometry("1200x850") 
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.image_file_path = tk.StringVar() 
        self.umbral_bajo = tk.DoubleVar(value=20.0) # Low Threshold
        self.umbral_alto = tk.DoubleVar(value=40.0) # High Threshold
        self.sigma = tk.DoubleVar(value=1.75) 
        
        self.current_image_list = [] 
        self.image_refs = []
        
        # Variables to store the processing results (PIL.Image objects)
        self.processed_images = {
            'smoothed': None,
            'thin_edges': None,
            'final_edges': None
        }
        
        self.setup_ui()
        
    def setup_ui(self):         
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Canny Edge Detection", 
                                 font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Image Upload Section (Row 1)
        self.setup_image_upload_section(main_frame)
        
        # Parameters Section (Row 2)
        self.setup_parameters_section(main_frame)
        
        # Image Display Section (Row 3)
        self.setup_image_section(main_frame)
        
        # EXECUTE Button Section (Row 4)
        execute_btn = ttk.Button(main_frame, text="EXECUTE ALGORITHM", 
                                 command=self.execute_processing,
                                 cursor="hand2")
        execute_btn.grid(row=4, column=0, columnspan=3, pady=(10, 5), sticky=(tk.W, tk.E))
        
        # EXPORT Button Section (Row 5)
        self.export_btn = ttk.Button(main_frame, text="EXPORT IMAGES", 
                                     command=self.export_images,
                                     cursor="hand2",
                                     state='disabled') # Initially disabled
        self.export_btn.grid(row=5, column=0, columnspan=3, pady=(5, 10), sticky=(tk.W, tk.E))
        
        # Status bar (Row 6)
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                                 relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), 
                         pady=(20, 0))
    
    def setup_image_upload_section(self, parent):         
        # Image section frame
        image_frame = ttk.LabelFrame(parent, text="Image Upload", padding="10")
        image_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), 
                          pady=(0, 10))
        image_frame.columnconfigure(1, weight=1)
        
        # File selection
        ttk.Label(image_frame, text="Image File:").grid(row=0, column=0, 
                                                         sticky=tk.W, padx=(0, 10))
        
        self.file_entry = ttk.Entry(image_frame, textvariable=self.image_file_path, 
                                     state='readonly', width=50)
        self.file_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        
        # Button 1: Browse 
        browse_btn = ttk.Button(image_frame, text="Browse", command=self.browse_file)
        browse_btn.grid(row=0, column=2, padx=(0, 10))
        
        # Button 2: Load Image 
        self.upload_btn = ttk.Button(image_frame, text="Load Image", 
                                     command=self.load_selected_image, 
                                     state='disabled')
        self.upload_btn.grid(row=0, column=3)
    
    def setup_parameters_section(self, parent):
        """Create parameters control section"""
        
        # Parameters section frame
        params_frame = ttk.LabelFrame(parent, text="Canny Parameters (T_low/T_high in % of max gradient)", padding="10")
        params_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), 
                              pady=(0, 10))
        params_frame.columnconfigure(1, weight=1)
        params_frame.columnconfigure(3, weight=1)
        
        # LOW THRESHOLD (UMBRAL_BAJO)
        ttk.Label(params_frame, text="Low Threshold:").grid(row=0, column=0, 
                                                             sticky=tk.W, padx=(0, 10))
        
        self.umbral_bajo_entry = ttk.Entry(params_frame, textvariable=self.umbral_bajo, 
                                             width=15, validate='key')
        self.umbral_bajo_entry['validatecommand'] = (self.root.register(self.validate_float), '%P')
        self.umbral_bajo_entry.grid(row=0, column=1, sticky=tk.W, padx=(0, 20))
        
        # HIGH THRESHOLD (UMBRAL_ALTO)
        ttk.Label(params_frame, text="High Threshold:").grid(row=0, column=2, 
                                                             sticky=tk.W, padx=(0, 10))
        
        self.umbral_alto_entry = ttk.Entry(params_frame, textvariable=self.umbral_alto, 
                                             width=15, validate='key')
        self.umbral_alto_entry['validatecommand'] = (self.root.register(self.validate_float), '%P')
        self.umbral_alto_entry.grid(row=0, column=3, sticky=tk.W)
        
        # SIGMA Slider
        ttk.Label(params_frame, text="SIGMA:").grid(row=1, column=0, 
                                                         sticky=tk.W, padx=(0, 10), pady=(10, 0))
        
        slider_frame = ttk.Frame(params_frame)
        slider_frame.grid(row=1, column=1, columnspan=3, sticky=(tk.W, tk.E), 
                              pady=(10, 0))
        slider_frame.columnconfigure(0, weight=1)
        
        self.sigma_scale = ttk.Scale(slider_frame, from_=0.1, to=5.0, 
                                     variable=self.sigma, orient=tk.HORIZONTAL,
                                     command=self.update_sigma_label)
        self.sigma_scale.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.sigma_label = ttk.Label(slider_frame, text=f"Value: {self.sigma.get():.2f}")
        self.sigma_label.grid(row=0, column=1)
    
    def setup_image_section(self, parent):         
        # Images section frame
        images_frame = ttk.LabelFrame(parent, text="Canny Stages", padding="10")
        images_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), 
                          pady=(0, 10))
        
        # Configure grid for 4 images
        for i in range(4):
            images_frame.columnconfigure(i, weight=1)
        images_frame.rowconfigure(1, weight=1)
        
        # Create 4 image display areas
        self.image_labels = []
        titles = ["Original Image", "1. Gaussian", "2. Non-Maximum", "3. Hysteresis"]
        for i in range(4):
            # Image title
            title_label = ttk.Label(images_frame, text=titles[i], 
                                     font=('Arial', 10, 'bold'))
            title_label.grid(row=0, column=i, pady=(0, 5))
            
            # Image display area
            image_frame = tk.Frame(images_frame, width=200, height=200, 
                                     bg='white', relief=tk.SUNKEN, bd=2)
            image_frame.grid(row=1, column=i, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))
            image_frame.grid_propagate(False)
            
            # Image label
            image_label = ttk.Label(image_frame, text=titles[i] if i > 0 else "No Image", 
                                     background='white', anchor=tk.CENTER)
            image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            
            self.image_labels.append(image_label)

    def _update_image_slot(self, slot_index, img_pil, title):
        label = self.image_labels[slot_index]
        
        # Clear existing content
        label.config(image="", text=title if img_pil is None else "")
        
        if img_pil is None:
            return

        try:
            image = img_pil.copy()
            # Resize image to fit display area
            image.thumbnail((190, 190), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Store reference (critical for Tkinter/Pillow to prevent garbage collection)
            if slot_index < len(self.image_refs):
                self.image_refs[slot_index] = photo
            else:
                self.image_refs.append(photo)
            
            # Update label
            label.config(image=photo, text="")
            label.image = photo
            
        except Exception as e:
            print(f"Error displaying image in slot {slot_index}: {e}")
            label.config(image="", text=f"Error displaying\n{title}")

    def validate_float(self, value):
        """Validate float input"""
        if value == "":
            return True
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def update_sigma_label(self, event=None):
        """Update sigma value label"""
        self.sigma_label.config(text=f"Value: {self.sigma.get():.2f}")
    
    def browse_file(self):
        """Open file dialog to select image path and enable Load button."""
        
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.gif', '*.tiff', '*.webp']
        
        filetypes = [
            ('Image files', ' '.join(image_extensions)),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=filetypes
        )
        
        if filename:
            self.image_file_path.set(filename)
            self.upload_btn.config(state='normal')
            self.status_var.set(f"Selected: {os.path.basename(filename)}. Click 'Load Image' to display.")
        else:
            self.image_file_path.set("")
            self.upload_btn.config(state='disabled')
            self.status_var.set("Ready")
            
    def load_selected_image(self):
        """Loads and displays the image file specified in the entry."""
        image_path = self.image_file_path.get()
        
        if not image_path or not os.path.exists(image_path):
            # WARNING/ERROR GUI Text
            messagebox.showerror("Error", "Please select a valid image file path.")
            self.status_var.set("Load failed: Invalid path.")
            self.upload_btn.config(state='disabled')
            return

        try:
            self.status_var.set("Loading image...")
            self.root.update()

            self.current_image_list = [image_path] 
            
            self.display_images()
            
            self.status_var.set(f"Image loaded: {os.path.basename(image_path)}")
            self.upload_btn.config(state='disabled') 
            
            # Clear previous results and disable export
            self.processed_images = {'smoothed': None, 'thin_edges': None, 'final_edges': None}
            self.export_btn.config(state='disabled')
            
        except Exception as e:
            # WARNING/ERROR GUI Text
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            self.status_var.set("Load failed")
            self.current_image_list = []
            self.display_images() 

    def display_images(self):
        """Loads and displays the original image only (slot 1). Clears others."""
        titles = ["Original Image", "1. Gaussian", "2. Non-Maximum", "3. Hysteresis"]

        # Clear all 4 display slots
        for i in range(4):
              self._update_image_slot(i, None, titles[i] if i > 0 else "No Image")
        
        # Display the original image if available
        if self.current_image_list:
            image_path = self.current_image_list[0]
            try:
                img_pil = Image.open(image_path)
                self._update_image_slot(0, img_pil, titles[0])
            except Exception as e:
                print(f"Error loading original image: {e}")
                self.image_labels[0].config(image="", text=f"Error loading\n{os.path.basename(image_path)}")

    def get_parameters(self):
        """Get current parameter values"""
        return {
            'umbral_bajo': self.umbral_bajo.get(),
            'umbral_alto': self.umbral_alto.get(),
            'sigma': self.sigma.get()
        }

    def execute_processing(self):
        """Handles the execution of the image processing task using the Canny algorithm."""
        
        if not self.current_image_list:
            # WARNING/ERROR GUI Text
            messagebox.showwarning("Warning", "No image loaded. Please select and upload an image file before executing.")
            return

        # Clear previous results and disable export
        self.processed_images = {'smoothed': None, 'thin_edges': None, 'final_edges': None}
        self.export_btn.config(state='disabled')
        
        # 2. Get parameters and image path
        params = self.get_parameters()
        image_path = self.image_file_path.get()
        
        image_name = os.path.basename(image_path)
        # Status message
        self.status_var.set(f"Executing Canny Edge Detection on {image_name}...")
        self.root.update()
        
        try:
            
            img_smooth, img_thin, img_final = canny_algorithm(
                image_path,
                sigma=params['sigma'], 
                T_low=params['umbral_bajo'], 
                T_high=params['umbral_alto']
            )

            # conversion to PIL.Image in case canny.py returns NumPy arrays
            imagen_suavizada = self.convert_to_pil(img_smooth)
            bordes_delgados = self.convert_to_pil(img_thin)
            bordes_finales = self.convert_to_pil(img_final)

            # 3.5 Store results
            self.processed_images['smoothed'] = imagen_suavizada
            self.processed_images['thin_edges'] = bordes_delgados
            self.processed_images['final_edges'] = bordes_finales
            
            # 4. Display results in slots 2, 3, and 4
            self._update_image_slot(1, imagen_suavizada, "1. Smoothed Image")
            self._update_image_slot(2, bordes_delgados, "2. Thin Edges (NMS)")
            self._update_image_slot(3, bordes_finales, "3. Final Edges (Hysteresis)")
            
            # Status message
            self.status_var.set(f"CANNY processing complete on {image_name}. Ready to export.")
            self.export_btn.config(state='normal') # Enable export button

        except ImportError:
              # WARNING/ERROR GUI Text
              error_message = f"Import Error: The module 'canny' was not found. Ensure your 'canny.py' file with the 'canny_algorithm' function is in the same directory."
              messagebox.showerror("Module Error", error_message)
              self.status_var.set("Error: 'canny' module not found.")
              print(error_message)
              
        except Exception as e:
            # WARNING/ERROR GUI Text
            error_message = f"Error during processing: {str(e)}"
            messagebox.showerror("Processing Error", error_message)
            self.status_var.set(f"Error: {str(e)}")
            print(error_message)
            self.processed_images = {'smoothed': None, 'thin_edges': None, 'final_edges': None}
            self.export_btn.config(state='disabled')

    def convert_to_pil(self, img):
        """Defensive conversion: Ensures input is a PIL Image if it's a NumPy array."""
        if isinstance(img, np.ndarray):
            return Image.fromarray(img.astype(np.uint8))
        return img

    def export_images(self):
        """Allows the user to select a directory and exports the 3 processed images."""
        
        # Check if there are results to export
        if not self.processed_images['final_edges']:
            # WARNING/ERROR GUI Text
            messagebox.showwarning("Warning", "No Canny results to export. Please execute the algorithm first.")
            self.status_var.set("Export failed: No data.")
            return
            
        # Ask the user for the directory to save to
        save_dir = filedialog.askdirectory(title="Select Folder to Export Images")
        
        if not save_dir:
            self.status_var.set("Export canceled by the user.")
            return

        try:
            # Use the original file name to name the exported files
            base_filename = os.path.basename(self.current_image_list[0]).split('.')[0]
            
            images_to_save = {
                'Smoothed': self.processed_images['smoothed'],
                'Thin_Edges': self.processed_images['thin_edges'],
                'Final_Edges': self.processed_images['final_edges']
            }
            
            saved_files_count = 0
            
            for name, img_pil in images_to_save.items():
                if img_pil:
                    # Naming format: [ORIGINAL_NAME]_Canny_[StageName].png
                    filename = f"{base_filename}_Canny_{name}.png"
                    save_path = os.path.join(save_dir, filename)
                    img_pil.save(save_path) 
                    saved_files_count += 1

            # Success GUI Text
            messagebox.showinfo("Success", f"Images successfully exported to:\n{save_dir}\n\nFiles saved: {saved_files_count}")
            self.status_var.set(f"Export completed. {saved_files_count} files saved in: {os.path.basename(save_dir)}")
            
        except Exception as e:
            # WARNING/ERROR GUI Text
            messagebox.showerror("Export Error", f"Failed to export images: {str(e)}")
            self.status_var.set("Export error.")
            print(f"Export error: {e}")