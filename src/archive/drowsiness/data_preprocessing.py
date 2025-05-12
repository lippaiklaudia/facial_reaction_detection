import os
import cv2
import numpy as np

base_dir = "data/drowsiness"
output_dir = "data/drowsiness_processed"
categories = ["Closed", "Open", "yawn", "no_yawn"]
img_size = 64 

# Adatok előfeldolgozása
def preprocess_and_save_images(base_dir, output_dir, categories, img_size):
    for split in ["train", "test"]:  # Train és Test mappák feldolgozása
        input_path = os.path.join(base_dir, split)
        output_path = os.path.join(output_dir, split)
        os.makedirs(output_path, exist_ok=True)
        
        for category in categories:
            input_category_path = os.path.join(input_path, category)
            output_category_path = os.path.join(output_path, category)
            os.makedirs(output_category_path, exist_ok=True)

            for img_name in os.listdir(input_category_path):
                try:
                    # Kép betöltése
                    img_path = os.path.join(input_category_path, img_name)
                    img = cv2.imread(img_path)
                    
                    # Szürkeárnyalatos konverzió
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Zajmentesítés GaussianBlur használatával
                    denoised_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
                    
                    # Átméretezés
                    resized_img = cv2.resize(denoised_img, (img_size, img_size))
                    
                    # Normalizálás (értékek 0-1 közé hozása)
                    normalized_img = resized_img / 255.0

                    # Mentés a feldolgozott mappába
                    save_path = os.path.join(output_category_path, img_name)
                    cv2.imwrite(save_path, (normalized_img * 255).astype(np.uint8))
                
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")

# Előfeldolgozás futtatása
preprocess_and_save_images(base_dir, output_dir, categories, img_size)

print("Előfeldolgozás befejzve. A feldolgozott képek a 'data/processed/' mappában találhatók.")
