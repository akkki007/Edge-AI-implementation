import cv2
import numpy as np
import os
import google.generativeai as genai
from PIL import Image

def load_image(image_path):
    # Load the image from the specified path
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not open or find the image.")
    return image

def process_image_with_gemini(image_path):  
    api_key = os.getenv('GOOGLE_API_KEY')  
    if not api_key:
        raise ValueError("API key not found. Please set the GOOGLE_API_KEY environment variable.")
    
    genai.configure(api_key=api_key)
    image = Image.open(image_path)
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(["Describe this image and provide any relevant information and identify harmful objects and point out: ", image])
    return response.text

def analyze_gemini_response(response):
    harmful_objects = ["gun", "knife", "weapon", "alcohol", "drug"]
    response = response.lower()
    for obj in harmful_objects:
        if obj in response:
            return False
    return True

def display_alert(is_safe, image):
    if is_safe:
        cv2.rectangle(image, (0, 0), (100, 100), (0, 255, 0), -1)
        cv2.putText(image, "Safe", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    else:
        cv2.rectangle(image, (0, 0), (100, 100), (0, 0, 255), -1)
        cv2.putText(image, "Alert!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Alert", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Prompt user to enter the image path
    image_path = input("Enter the path of the image to be processed: ")

    # Load the image
    image = load_image(image_path)

    if image is not None:
        cv2.imshow("Loaded Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Process the image with Gemini
        print("Processing image with Gemini...")
        result = process_image_with_gemini(image_path)
        print("Gemini Analysis:")
        print(result)

        # Analyze the Gemini response
        is_safe = analyze_gemini_response(result)

        # Display alert
        display_alert(is_safe, image)
    else:
        print("Failed to load image.")

if __name__ == "__main__":
    main()
