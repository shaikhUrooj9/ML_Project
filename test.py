import os
import numpy as np
from inference import get_onnx_session, preprocess_image, load_labels
from PIL import Image

def run_batch_testing():
    # Load labels and session using our inference module functions
    labels = load_labels()
    session = get_onnx_session()
    
    if session is None:
        print(" Error: model.onnx not found. Please pull it from Colab first.")
        return

    input_name = session.get_inputs()[0].name
    test_images = ['sample1.jpg', 'sample2.jpg', 'sample3.jpg', 'sample4.jpg']
    
    print("\n=== Running Project Task 1: Batch Testing (4 Samples) ===")
    for img_path in test_images:
        # Generate temporary mock images if testing files don't exist yet
        if not os.path.exists(img_path):
            Image.new('RGB', (224, 224), color='gray').save(img_path)
            
        # Process and predict
        input_data = preprocess_image(img_path)
        raw_outputs = session.run(None, {input_name: input_data})
        predicted_index = np.argmax(raw_outputs[0])
        
        print(f"📷 Test Image: {os.path.basename(img_path)}  Model Prediction Result: {labels[predicted_index]}")

if __name__ == "__main__":
    run_batch_testing()