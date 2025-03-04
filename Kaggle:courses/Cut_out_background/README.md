# First Experience in Computer Vision

**Task**: Replace the background in photographs.

## Project Stages:
1. **Segmentation**: Perform image segmentation to isolate the person from the background using a pre-trained model.
2. **Original Background Removal**: Remove the original background from the segmented image.
3. **New Background Addition**: Add a new background to the image, ensuring a seamless blend with the foreground.

**Libraries Used**:
- Torch, torchvision, PIL, numpy

For segmentation, a pre-trained model from Hugging Face (HF) was utilized. This model efficiently isolates the subject in the image, allowing for a clean removal of the background. 

### Results:
The project successfully demonstrated the ability to replace backgrounds in images, showcasing the flexibility and fun of computer vision techniques. Examples of the original and background-replaced images are attached as .jpg files.

### Conclusion:
This project serves as a foundational experience in computer vision, highlighting the steps involved in image processing and the use of deep learning models for practical applications.