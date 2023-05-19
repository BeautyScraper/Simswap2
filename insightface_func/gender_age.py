import insightface

def detect_age_gender(image_path):
    # Load the model for age and gender detection
    age_gender_model = insightface.model_zoo.get_model('RetinaFace-500MF')
    age_gender_model.prepare(ctx_id = -1)
    
    # Load the image
    img = insightface.data.image.read(image_path)
    # Get the face bounding box
    bbox, landmarks = age_gender_model.detect(img)
    if bbox.shape[0]==0:
        return None
    # Get the gender and age predictions
    age, gender = age_gender_model.predict(img, bbox[0], landmarks[0])
    age = age[0][0]
    gender = gender[0][0]
    # Get the gender label
    if gender > 0.5:
        gender_label = 'Male'
    else:
        gender_label = 'Female'
    return (age, gender_label)
